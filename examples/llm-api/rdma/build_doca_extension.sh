#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 10 — build izzy's DOCA RDMA extension inside the TRT-LLM dev
# container, applying the local patches needed to make it work on:
#   - DOCA SDK 3.3.0 (ubuntu24.04 / x86_64)
#   - GCC 13 + CUDA 13 + Ubuntu 24.04 host
#   - B200 (sm_100) + ConnectX-7 + MLNX OFED 25.07
#
# The patches in ./izzy_patches/ are required on top of izzy upstream
# (pearl-izzy branch); they cover:
#   - ::listen namespace qualifier (GCC 13 name lookup)
#   - pybind11 LTO off (so CUDA fatbin survives the link)
#   - CPU_PROXY support: DBR via calloc(), cpu_proxy_progress /
#     poll_send_cq_cpu host APIs, post_recv impl, plain RDMA_WRITE
#     kernel branch.
#
# Usage:
#   build_doca_extension.sh <izzy-src-dir>
# Where <izzy-src-dir> is a writable checkout of izzy doca_rdma.

set -euo pipefail
SRC=${1:-/opt/doca_rdma_izzy}
BUILD=${SRC}/build
ARCH=${DOCA_RDMA_CUDA_ARCHS:-100}  # B200 = sm_100
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/izzy_patches"

if [ ! -d "${SRC}/src" ]; then
  echo "izzy source not found at ${SRC}/src" >&2
  exit 1
fi

# Apply patches idempotently — if patch is already applied, skip.
for p in "${PATCH_DIR}"/*.patch; do
  [ -e "$p" ] || continue
  if patch -d "${SRC}" -p1 --dry-run -R --silent < "$p" >/dev/null 2>&1; then
    echo "skip (already applied): $(basename "$p")"
    continue
  fi
  echo "apply: $(basename "$p")"
  patch -d "${SRC}" -p1 < "$p"
done

TORCH_CMAKE=$(python3 -c "import torch.utils; print(torch.utils.cmake_prefix_path)")
PYBIND_CMAKE=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

mkdir -p "${BUILD}"
cd "${BUILD}"
cmake .. \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE};${PYBIND_CMAKE}" \
  -DDOCA_RDMA_CUDA_ARCHS=${ARCH} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF

cmake --build . -j4
ls -la *.so *.a lpu_rdma_server 2>/dev/null || true
