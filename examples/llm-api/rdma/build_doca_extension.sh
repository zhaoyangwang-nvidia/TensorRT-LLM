#!/usr/bin/env bash
# Phase 10 — build izzy's DOCA RDMA extension inside the TRT-LLM dev container.
#
# Prereq:
#   - DOCA SDK 3.2.0 installed (see phase10_doca_cuda_graph.md for apt steps)
#   - izzy doca_rdma source mounted somewhere (e.g. /opt/doca_rdma_izzy)
#   - PyTorch + pybind11 importable
#
# Two source-level patches are required vs upstream izzy:
#   1. src/doca_rdma_endpoint.cpp ~line 1324: use ``::listen`` to avoid
#      name clash with DocaRdmaEndpoint::listen member.
#   2. CMakeLists.txt: after find_package(pybind11 REQUIRED), add
#      ``set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)`` so the pybind
#      module's link step doesn't gc fatbinData CUDA symbols.

set -euo pipefail
SRC=${1:-/opt/doca_rdma_izzy}
BUILD=${SRC}/build
ARCH=${DOCA_RDMA_CUDA_ARCHS:-100}  # B200 = sm_100

TORCH_CMAKE=$(python3 -c "import torch.utils; print(torch.utils.cmake_prefix_path)")
PYBIND_CMAKE=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

rm -rf "${BUILD}"
mkdir -p "${BUILD}"
cd "${BUILD}"
cmake .. \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE};${PYBIND_CMAKE}" \
  -DDOCA_RDMA_CUDA_ARCHS=${ARCH} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF

make -j4
ls -la *.so *.a lpu_rdma_server
