<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DOCA CPU_PROXY smoke harness

This directory carries the patches and minimal scripts needed to drive
izzy's `doca_rdma` extension in **CPU_PROXY** mode, where the GPU
prepares WQEs but a host-side progress thread rings the NIC doorbell.
The combination unblocks GPU-initiated RDMA on hardware/firmware
combos where the GPU SM doorbell paths (`GPU_SM_DB`, `GPU_SM_BF`,
`GPU_SM_NO_DBR`) are not yet certified, e.g.

> ConnectX-7 + MLNX OFED 25.07 + DOCA 3.3 + B200 (sm_100)

## Files

| Path | Role |
|---|---|
| `izzy_patches/01-cpu-proxy-mode.patch` | Patch against izzy upstream (`pearl-izzy` branch) adding `cpu_proxy_*` host APIs, host-backed DBR, plain RDMA_WRITE branch, and the GCC 13 + pybind LTO build fixes |
| `build_doca_extension.sh` | Wrapper that applies patches in `izzy_patches/` then builds the extension |
| `doca_server_cpu_proxy.py` | Responder smoke (binds, exchanges buffer info, drives CPU progress, dumps recv buffer) |
| `doca_client_cpu_proxy.py` | Requester smoke (connects, issues one GPU-kernel `write_with_imm`, polls send CQ from CPU) |

## Prereqs

1. DOCA SDK 3.3 installed inside the container (apt repo `doca/3.3.0/ubuntu24.04`); see `docs/source/features/rdma-izzy-refactor/phase10_doca_cuda_graph.md` for the exact steps.
2. izzy `doca_rdma` source mounted somewhere writable (e.g. bind-mount `/opt/doca_rdma_izzy`). The patch is applied in-place.
3. `tensorrt_llm/devel` (or any container that has CUDA 13 + PyTorch + pybind11 + the DOCA SDK).
4. NIC reachable from inside the container — devices passed in via `--device /dev/infiniband/...`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`.

## Build

```bash
# inside the container
cd /path/to/tensorrt_llm/examples/llm-api/rdma
./build_doca_extension.sh /opt/doca_rdma_izzy
```

Outputs land in `/opt/doca_rdma_izzy/build/`:

* `doca_rdma.cpython-*.so` — pybind module (`DocaRdmaEndpoint`)
* `doca_rdma_ops.so` — PyTorch CUDA ops (`torch.ops.doca_rdma.*`)
* `libdoca_rdma_kernels.a`
* `lpu_rdma_server`

## Run (single-host, two-NIC loopback)

```bash
export GLUE_RDMA_ALLOW_PROXY_HANDLERS=1
export GLUE_RDMA_NIC_HANDLERS=1
export PYTHONPATH=/opt/doca_rdma_izzy/build

# responder on GPU 0 / mlx5_0
GPU_ID=0 NIC=mlx5_0 PORT=47770 \
  python3 doca_server_cpu_proxy.py &

sleep 4

# requester on GPU 1 / mlx5_1, plain RDMA_WRITE (no IMM)
GPU_ID=1 NIC=mlx5_1 PORT=47770 HOST=127.0.0.1 IMM_VALUE=0 \
  python3 doca_client_cpu_proxy.py
```

Pass criteria (responder log):

```
[server] recv buf first 32: [160, 161, 162, ..., 191]
[server] match expected pattern (0xa0..): True
```

Setting `IMM_VALUE=0xdeadbeef` switches the kernel to `RDMA_WRITE_IMM`.
That path completes the requester send CQ with an error CQE
(`syndrome=0x16` = RNR_RETRY_EXC_ERR) because the responder does not yet
have a working CPU-side recv-WR posting path — fixing that is the
follow-up tracked in `phase10_doca_cuda_graph.md`.

## What the patch changes

See `izzy_patches/01-cpu-proxy-mode.patch`'s leading commentary for the
authoritative list. In short:

* `CMakeLists.txt` — disable IPO/LTO so pybind link doesn't drop CUDA
  fatbin symbols.
* `include/doca_rdma_endpoint.h` — declare three new host-side methods.
* `src/doca_rdma_endpoint.cpp` — host-backed DBR (`calloc`) when the
  env opts into CPU_PROXY; implement `cpu_proxy_enabled`,
  `cpu_proxy_progress`, `poll_send_cq_cpu` (with `mlx5_err_cqe`
  decoding); replace the `post_recv` stub with a
  `doca_verbs_bridge_post_recv` call; `::listen` qualifier for GCC 13.
* `src/doca_rdma_torch_ops.cu` — when `imm_value == 0`, emit MLX5
  `RDMA_WRITE` (opcode `0x08`) instead of `RDMA_WRITE_IMM` (`0x09`).
* `src/pybind_module.cpp` — expose the three new methods to Python.
