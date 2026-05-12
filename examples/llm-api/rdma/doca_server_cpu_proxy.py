# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DOCA RDMA CPU_PROXY responder side smoke harness.

Companion to ``doca_client_cpu_proxy.py``. The pair validates that the
patched izzy doca_rdma extension correctly drives a CPU-proxied NIC
doorbell end-to-end on B200 + ConnectX-7 + DOCA 3.3.

Env:
    GPU_ID    integer CUDA device index (e.g. 0 or 1)
    NIC       DOCA / ibverbs device name (e.g. mlx5_0)
    PORT      TCP control-plane port used for QP exchange

Server: accepts one peer, exchanges buffer info, then drives the
CPU_PROXY progress tick in a background thread. After a fixed sleep the
recv buffer is copied to host and the first 32 bytes are printed so a
human (or the companion client) can spot-check that the RDMA WRITE
actually landed.
"""

import os
import threading
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
import torch

torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
import doca_rdma  # noqa: E402

GPU = int(os.environ["GPU_ID"])
NIC = os.environ["NIC"]
PORT = int(os.environ["PORT"])
PEER = "doca_smoke"

torch.cuda.set_device(GPU)
ep = doca_rdma.DocaRdmaEndpoint()
ep.init(GPU, NIC)
s_send = ep.alloc_doca_gpu_buffer(96)
s_recv = ep.alloc_doca_gpu_buffer(96)
print(f"[server] alloc OK recv=0x{s_recv.addr:x}", flush=True)
print(f"[server] listening port={PORT}", flush=True)
ep.accept_peer(PORT, PEER)
ep.exchange_buffer_info(PEER, s_send, s_recv)
print(f"[server] connected, cpu_proxy_enabled={ep.cpu_proxy_enabled(PEER)}", flush=True)

stop = threading.Event()


def progress() -> None:
    while not stop.is_set():
        ep.cpu_proxy_progress(PEER)


threading.Thread(target=progress, daemon=True).start()
print("[server] progress thread up", flush=True)

print("[server] sleeping 5s for inbound write...", flush=True)
time.sleep(5)

recv_t = torch.ops.doca_rdma.wrap_gpu_memory(s_recv.addr, 96, GPU)
v = recv_t.view(torch.uint8).cpu().tolist()
print(f"[server] recv buf first 32: {v[:32]}", flush=True)
expected = [0xA0 + i for i in range(8)]
print(f"[server] match expected pattern (0xa0..): {v[:8] == expected}", flush=True)
stop.set()
