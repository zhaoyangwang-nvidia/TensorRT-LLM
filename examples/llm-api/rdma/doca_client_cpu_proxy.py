# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DOCA RDMA CPU_PROXY requester side smoke harness.

Companion to ``doca_server_cpu_proxy.py``. Issues one
``write_with_imm`` (or, with ``IMM_VALUE=0``, a plain RDMA_WRITE) via
the GPU kernel, then polls the send CQ from a CPU thread. Designed to
prove end-to-end that:

  1. the GPU kernel writes the WQE + sq_db producer counter
  2. the CPU progress thread observes the bump and rings the NIC
     doorbell through the patched CPU_PROXY path
  3. the NIC actually transmits and (for plain WRITE) the data lands in
     the responder's recv buffer

Env:
    GPU_ID    integer CUDA device index
    NIC       DOCA / ibverbs device name
    PORT      TCP control-plane port matching the server
    HOST      TCP host the server binds (e.g. 127.0.0.1 for one-box test)
    IMM_VALUE optional immediate value. ``0`` (default) selects opcode
              RDMA_WRITE (no IMM); any non-zero value selects
              RDMA_WRITE_IMM and triggers the responder-side recv-WR
              consumption path.
"""

import os
import subprocess
import threading
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
import torch

torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
import doca_rdma  # noqa: E402

GPU = int(os.environ["GPU_ID"])
NIC = os.environ["NIC"]
PORT = int(os.environ["PORT"])
HOST = os.environ["HOST"]
PEER = "doca_smoke"
IMM = int(os.environ.get("IMM_VALUE", "0"), 0)

torch.cuda.set_device(GPU)
ep = doca_rdma.DocaRdmaEndpoint()
ep.init(GPU, NIC)
c_send = ep.alloc_doca_gpu_buffer(96)
c_recv = ep.alloc_doca_gpu_buffer(96)
ep.connect_to_peer(HOST, PORT, PEER)
ep.exchange_buffer_info(PEER, c_send, c_recv)
print(f"[client] connected, cpu_proxy_enabled={ep.cpu_proxy_enabled(PEER)}", flush=True)

send_t = torch.ops.doca_rdma.wrap_gpu_memory(c_send.addr, 96, GPU)
pattern = torch.arange(96, dtype=torch.uint8, device=f"cuda:{GPU}").add_(0xA0).bitwise_and_(0xFF)
send_t.view(torch.uint8).copy_(pattern)
torch.cuda.synchronize()

stop = threading.Event()


def progress() -> None:
    while not stop.is_set():
        ep.cpu_proxy_progress(PEER)


threading.Thread(target=progress, daemon=True).start()
print("[client] progress thread up", flush=True)

remote = ep.get_remote_recv_buf(PEER)
local_send = ep.get_local_send_buf(PEER)
qp_ptr = ep.get_qp_dev_ptr(PEER)
wqe_out = torch.zeros(1, dtype=torch.int64, device=f"cuda:{GPU}")
print("[client] issuing write_with_imm", flush=True)
torch.ops.doca_rdma.write_with_imm(
    qp_ptr, local_send.addr, 96, local_send.lkey, remote.addr, remote.rkey, IMM, wqe_out
)
torch.cuda.synchronize()
print(f"[client] wqe_idx={wqe_out.item()}", flush=True)

cum = 0.0
for delay in [0.2, 0.5, 1.0, 2.0, 4.0]:
    time.sleep(delay)
    cum += delay
    n, packs = ep.poll_send_cq_cpu(PEER, 16)
    print(f"[client] after {cum:.1f}s send_cq poll: n={n}", flush=True)
    if n > 0:
        break

r = subprocess.run(["rdma", "statistic", "show", "link", NIC], capture_output=True, text=True)
for tok in r.stdout.split(","):
    t = tok.strip()
    if t.startswith("rx_write_requests") or "req_cqe_error" in t or "rnr_retr" in t:
        print(f"[client] NIC counter: {t}", flush=True)

stop.set()
print("[client] DONE", flush=True)
