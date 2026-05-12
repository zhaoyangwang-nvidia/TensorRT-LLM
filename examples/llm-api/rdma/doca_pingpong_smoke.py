# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional ping-pong smoke for DocaPollChannel.

Two-process harness: one side runs as ``--role server`` (binds and
accepts), the other as ``--role client`` (connects). Each side
exchanges ``N`` messages with a deterministic 96-byte content built
from the round number, and verifies the peer echoed it back.

Run inside the same container with two GPUs / two NICs:

    # GPU 0 / mlx5_0 (server)
    GPU_ID=0 NIC=mlx5_0 PORT=47800 \
        python3 doca_pingpong_smoke.py --role server --rounds 4

    # GPU 1 / mlx5_1 (client)
    GPU_ID=1 NIC=mlx5_1 PORT=47800 HOST=127.0.0.1 \
        python3 doca_pingpong_smoke.py --role client --rounds 4
"""

import argparse
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
import torch  # noqa: E402

torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
import doca_rdma  # noqa: E402
from doca_poll_channel import DocaPollChannel  # noqa: E402


def _make_msg(role: str, seq: int) -> bytes:
    """96-byte deterministic message: ``b'PING'``/``b'PONG'`` + uint32 seq + 88 byte pad."""
    tag = b"PING" if role == "client" else b"PONG"
    return tag + seq.to_bytes(4, "little") + bytes(range(88))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["server", "client"], required=True)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    gpu = int(os.environ["GPU_ID"])
    nic = os.environ["NIC"]
    port = int(os.environ["PORT"])
    peer_name = "doca_pp"

    torch.cuda.set_device(gpu)
    ep = doca_rdma.DocaRdmaEndpoint()
    ep.init(gpu, nic)
    chan = DocaPollChannel(ep, peer_name, gpu)

    if args.role == "server":
        ep.accept_peer(port, peer_name)
        ep.exchange_buffer_info(peer_name, chan.send_buf, chan.recv_buf)
    else:
        host = os.environ["HOST"]
        ep.connect_to_peer(host, port, peer_name)
        ep.exchange_buffer_info(peer_name, chan.send_buf, chan.recv_buf)
    print(
        f"[{args.role}] connected, cpu_proxy_enabled={ep.cpu_proxy_enabled(peer_name)}",
        flush=True,
    )

    chan.start_progress()

    failures = 0
    for i in range(args.rounds):
        if args.role == "client":
            tx = _make_msg("client", i)
            chan.send_msg(tx)
            print(f"[client] sent #{i}", flush=True)
            rx = chan.recv_msg(timeout_s=10.0)
            expect = _make_msg("server", i)
            ok = rx == expect
            print(f"[client] recv #{i} ok={ok}", flush=True)
            failures += 0 if ok else 1
        else:
            rx = chan.recv_msg(timeout_s=30.0)
            expect = _make_msg("client", i)
            ok_rx = rx == expect
            print(f"[server] recv #{i} ok={ok_rx}", flush=True)
            tx = _make_msg("server", i)
            chan.send_msg(tx)
            print(f"[server] sent #{i}", flush=True)
            failures += 0 if ok_rx else 1

    chan.stop()
    print(f"[{args.role}] DONE failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
