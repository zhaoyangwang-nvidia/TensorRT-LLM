# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional fixed-size message channel over DOCA RDMA CPU_PROXY.

The izzy ``doca_rdma`` extension only drives ``RDMA_WRITE`` (no IMM)
reliably under CPU_PROXY mode; ``RDMA_WRITE_IMM`` requires posted recv
WRs on the responder, which the external-datapath QP does not yet
expose. This channel works around that by carrying the message in a
short two-step write sequence:

1. The 96-byte ``DraftApiProtocol`` payload is written into bytes
   ``[0, 96)`` of the responder's recv buffer.
2. A single ``uint32`` "signal word" is written into bytes
   ``[96, 100)``. Each new message increments the value.

Reliable-Connection ordering guarantees that the responder sees write
(2) only after write (1) is committed to memory, so the receiver can
spin on the signal word and, once it changes, read the payload knowing
it is fully landed. There is no IMM, no recv-WR posting, and no CQ
poll on the critical path.

Threading model: one host thread per endpoint runs
``cpu_proxy_progress`` to drive the NIC doorbell; everything else is
synchronous from Python.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import torch

_MSG_BYTES = 96
_SIGNAL_BYTES = 4
# We logically use 100 bytes (96 payload + 4 signal) but ask the
# allocator for a full 4 KiB. izzy's ``alloc_doca_gpu_buffer`` packs
# small requests into a shared CUDA allocation; small distinct allocs
# end up at adjacent offsets within the same 4 KiB block, which causes
# the send and recv buffers to alias. Asking for a page each forces
# distinct CUDA allocations.
_BUF_BYTES = 4096


class DocaPollChannel:
    """One end of a bidirectional 96-byte message channel.

    Parameters
    ----------
    ep:
        Already initialised, connected ``DocaRdmaEndpoint``.
    peer:
        Peer name used in ``connect_to_peer`` / ``accept_peer``.
    gpu_id:
        CUDA device the buffers live on; the endpoint must already be on
        that device.
    """

    def __init__(self, ep, peer: str, gpu_id: int):
        self.ep = ep
        self.peer = peer
        self.gpu_id = gpu_id
        # Alloc both send and recv buffers ourselves (the endpoint API
        # exposes BufferInfo with addr/lkey/rkey for each); we want 128
        # bytes so the signal word has its own cache line.
        self.send_buf = ep.alloc_doca_gpu_buffer(_BUF_BYTES)
        self.recv_buf = ep.alloc_doca_gpu_buffer(_BUF_BYTES)
        assert (
            self.send_buf.addr + _BUF_BYTES <= self.recv_buf.addr
            or self.recv_buf.addr + _BUF_BYTES <= self.send_buf.addr
        ), "send_buf and recv_buf overlap — izzy alloc_doca_gpu_buffer packs small allocations"
        # Cache torch views so we can read/write payload and signal in
        # Python without re-creating tensors each call.
        self.send_payload = torch.ops.doca_rdma.wrap_gpu_memory(
            self.send_buf.addr, _BUF_BYTES, gpu_id
        ).view(torch.uint8)
        self.send_signal = torch.ops.doca_rdma.wrap_gpu_memory(
            self.send_buf.addr + _MSG_BYTES, _SIGNAL_BYTES, gpu_id
        ).view(torch.int32)
        self.recv_payload = torch.ops.doca_rdma.wrap_gpu_memory(
            self.recv_buf.addr, _BUF_BYTES, gpu_id
        ).view(torch.uint8)
        self.recv_signal = torch.ops.doca_rdma.wrap_gpu_memory(
            self.recv_buf.addr + _MSG_BYTES, _SIGNAL_BYTES, gpu_id
        ).view(torch.int32)
        # Zero everything once.
        self.send_payload.zero_()
        self.recv_payload.zero_()
        torch.cuda.synchronize()

        self._send_seq = 0
        self._last_recv_seq = 0
        self._progress_stop = threading.Event()
        self._progress_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def buffers(self):
        """Return ``(send_buf, recv_buf)`` for ``exchange_buffer_info``."""
        return self.send_buf, self.recv_buf

    def start_progress(self):
        """Spawn a daemon thread that drives ``cpu_proxy_progress``."""
        assert self._progress_thread is None

        def _run():
            while not self._progress_stop.is_set():
                self.ep.cpu_proxy_progress(self.peer)

        self._progress_thread = threading.Thread(target=_run, daemon=True)
        self._progress_thread.start()

    def stop(self):
        self._progress_stop.set()
        if self._progress_thread is not None:
            self._progress_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Send / recv
    # ------------------------------------------------------------------
    def send_msg(self, msg_bytes: bytes) -> None:
        """Push a 96-byte message to the peer.

        We issue a single 100-byte RDMA_WRITE that carries both the
        96-byte payload and a 4-byte trailing signal word. The MLX5
        responder commits bytes of one RDMA_WRITE in increasing-address
        order, so a peer that observes the signal word (highest 4
        bytes) is guaranteed the payload bytes are already in memory.
        """
        assert len(msg_bytes) == _MSG_BYTES, f"expected {_MSG_BYTES} bytes, got {len(msg_bytes)}"
        # Stage payload + signal in a single contiguous region.
        cpu = torch.frombuffer(bytearray(msg_bytes), dtype=torch.uint8)
        self.send_payload[:_MSG_BYTES].copy_(cpu)
        self._send_seq += 1
        self.send_signal.fill_(self._send_seq)
        torch.cuda.synchronize()

        remote = self.ep.get_remote_recv_buf(self.peer)
        local_send = self.ep.get_local_send_buf(self.peer)
        wqe_out = torch.zeros(1, dtype=torch.int64, device=f"cuda:{self.gpu_id}")
        torch.ops.doca_rdma.write_with_imm(
            self.ep.get_qp_dev_ptr(self.peer),
            local_send.addr,
            _BUF_BYTES,
            local_send.lkey,
            remote.addr,
            remote.rkey,
            0,  # IMM=0 → plain RDMA_WRITE
            wqe_out,
        )
        torch.cuda.synchronize()

    def recv_msg(self, timeout_s: float = 30.0, poll_interval_s: float = 0.0005) -> bytes:
        """Block until a new message arrives, then return its 96 bytes."""
        deadline = time.monotonic() + timeout_s
        while True:
            cur = int(self.recv_signal.cpu().item())
            if cur != self._last_recv_seq:
                self._last_recv_seq = cur
                return bytes(self.recv_payload[:_MSG_BYTES].cpu().tolist())
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"DocaPollChannel.recv_msg timed out after {timeout_s:.1f}s "
                    f"(last_seen={self._last_recv_seq})"
                )
            time.sleep(poll_interval_s)
