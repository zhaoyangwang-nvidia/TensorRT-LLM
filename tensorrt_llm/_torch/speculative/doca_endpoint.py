# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Endpoint protocol implementation over DOCA RDMA + CPU_PROXY doorbell.

Third leg of the 4-phase draft-offload transport progression. Uses the
izzy ``doca_rdma`` pybind extension (must be built and present at
``--doca-rdma-build`` or ``$DOCA_RDMA_BUILD``, typically
``/opt/doca_rdma_izzy/build``).

Wire frame is 104 bytes total:

    bytes [0 :  4)   little-endian uint32 ``imm_data`` (routing)
    bytes [4 :100)   96-byte ``DraftApiProtocol`` payload
    bytes [100:104)  little-endian uint32 monotonic poll counter

``imm_data`` is carried in the frame rather than via mlx5
WRITE_WITH_IMM because the CPU_PROXY path on ConnectX-7 + DOCA 3.3 +
MLNX_OFED 25.07 cannot post recv WRs on the external-datapath QP (see
``phase10_doca_cuda_graph.md`` task #29). The poll counter is
incremented on each send so the receiver's spin-poll observes a unique
value — RC ordering guarantees the responder commits one RDMA_WRITE in
increasing-address order, so seeing a new counter value implies the
imm_data and payload bytes are already in memory.

A GPU kernel (``torch.ops.doca_rdma.write_with_imm`` with ``imm=0``,
selected by our izzy patch as plain ``RDMA_WRITE``) issues the WQE. A
daemon thread on each side drives ``cpu_proxy_progress`` so the NIC
doorbell rings without blocking the spec-decode engine.

Caveat: TRT-LLM's torch caching allocator and DOCA's
``doca_gpu_mem_alloc`` share the same CUDA virtual-address space and
can collide. See ``phase13_doca_cpu_proxy.md`` for the failure mode
and the mitigations (host-pinned DOCA buffers, allocator pre-warm,
two-process split).
"""

from __future__ import annotations

import os
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .draft_api_protocol import DraftApiProtocol
from .spec_decode_channel import EndpointPacket, EndpointStatus

_IMM_BYTES = 4
_SIGNAL_BYTES = 4
_FRAME_BYTES = _IMM_BYTES + DraftApiProtocol.kMessageBytes + _SIGNAL_BYTES  # = 104
# izzy's allocator packs small allocations into a shared CUDA region,
# which can alias send and recv buffers. Ask for a whole page each so
# the addresses are distinct.
_BUF_BYTES = 4096


@dataclass
class DocaEndpointConfig:
    """Connection config for ``_DocaEndpointBackend``.

    Mirrors ``IbverbsEndpointConfig`` so upper layers can swap transports
    by flipping a single string.
    """

    gpu_id: int = 0
    nic_name: str = "mlx5_0"
    is_server: bool = False
    remote_host: str = "127.0.0.1"
    remote_port: int = 47000
    bind_port: int = 0  # 0 ⇒ same as remote_port (single-side test)
    remote_peer_name: str = "draft_doca"
    recv_queue_depth: int = 256
    payload_bytes: int = DraftApiProtocol.kMessageBytes
    handshake_timeout_s: float = 180.0
    # Path to izzy's build dir (contains ``doca_rdma.cpython-*.so`` +
    # ``doca_rdma_ops.so``). Defaults to ``$DOCA_RDMA_BUILD`` or
    # ``/opt/doca_rdma_izzy/build`` if neither is set.
    izzy_build_dir: Optional[str] = None


def _load_izzy(build_dir: Optional[str]):
    """Import izzy's pybind module + load its torch ops .so."""
    path = build_dir or os.environ.get("DOCA_RDMA_BUILD") or "/opt/doca_rdma_izzy/build"
    if path and path not in sys.path:
        sys.path.insert(0, path)
    import torch

    ops_path = os.path.join(path, "doca_rdma_ops.so")
    if os.path.exists(ops_path):
        try:
            torch.ops.load_library(ops_path)
        except RuntimeError:
            # Already loaded — fine.
            pass
    import doca_rdma  # noqa: E402

    return doca_rdma


class _DocaEndpointBackend:
    """``Endpoint`` Protocol over izzy ``doca_rdma`` + CPU_PROXY.

    GPU kernel emits one ``RDMA_WRITE`` per ``send``; a CPU progress
    thread drives the NIC doorbell. The receiver polls the trailing
    4-byte signal of the recv buffer for new frames.
    """

    def __init__(self, config: DocaEndpointConfig):
        self._cfg = config
        self._doca = None
        self._ep = None
        self._send_buf = None  # izzy BufferInfo
        self._recv_buf = None
        self._send_payload = None  # torch tensor view of send buffer
        self._send_signal = None
        self._recv_payload = None
        self._recv_signal = None
        self._send_seq = 0
        self._last_recv_seq = 0
        self._progress_stop = threading.Event()
        self._progress_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None

        self._inbox = deque()
        self._inbox_lock = threading.Lock()

        self._recv_credits = 0
        self._started = False

    # ------------------------------------------------------------------
    # Endpoint Protocol
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            return EndpointStatus.kOk
        try:
            self._doca = _load_izzy(self._cfg.izzy_build_dir)
            import torch

            torch.cuda.set_device(int(self._cfg.gpu_id))
            self._ep = self._doca.DocaRdmaEndpoint()
            self._ep.init(int(self._cfg.gpu_id), str(self._cfg.nic_name))
            self._send_buf = self._ep.alloc_doca_gpu_buffer(_BUF_BYTES)
            self._recv_buf = self._ep.alloc_doca_gpu_buffer(_BUF_BYTES)

            self._send_payload = torch.ops.doca_rdma.wrap_gpu_memory(
                self._send_buf.addr, _BUF_BYTES, int(self._cfg.gpu_id)
            ).view(torch.uint8)
            self._send_signal = torch.ops.doca_rdma.wrap_gpu_memory(
                self._send_buf.addr + _IMM_BYTES + DraftApiProtocol.kMessageBytes,
                _SIGNAL_BYTES,
                int(self._cfg.gpu_id),
            ).view(torch.int32)
            self._recv_payload = torch.ops.doca_rdma.wrap_gpu_memory(
                self._recv_buf.addr, _BUF_BYTES, int(self._cfg.gpu_id)
            ).view(torch.uint8)
            self._recv_signal = torch.ops.doca_rdma.wrap_gpu_memory(
                self._recv_buf.addr + _IMM_BYTES + DraftApiProtocol.kMessageBytes,
                _SIGNAL_BYTES,
                int(self._cfg.gpu_id),
            ).view(torch.int32)

            self._send_payload.zero_()
            self._recv_payload.zero_()
            torch.cuda.synchronize()

            peer = str(self._cfg.remote_peer_name)
            if self._cfg.is_server:
                ok = self._ep.accept_peer(int(self._cfg.bind_port or self._cfg.remote_port), peer)
            else:
                # The draft side may still be opening its DOCA accept
                # port after loading the model. Retry connect until
                # handshake_timeout_s elapses.
                deadline = time.monotonic() + float(self._cfg.handshake_timeout_s)
                ok = False
                while time.monotonic() < deadline:
                    ok = self._ep.connect_to_peer(
                        str(self._cfg.remote_host), int(self._cfg.remote_port), peer
                    )
                    if ok:
                        break
                    time.sleep(1.0)
            if not ok:
                return EndpointStatus.kError
            ok = self._ep.exchange_buffer_info(peer, self._send_buf, self._recv_buf)
            if not ok:
                return EndpointStatus.kError

            self._progress_stop.clear()
            self._progress_thread = threading.Thread(
                target=self._progress_loop, name="DocaEp-progress", daemon=True
            )
            self._progress_thread.start()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="DocaEp-poll", daemon=True
            )
            self._poll_thread.start()
            self._started = True
            return EndpointStatus.kOk
        except Exception:
            self._cleanup()
            return EndpointStatus.kError

    def stop(self):
        self._progress_stop.set()
        self._cleanup()
        self._started = False
        return EndpointStatus.kOk

    def prime_recv(self, count):
        if not self._started:
            return EndpointStatus.kNotStarted
        if count <= 0:
            return EndpointStatus.kInvalidPayload
        if self._recv_credits + count > self._cfg.recv_queue_depth:
            return EndpointStatus.kQueueFull
        self._recv_credits += count
        return EndpointStatus.kOk

    def poll_once(self):
        if not self._started:
            return EndpointStatus.kNotStarted
        with self._inbox_lock:
            return EndpointStatus.kOk if self._inbox else EndpointStatus.kEmpty

    def send(self, packet):
        if not self._started:
            return EndpointStatus.kNotStarted
        if self._recv_credits <= 0:
            return EndpointStatus.kNoRecvCredits
        payload = bytes(packet.payload)
        if len(payload) != self._cfg.payload_bytes:
            return EndpointStatus.kInvalidPayload

        import torch

        # Pack: 4 byte imm + 96 byte payload + 4 byte signal counter.
        self._send_seq = (self._send_seq + 1) & 0x7FFFFFFF
        imm32 = int(packet.imm_data) & 0xFFFFFFFF
        # CPU-staged buffer to upload contiguously.
        frame = bytearray(_IMM_BYTES + DraftApiProtocol.kMessageBytes + _SIGNAL_BYTES)
        struct.pack_into("<I", frame, 0, imm32)
        frame[_IMM_BYTES : _IMM_BYTES + DraftApiProtocol.kMessageBytes] = payload
        struct.pack_into("<I", frame, _IMM_BYTES + DraftApiProtocol.kMessageBytes, self._send_seq)
        self._send_payload[:_FRAME_BYTES].copy_(torch.frombuffer(frame, dtype=torch.uint8))
        torch.cuda.synchronize()

        remote = self._ep.get_remote_recv_buf(self._cfg.remote_peer_name)
        local_send = self._ep.get_local_send_buf(self._cfg.remote_peer_name)
        qp_ptr = self._ep.get_qp_dev_ptr(self._cfg.remote_peer_name)
        wqe_out = torch.zeros(1, dtype=torch.int64, device=f"cuda:{int(self._cfg.gpu_id)}")
        # imm_value=0 takes our patched RDMA_WRITE (no IMM) branch.
        torch.ops.doca_rdma.write_with_imm(
            qp_ptr,
            local_send.addr,
            _FRAME_BYTES,
            local_send.lkey,
            remote.addr,
            remote.rkey,
            0,
            wqe_out,
        )
        torch.cuda.synchronize()
        self._recv_credits -= 1
        return EndpointStatus.kOk

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        with self._inbox_lock:
            if not self._inbox:
                return EndpointStatus.kEmpty, None
            return EndpointStatus.kOk, self._inbox.popleft()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _progress_loop(self):
        while not self._progress_stop.is_set():
            try:
                self._ep.cpu_proxy_progress(self._cfg.remote_peer_name)
            except Exception:
                return

    def _poll_loop(self):
        # Spins on the trailing signal of the recv buffer. When it
        # changes, copy the 100-byte head (imm + payload) into the
        # inbox.
        while not self._progress_stop.is_set():
            try:
                cur = int(self._recv_signal.cpu().item())
            except Exception:
                return
            if cur != self._last_recv_seq:
                self._last_recv_seq = cur
                head = bytes(
                    self._recv_payload[: _IMM_BYTES + DraftApiProtocol.kMessageBytes].cpu().tolist()
                )
                imm = struct.unpack("<I", head[:_IMM_BYTES])[0]
                payload = head[_IMM_BYTES:]
                with self._inbox_lock:
                    self._inbox.append(EndpointPacket(imm_data=int(imm), payload=payload))
            else:
                time.sleep(0.0005)

    def _cleanup(self):
        self._progress_stop.set()
        # Threads are daemon — they'll exit when the program exits.
        # Best effort: nudge the endpoint shutdown.
        try:
            if self._ep is not None:
                self._ep.cleanup()
        except Exception:
            pass
        self._ep = None
