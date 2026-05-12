# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""DOCA-progression Phase 3 — draft server using DOCA CPU_PROXY transport.

Same structure as ``draft_tcp_server.py``, but the data plane runs over
the new ``_DocaEndpointBackend`` (izzy ``doca_rdma`` + CPU progress
thread driving the NIC doorbell). Control plane is still JSON-over-TCP
on ``--control-port``.

Lifecycle:

1. Start TCP control listener (``--control-port``).
2. Wait for ``TcpModelInit`` → lazy-load ``LLM(model_path)``.
3. Start DOCA data plane via ``SpecDecodeChannel(_DocaEndpointBackend(
   is_server=True))``. Accepts the target's DOCA connection
   (``ep.accept_peer``) and exchanges buffer info.
4. Per-round loop: poll the DOCA recv buffer for new frames; on each
   ``TargetToDraft``, run ``backend.step`` to generate K draft tokens
   and reply with a ``DraftToTarget`` 96-byte frame.

Usage::

    GPU_ID=3 python3 examples/llm-api/rdma/draft_doca_server.py \\
        --port 47300 --control-port 47301 --nic mlx5_0 \\
        --max-num-requests 4096

Caveat: Loading ``tensorrt_llm.LLM`` in the same process as the DOCA
endpoint hits the torch×DOCA GPU memory pool conflict described in
``phase13_doca_cpu_proxy.md``. Use ``--backend mock`` for transport
smoke tests, and see the phase 13 doc for mitigation paths.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

from tensorrt_llm._torch.speculative.doca_endpoint import (  # noqa: E402
    DocaEndpointConfig,
    _DocaEndpointBackend,
)
from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.spec_decode_channel import SpecDecodeChannel  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import draft_session_protocol as dsp  # noqa: E402


class _TrtLlmDraftBackend:
    """``LLM``-backed step backend, identical interface to draft_tcp_server."""

    def __init__(self, model_path, dtype, max_draft_len, kv_cache_free_fraction):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        print(
            f"[draft][doca] loading LLM model_path={model_path} dtype={dtype} "
            f"max_draft_len={max_draft_len}",
            flush=True,
        )
        self.max_draft_len = max_draft_len
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                free_gpu_memory_fraction=kv_cache_free_fraction,
            ),
            cuda_graph_config=None,
            max_batch_size=8,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_draft_len,
            temperature=0.0,
            top_p=1.0,
        )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._sessions = {}
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.eos_token_id = int(self.tokenizer.eos_token_id or 0)

    def init_session(self, request_id, prompt_tokens):
        self._sessions[int(request_id)] = list(int(t) for t in prompt_tokens)
        return len(prompt_tokens) - 1

    def step(self, request_id, last_token, position, round_seq):
        sess = self._sessions.get(int(request_id))
        if sess is None:
            raise RuntimeError(f"no session for request {request_id}")
        if position < len(sess):
            sess[:] = sess[: int(position)]
        if not sess or sess[-1] != int(last_token):
            sess.append(int(last_token))
        outputs = self.llm.generate([sess], sampling_params=self.sampling_params)
        draft_tokens = list(outputs[0].outputs[0].token_ids)[: self.max_draft_len]
        sess.extend(int(t) for t in draft_tokens)
        return draft_tokens


class _MockDraftBackend:
    def __init__(self, max_draft_len):
        self.max_draft_len = max_draft_len
        self._sessions = {}
        self.vocab_size = 0
        self.eos_token_id = 0

    def init_session(self, request_id, prompt_tokens):
        self._sessions[int(request_id)] = len(prompt_tokens)
        return len(prompt_tokens) - 1

    def step(self, request_id, last_token, position, round_seq):
        return [((int(round_seq) * 1000 + i) % (1 << 31)) for i in range(1, self.max_draft_len + 1)]


class DraftDocaServer:
    def __init__(self, args):
        self._args = args
        self._backend = None
        self._backend_lock = threading.Lock()
        self._backend_ready = threading.Event()
        self._stop = False
        self._round_count = 0

        doca_cfg = DocaEndpointConfig(
            gpu_id=0,
            nic_name=args.nic,
            is_server=True,
            bind_port=int(args.port),
            remote_peer_name="draft_doca",
            recv_queue_depth=int(args.max_num_requests),
            payload_bytes=DraftApiProtocol.kMessageBytes,
            handshake_timeout_s=float(args.handshake_timeout_s),
        )
        self._endpoint = _DocaEndpointBackend(doca_cfg)
        self._channel = SpecDecodeChannel(self._endpoint)
        self._control_thread = None

    # -------- control plane --------

    def _start_control_listener(self):
        port = int(self._args.control_port)

        def _serve(conn):
            try:
                conn.settimeout(self._args.handshake_timeout_s)
                line = dsp.recv_line(conn, timeout_s=self._args.handshake_timeout_s)
                msg = dsp.parse(line)
                if isinstance(msg, dsp.TcpModelInit):
                    self._handle_model_init(conn, msg)
                elif isinstance(msg, dsp.TcpPromptInit):
                    self._handle_prompt_init(conn, msg)
                elif isinstance(msg, dsp.TcpCancelRequest):
                    self._handle_cancel(conn, msg)
                else:
                    conn.sendall(
                        json.dumps({"status": "error", "error": "unknown msg"}).encode("utf-8")
                        + b"\n"
                    )
            except Exception as exc:
                print(f"[draft][doca-ctrl] err: {exc!r}", flush=True)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def _listen_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sock.listen(8)
            print(f"[draft][doca-ctrl] listening on port {port}", flush=True)
            while not self._stop:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    break
                threading.Thread(target=_serve, args=(conn,), daemon=True).start()

        self._control_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._control_thread.start()

    def _handle_model_init(self, conn, msg):
        with self._backend_lock:
            if self._backend is not None:
                ack = dsp.TcpModelInitAck(
                    status="ok",
                    vocab_size=int(getattr(self._backend, "vocab_size", 0)),
                    eos_token_id=int(getattr(self._backend, "eos_token_id", 0)),
                )
                conn.sendall(dsp.serialize(ack))
                return
            try:
                if self._args.backend == "mock":
                    self._backend = _MockDraftBackend(int(msg.max_draft_len))
                else:
                    self._backend = _TrtLlmDraftBackend(
                        model_path=msg.model_path,
                        dtype=msg.dtype,
                        max_draft_len=int(msg.max_draft_len),
                        kv_cache_free_fraction=float(msg.kv_cache_free_fraction),
                    )
                self._backend_ready.set()
                ack = dsp.TcpModelInitAck(
                    status="ok",
                    vocab_size=int(getattr(self._backend, "vocab_size", 0)),
                    eos_token_id=int(getattr(self._backend, "eos_token_id", 0)),
                )
            except Exception as exc:
                self._backend = None
                ack = dsp.TcpModelInitAck(status="error", error=str(exc))
                print(f"[draft][doca-ctrl] model_init failed: {exc!r}", flush=True)
            conn.sendall(dsp.serialize(ack))

    def _handle_prompt_init(self, conn, msg):
        if not self._backend_ready.is_set():
            conn.sendall(
                dsp.serialize(
                    dsp.TcpPromptInitAck(
                        request_id=int(msg.request_id),
                        status="error",
                        error="backend not initialized",
                    )
                )
            )
            return
        try:
            last_pos = self._backend.init_session(
                request_id=int(msg.request_id), prompt_tokens=msg.prompt_tokens
            )
            ack = dsp.TcpPromptInitAck(
                request_id=int(msg.request_id), status="ok", last_token_position=int(last_pos)
            )
        except Exception as exc:
            ack = dsp.TcpPromptInitAck(
                request_id=int(msg.request_id), status="error", error=str(exc)
            )
        conn.sendall(dsp.serialize(ack))

    def _handle_cancel(self, conn, msg):
        if self._backend is not None and hasattr(self._backend, "_sessions"):
            self._backend._sessions.pop(int(msg.request_id), None)
        conn.sendall(dsp.serialize(dsp.TcpCancelRequestAck(request_id=int(msg.request_id))))

    # -------- data plane --------

    def run(self):
        self._start_control_listener()
        print("[draft][doca] waiting for TcpModelInit ...", flush=True)
        if not self._backend_ready.wait(timeout=self._args.backend_init_timeout_s):
            raise RuntimeError("timed out waiting for TcpModelInit")
        print("[draft][doca] backend ready; opening DOCA data plane", flush=True)

        status = self._channel.start()
        if status != SpecDecodeChannel.Status.kOk:
            raise RuntimeError("channel.start failed: " + SpecDecodeChannel.to_string(status))
        status = self._channel.prime_recv(int(self._args.max_num_requests))
        if status not in (
            SpecDecodeChannel.Status.kOk,
            SpecDecodeChannel.Status.kEndpointQueueFull,
        ):
            raise RuntimeError("prime_recv failed: " + SpecDecodeChannel.to_string(status))
        print("[draft][doca] data plane connected; entering loop", flush=True)

        seen_routes = {}
        while not self._stop:
            status, request_id, received = self._channel.pump_once_for_bound_request()
            if status == SpecDecodeChannel.Status.kEndpointEmpty:
                time.sleep(0.0005)
                continue
            if status == SpecDecodeChannel.Status.kRouteNotFound and received is not None:
                route = SpecDecodeChannel.Route(
                    stream_id=int(received.imm_data.stream_id),
                    slot=int(received.imm_data.slot),
                )
                handle = int(received.imm_data.slot)
                bind_status = self._channel.bind_request_route(handle, route)
                if bind_status != SpecDecodeChannel.Status.kOk:
                    continue
                seen_routes[route] = handle
                request_id = handle
            if status not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kRouteNotFound,
            ):
                continue
            if received is None or request_id is None:
                continue
            if int(received.imm_data.msg_type) != int(DraftApiProtocol.MessageType.kTargetToDraft):
                continue

            self._round_count += 1
            last_token = int(received.message.tokens[0])
            position = int(received.message.position)
            round_seq = int(received.message.round_seq_num)
            try:
                draft_tokens = self._backend.step(
                    request_id=int(request_id),
                    last_token=last_token,
                    position=position,
                    round_seq=round_seq,
                )
            except Exception as exc:
                print(f"[draft][doca] backend.step err: {exc!r}", flush=True)
                continue
            tokens = list(draft_tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(draft_tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=round_seq,
                position=0,
                num_tokens=min(len(draft_tokens), DraftApiProtocol.kMaxTokens),
                tokens=tokens,
            )
            self._channel.send_for_request(
                int(request_id),
                msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                message=response,
            )
            if self._round_count <= 5 or self._round_count % 50 == 0:
                print(
                    f"[draft][doca] round={self._round_count} req={int(request_id)} "
                    f"last_token={last_token} → tokens={draft_tokens[:5]}",
                    flush=True,
                )
        self._channel.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47300, help="DOCA data-plane port")
    ap.add_argument("--control-port", type=int, default=47301)
    ap.add_argument("--nic", default="mlx5_0")
    ap.add_argument("--max-num-requests", type=int, default=4096)
    ap.add_argument("--backend", choices=["trtllm", "mock"], default="trtllm")
    ap.add_argument("--handshake-timeout-s", type=float, default=300.0)
    ap.add_argument("--backend-init-timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    server = DraftDocaServer(args)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
