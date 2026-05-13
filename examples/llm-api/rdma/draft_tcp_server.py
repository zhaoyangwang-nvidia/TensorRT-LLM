# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""DOCA-progression Phase 1 — pure-TCP draft server.

Mirrors ``draft_rdma_server_v2.py``'s server loop but the data plane is
``_TcpRdmaBackend`` (100-byte TCP frames) instead of ibverbs
WRITE_WITH_IMM. The control plane (model_init / prompt_init / cancel)
is a separate JSON-over-TCP listener.

Lifecycle (matches ``IbverbsDraftOffloadLayer._ensure_started`` ordering):

1. Start TCP control listener (``--control-port``).
2. Wait for ``TcpModelInit`` → lazy-load ``tensorrt_llm.LLM(model_path,
   ...)``. Send ``TcpModelInitAck``.
3. Start data-plane listener (``--port``); accept the target's data
   connection.
4. Per-round loop: read 100-byte TargetToDraft frame from data plane;
   if it's the first round for that slot, allow lazy KV warmup via the
   prompt-init that target pushed in parallel; run ``LLM.generate`` to
   produce ``max_draft_len`` draft tokens; send back DraftToTarget.

Usage:

    GPU_ID=2 \\
    python3 examples/llm-api/rdma/draft_tcp_server.py \\
        --port 47900 --control-port 47901 --max-num-requests 16
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

# Speculative-decoding modules ship with tensorrt_llm; once the wheel is
# installed (`pip install -e .` in TensorRT-LLM/) these import natively.
from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.spec_decode_channel import SpecDecodeChannel  # noqa: E402
from tensorrt_llm._torch.speculative.tcp_endpoint import (  # noqa: E402
    TcpEndpointConfig,
    _TcpRdmaBackend,
)

# Sibling module — reuse the existing JSON message dataclasses.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import draft_session_protocol as dsp  # noqa: E402


class _TrtLlmDraftBackend:
    """Wraps ``tensorrt_llm.LLM`` so each slot has its own session state."""

    def __init__(
        self, model_path: str, dtype: str, max_draft_len: int, kv_cache_free_fraction: float
    ):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        print(
            f"[draft] loading LLM model_path={model_path} dtype={dtype} "
            f"max_draft_len={max_draft_len}",
            flush=True,
        )
        self.model_path = str(model_path)
        self.dtype = str(dtype)
        self.max_draft_len = max_draft_len
        self.kv_cache_free_fraction = float(kv_cache_free_fraction)
        # ``free_gpu_memory_fraction`` is the share of *remaining* GPU mem
        # the TRT-LLM KV cache may use; 0.4 leaves plenty of headroom on
        # a shared GPU. ``cuda_graph_config=None`` keeps startup short.
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
        # request_id (== wire slot) -> running token list (prompt + accepted)
        self._sessions = {}
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.eos_token_id = int(self.tokenizer.eos_token_id or 0)
        print(
            f"[draft] LLM ready vocab={self.vocab_size} eos={self.eos_token_id}",
            flush=True,
        )

    def matches_model_init(self, msg: "dsp.TcpModelInit") -> bool:
        return (
            self.model_path == str(msg.model_path)
            and self.dtype == str(msg.dtype)
            and int(self.max_draft_len) == int(msg.max_draft_len)
        )

    def reset_sessions(self):
        self._sessions.clear()

    def init_session(self, request_id: int, prompt_tokens):
        """Called on TcpPromptInit — establish the per-request prefix."""
        self._sessions.pop(int(request_id), None)
        self._sessions[int(request_id)] = list(int(t) for t in prompt_tokens)
        return len(prompt_tokens) - 1

    def step(self, request_id: int, last_token: int, position: int, round_seq: int):
        sess = self._sessions.get(int(request_id))
        if sess is None:
            raise RuntimeError(f"no session for request {request_id}; send TcpPromptInit first")
        # Reject rewind: target tells us where it actually committed.
        if position < len(sess):
            sess[:] = sess[: int(position)]
        if not sess or sess[-1] != int(last_token):
            sess.append(int(last_token))
        outputs = self.llm.generate([sess], sampling_params=self.sampling_params)
        draft_tokens = list(outputs[0].outputs[0].token_ids)[: self.max_draft_len]
        sess.extend(int(t) for t in draft_tokens)
        return draft_tokens


class _MockDraftBackend:
    """Deterministic dummy backend for transport debugging (no model)."""

    def __init__(self, max_draft_len: int):
        self.max_draft_len = max_draft_len
        self._sessions = {}
        self.vocab_size = 0
        self.eos_token_id = 0

    def matches_model_init(self, msg: "dsp.TcpModelInit") -> bool:
        return int(self.max_draft_len) == int(msg.max_draft_len)

    def reset_sessions(self):
        self._sessions.clear()

    def init_session(self, request_id, prompt_tokens):
        self._sessions.pop(int(request_id), None)
        self._sessions[int(request_id)] = len(prompt_tokens)
        return len(prompt_tokens) - 1

    def step(self, request_id, last_token, position, round_seq):
        return [((int(round_seq) * 1000 + i) % (1 << 31)) for i in range(1, self.max_draft_len + 1)]


class DraftTcpServer:
    """Process-level draft server. One target session per instance."""

    def __init__(self, args):
        self._args = args
        self._backend = None
        self._backend_lock = threading.Lock()
        self._backend_ready = threading.Event()
        self._stop = False

        # Data-plane endpoints are per target connection.  A long-lived
        # draft process can serve a second target run by accepting a new
        # connection after the previous target exits.
        self._endpoint = None
        self._channel = None
        self._control_listener_thread = None
        self._round_count = 0

    def _new_channel(self):
        tcp_cfg = TcpEndpointConfig(
            is_server=True,
            bind_host="0.0.0.0",
            bind_port=int(self._args.port),
            recv_queue_depth=int(self._args.max_num_requests),
            payload_bytes=DraftApiProtocol.kMessageBytes,
            handshake_timeout_s=float(self._args.handshake_timeout_s),
        )
        endpoint = _TcpRdmaBackend(tcp_cfg)
        return endpoint, SpecDecodeChannel(endpoint)

    # ------------------------------------------------------------------
    # Control plane (JSON-over-TCP)
    # ------------------------------------------------------------------

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
                print(f"[draft][control] error: {exc!r}", flush=True)
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
            print(f"[draft][control] listening on port {port}", flush=True)
            while not self._stop:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    break
                threading.Thread(target=_serve, args=(conn,), daemon=True).start()

        self._control_listener_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._control_listener_thread.start()

    def _handle_model_init(self, conn, msg: "dsp.TcpModelInit"):
        with self._backend_lock:
            if self._backend is not None:
                if self._backend.matches_model_init(msg):
                    self._backend.reset_sessions()
                    ack = dsp.TcpModelInitAck(
                        status="ok",
                        vocab_size=int(getattr(self._backend, "vocab_size", 0)),
                        eos_token_id=int(getattr(self._backend, "eos_token_id", 0)),
                    )
                    print(
                        f"[draft][control] model_init reused existing model={msg.model_path}; "
                        "cleared sessions",
                        flush=True,
                    )
                    conn.sendall(dsp.serialize(ack))
                    return
                print(
                    "[draft][control] model changed; reloading old=%s new=%s"
                    % (getattr(self._backend, "model_path", "<mock>"), msg.model_path),
                    flush=True,
                )
                self._backend = None
                self._backend_ready.clear()
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
                print(f"[draft][control] model_init failed: {exc!r}", flush=True)
            conn.sendall(dsp.serialize(ack))

    def _handle_prompt_init(self, conn, msg: "dsp.TcpPromptInit"):
        if not self._backend_ready.is_set():
            ack = dsp.TcpPromptInitAck(
                request_id=int(msg.request_id),
                status="error",
                error="backend not initialized",
            )
            conn.sendall(dsp.serialize(ack))
            return
        try:
            if hasattr(self._backend, "reset_sessions"):
                self._backend.reset_sessions()
            last_pos = self._backend.init_session(
                request_id=int(msg.request_id),
                prompt_tokens=msg.prompt_tokens,
            )
            ack = dsp.TcpPromptInitAck(
                request_id=int(msg.request_id),
                status="ok",
                last_token_position=int(last_pos),
            )
        except Exception as exc:
            ack = dsp.TcpPromptInitAck(
                request_id=int(msg.request_id),
                status="error",
                error=str(exc),
            )
        conn.sendall(dsp.serialize(ack))

    def _handle_cancel(self, conn, msg: "dsp.TcpCancelRequest"):
        if self._backend is not None and hasattr(self._backend, "_sessions"):
            self._backend._sessions.pop(int(msg.request_id), None)
        ack = dsp.TcpCancelRequestAck(request_id=int(msg.request_id))
        conn.sendall(dsp.serialize(ack))

    # ------------------------------------------------------------------
    # Data plane (96-byte protocol over _TcpRdmaBackend)
    # ------------------------------------------------------------------

    def run(self):
        self._start_control_listener()

        print(
            f"[draft] waiting for TcpModelInit on port {self._args.control_port} ...",
            flush=True,
        )
        if not self._backend_ready.wait(timeout=self._args.backend_init_timeout_s):
            raise RuntimeError("timed out waiting for TcpModelInit")
        print(
            "[draft] backend ready, opening data-plane port "
            f"{self._args.port} (waiting for target connect)...",
            flush=True,
        )

        while not self._stop:
            self._endpoint, self._channel = self._new_channel()
            s = self._channel.start()
            if s != SpecDecodeChannel.Status.kOk:
                # A long-lived draft process may sit idle between target
                # processes.  TcpRdmaBackend.start() uses
                # handshake_timeout_s for each accept attempt, so treat an
                # accept timeout as "no target yet" and keep listening.
                print(
                    "[draft] data-plane accept failed: %s; retrying"
                    % SpecDecodeChannel.to_string(s),
                    flush=True,
                )
                self._channel.stop()
                time.sleep(0.5)
                continue
            s = self._channel.prime_recv(int(self._args.max_num_requests))
            if s not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kEndpointQueueFull,
            ):
                raise RuntimeError("prime_recv failed: " + SpecDecodeChannel.to_string(s))
            print("[draft] data-plane connected; entering round loop", flush=True)
            self._serve_data_plane()
            self._channel.stop()
            print("[draft] data-plane disconnected; waiting for next target", flush=True)

    def _serve_data_plane(self):
        assert self._channel is not None

        # Lazy slot→request_id binding (mirrors draft_rdma_server_v2.py).
        seen_routes = {}
        while not self._stop:
            status, request_id, received = self._channel.pump_once_for_bound_request()
            if status == SpecDecodeChannel.Status.kEndpointNotStarted:
                return
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
                    print(
                        "[draft] bind failed slot=%s status=%s"
                        % (route.slot, SpecDecodeChannel.to_string(bind_status)),
                        flush=True,
                    )
                    continue
                seen_routes[route] = handle
                request_id = handle

            if status not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kRouteNotFound,
            ):
                print(
                    "[draft] pump status=%s — skipping" % SpecDecodeChannel.to_string(status),
                    flush=True,
                )
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
                print(f"[draft] backend.step failed: {exc!r}", flush=True)
                continue

            tokens = list(draft_tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(draft_tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=round_seq,
                position=0,
                num_tokens=min(len(draft_tokens), DraftApiProtocol.kMaxTokens),
                tokens=tokens,
            )
            send_status = self._channel.send_for_request(
                int(request_id),
                msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                message=response,
            )
            if send_status != SpecDecodeChannel.Status.kOk:
                print(
                    "[draft] send failed: %s" % SpecDecodeChannel.to_string(send_status),
                    flush=True,
                )
                return
            if self._round_count <= 5 or self._round_count % 50 == 0:
                print(
                    f"[draft] round={self._round_count} req={int(request_id)} "
                    f"slot={int(received.imm_data.slot)} round_seq={round_seq} "
                    f"last_token={last_token} tokens={draft_tokens[:5]}",
                    flush=True,
                )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47900, help="Data-plane TCP port (100-byte frames)")
    ap.add_argument(
        "--control-port",
        type=int,
        default=47901,
        help="Control-plane TCP port (TcpModelInit / TcpPromptInit / cancel)",
    )
    ap.add_argument("--max-num-requests", type=int, default=16)
    ap.add_argument("--backend", choices=["trtllm", "mock"], default="trtllm")
    ap.add_argument("--handshake-timeout-s", type=float, default=300.0)
    ap.add_argument("--backend-init-timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    server = DraftTcpServer(args)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
