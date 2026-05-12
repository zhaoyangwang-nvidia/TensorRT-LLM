# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DOCA RDMA draft server speaking the 96-byte Draft API protocol.

Mirrors ``draft_rdma_server_v2.py``'s server loop but uses
``DocaPollChannel`` as the RDMA transport (CPU_PROXY mode, polling-
based notification rather than WRITE_WITH_IMM). The Draft API frame
format is identical, so any target that speaks the 96-byte protocol can
drive this server.

Backends are reused from ``draft_rdma_server_v2.py``:

* ``--backend mock``: deterministic dummy tokens (no model needed).
* ``--backend trtllm``: real per-request inference via
  ``tensorrt_llm.LLM`` with KV-cache block reuse. The prompt is fixed
  at startup via ``--prompt`` (no separate TCP prompt-init channel in
  this v1 — the goal here is an end-to-end CPU_PROXY transport demo).

Run:

    GPU_ID=0 NIC=mlx5_0 PORT=47820 \
        python3 doca_draft_server.py --backend trtllm \
        --model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
        --prompt "Explain GPUDirect RDMA in one short sentence." \
        --max-draft-len 4 --rounds 3
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
import torch  # noqa: E402

torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
import doca_rdma  # noqa: E402
from doca_poll_channel import DocaPollChannel  # noqa: E402

# Pull DraftApiProtocol and backends from the spec-dec tree without
# requiring a full pip install (mirrors draft_rdma_server_v2.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
_SPEC_ROOT = os.path.join(_TRT, "tensorrt_llm", "_torch", "speculative")
sys.path.insert(0, _SPEC_ROOT)
spec = importlib.util.spec_from_file_location(
    "draft_api_protocol", os.path.join(_SPEC_ROOT, "draft_api_protocol.py")
)
draft_api_protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft_api_protocol)
DraftApiProtocol = draft_api_protocol.DraftApiProtocol

# Inlined backends. ``draft_rdma_server_v2``'s top level loads the
# ibverbs endpoint (which would also bind /dev/infiniband resources at
# import), so we avoid importing it as a whole and just keep the small
# parts we actually need here.


class _MockBackend:
    """Deterministic dummy backend for transport smoke tests."""

    def __init__(self, max_draft_len):
        self.max_draft_len = max_draft_len
        self._sessions = {}

    def init_session(self, request_id, prompt_tokens):
        self._sessions[request_id] = len(prompt_tokens)
        return len(prompt_tokens) - 1

    def step(self, request_id, last_token, position, round_seq):
        return [((round_seq * 1000 + i) % (1 << 31)) for i in range(1, self.max_draft_len + 1)]


class _TransformersBackend:
    """Real per-request stateful inference using HuggingFace ``transformers``.

    Used when ``tensorrt_llm`` is unavailable in the runtime (e.g. the
    ``tensorrt_llm/devel`` build container without the wheel installed).
    The Draft API surface is identical to ``_TrtLlmBackend`` — both
    expose ``init_session`` / ``step`` — so swapping back to TRT-LLM is
    a config flip when the wheel is available.
    """

    def __init__(self, model_path, prompt, max_draft_len, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_draft_len = max_draft_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(device)
        self.fixed_prompt_ids = None
        if prompt:
            self.fixed_prompt_ids = self.tokenizer.encode(prompt)
        self._sessions = {}

    def init_session(self, request_id, prompt_tokens):
        self._sessions[request_id] = {"tokens": list(prompt_tokens)}
        return len(prompt_tokens) - 1

    def _ensure_session(self, request_id):
        sess = self._sessions.get(request_id)
        if sess is not None:
            return sess
        if self.fixed_prompt_ids is None:
            raise RuntimeError("no session and no fixed prompt")
        sess = {"tokens": list(self.fixed_prompt_ids)}
        self._sessions[request_id] = sess
        return sess

    def step(self, request_id, last_token, position, round_seq):
        import torch

        sess = self._ensure_session(request_id)
        if position < len(sess["tokens"]):
            sess["tokens"] = sess["tokens"][: int(position)]
        if not sess["tokens"] or sess["tokens"][-1] != int(last_token):
            sess["tokens"].append(int(last_token))
        input_ids = torch.tensor([sess["tokens"]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.max_draft_len,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1] :].tolist()[: self.max_draft_len]
        sess["tokens"].extend(int(t) for t in new_tokens)
        return new_tokens


class _TrtLlmBackend:
    """Real per-request stateful inference via ``tensorrt_llm.LLM``."""

    def __init__(self, model_path, prompt, max_draft_len, device="cuda:0"):
        from transformers import AutoTokenizer

        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        self.max_draft_len = max_draft_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                free_gpu_memory_fraction=0.4,
            ),
            cuda_graph_config=None,
            max_batch_size=8,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_draft_len,
            temperature=0.0,
            top_p=1.0,
        )
        self.fixed_prompt_ids = None
        if prompt:
            self.fixed_prompt_ids = self.tokenizer.encode(prompt)
        self._sessions = {}

    def init_session(self, request_id, prompt_tokens):
        self._sessions[request_id] = {"tokens": list(prompt_tokens)}
        return len(prompt_tokens) - 1

    def _ensure_session(self, request_id):
        sess = self._sessions.get(request_id)
        if sess is not None:
            return sess
        if self.fixed_prompt_ids is None:
            raise RuntimeError("no session for request %d and no fixed prompt set" % request_id)
        sess = {"tokens": list(self.fixed_prompt_ids)}
        self._sessions[request_id] = sess
        return sess

    def step(self, request_id, last_token, position, round_seq):
        sess = self._ensure_session(request_id)
        if position < len(sess["tokens"]):
            sess["tokens"] = sess["tokens"][: int(position)]
        if not sess["tokens"] or sess["tokens"][-1] != int(last_token):
            sess["tokens"].append(int(last_token))
        outputs = self.llm.generate([sess["tokens"]], sampling_params=self.sampling_params)
        draft_tokens = list(outputs[0].outputs[0].token_ids)[: self.max_draft_len]
        sess["tokens"].extend(int(t) for t in draft_tokens)
        return draft_tokens


def _make_draft_to_target(round_seq, position, draft_tokens, max_draft_len):
    """Wrap draft tokens in a DraftToTarget message."""
    tokens = list(draft_tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(draft_tokens))
    return DraftApiProtocol.Message(
        version=DraftApiProtocol.kVersionV1,
        message_type=DraftApiProtocol.MessageType.kDraftToTarget,
        round_seq_num=round_seq,
        position=position + max_draft_len,
        num_tokens=len(draft_tokens),
        tokens=tokens,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "transformers", "trtllm"], default="mock")
    ap.add_argument("--model", help="HF model path (transformers / trtllm backends)")
    ap.add_argument("--prompt", help="Fixed prompt for stateful backends")
    ap.add_argument("--max-draft-len", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    gpu = int(os.environ["GPU_ID"])
    nic = os.environ["NIC"]
    port = int(os.environ["PORT"])
    peer_name = "doca_draft"

    # Initialize DOCA + channel buffers BEFORE the inference backend
    # touches CUDA. Mixing the two allocators in the opposite order has
    # been observed to leave the channel buffers pointing into ranges
    # that are not actually written by the peer (presumably because the
    # CUDA driver remaps them under us during backend init).
    torch.cuda.set_device(gpu)
    ep = doca_rdma.DocaRdmaEndpoint()
    ep.init(gpu, nic)
    chan = DocaPollChannel(ep, peer_name, gpu)
    ep.accept_peer(port, peer_name)
    ep.exchange_buffer_info(peer_name, chan.send_buf, chan.recv_buf)
    print(f"[draft] connected, cpu_proxy_enabled={ep.cpu_proxy_enabled(peer_name)}", flush=True)

    if args.backend == "trtllm":
        if not args.model:
            ap.error("--model is required for --backend trtllm")
        backend = _TrtLlmBackend(args.model, args.prompt or "", args.max_draft_len)
    elif args.backend == "transformers":
        if not args.model:
            ap.error("--model is required for --backend transformers")
        backend = _TransformersBackend(
            args.model, args.prompt or "", args.max_draft_len, device=f"cuda:{gpu}"
        )
    else:
        backend = _MockBackend(args.max_draft_len)

    chan.start_progress()

    if hasattr(backend, "init_session"):
        # Mock and trtllm both use init_session; for mock the prompt is
        # ignored. We provide a no-op prompt to satisfy the API.
        backend.init_session(request_id=1, prompt_tokens=[1])

    for r in range(args.rounds):
        wire = chan.recv_msg(timeout_s=120.0)
        status, msg = DraftApiProtocol.decode(wire)
        assert status == DraftApiProtocol.Status.kOk, (
            f"protocol decode failed: {DraftApiProtocol.to_string(status)}"
        )
        print(
            f"[draft] round {r} got TargetToDraft round_seq={msg.round_seq_num} "
            f"pos={msg.position} last_token={msg.tokens[0]}",
            flush=True,
        )
        draft_tokens = backend.step(
            request_id=1,
            last_token=int(msg.tokens[0]),
            position=int(msg.position),
            round_seq=int(msg.round_seq_num),
        )
        # Cap to max_draft_len; pad with zeros to kMaxTokens.
        draft_tokens = list(draft_tokens)[: args.max_draft_len]
        reply = _make_draft_to_target(
            msg.round_seq_num, msg.position, draft_tokens, args.max_draft_len
        )
        enc_status, frame = DraftApiProtocol.encode(reply)
        assert enc_status == DraftApiProtocol.Status.kOk, (
            f"encode failed: {DraftApiProtocol.to_string(enc_status)}"
        )
        chan.send_msg(frame)
        print(f"[draft] round {r} sent DraftToTarget tokens={draft_tokens}", flush=True)

    chan.stop()
    print("[draft] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
