<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 11 — TCP-based 2-card TRT-LLM speculative decoding

First step of the four-phase progression toward GPU-initiated DOCA RDMA
speculative decoding. This phase establishes the **business-layer**
correctness baseline using a plain TCP transport, so phases 2–4 can swap
in ibverbs / DOCA CPU\_PROXY / DOCA GPU\_SM\_DB while keeping the rest
of the stack frozen.

## Architecture

```
+-----------------------------+   TCP control (47901)   +------------------+
| target process (GPU A)      | <---------------------> | draft process    |
|                             |  TcpModelInit           | (GPU B)          |
| LLM(target_model)           |  TcpPromptInit          |                  |
|   + speculative_config:     |                         | listens on:      |
|     DraftTargetDecoding     |                         |   --control-port |
|     draft_offload_v2=True   |                         |   --port (data)  |
|     transport='tcp'         |                         |                  |
|                             |   TCP data (47900)      | LLM(draft_model) |
| -> IbverbsDraftOffloadLayer | <---------------------> |   greedy K-token |
|     (renamed dispatch)      |  96-byte DraftAPI       |   .generate()    |
|     uses _TcpRdmaBackend    |                         |                  |
+-----------------------------+                         +------------------+
```

Target process is the normal entry point: `LLM(target_model,
speculative_config=DraftTargetDecodingConfig(...))`. The speculative
config carries the new fields `draft_offload_v2_transport='tcp'`,
`draft_offload_v2_model_path=<draft path>`, etc., which tell the V2
offload layer to (a) push a `TcpModelInit` to the draft, (b) open a
plain-TCP data plane in place of ibverbs, and (c) auto-push prompts at
context phase.

Draft process is a stateless server — at start it has no model loaded;
the target's `TcpModelInit` decides which model to lazy-load.

## Files

### TRT-LLM core (`tensorrt_llm/_torch/speculative/`)

| Path | Change | Why |
|---|---|---|
| `tcp_endpoint.py` | **new** ~210 lines | Implements the `Endpoint` Protocol over a single full-duplex TCP socket. 100-byte frames (4 imm + 96 payload). Daemon thread drains into a thread-safe inbox. Client retries connect against the data port (draft may still be opening it after a slow `LLM` load). |
| `ibverbs_draft_offload.py` | +50 / -30 | `IbverbsDraftOffloadConfig` grows `transport`, `draft_model_path`, `draft_model_dtype`, `draft_kv_cache_free_fraction`. `__init__` dispatches on `transport` to build either `_IbverbsRdmaBackend` or `_TcpRdmaBackend`. `_ensure_started()` pushes `TcpModelInit` before opening the data plane. `_push_tcp_model_init()` opens a fresh TCP connection to the control port, sends JSON, waits for ack with a 180s timeout (an 8B model load is the worst case). |
| `draft_target.py` | +25 / -1 | `DraftTargetDecoder` ctor wires the new config fields through. `_rdma_offload_draft_tokens` gains `input_ids` / `attn_metadata` params and, when `num_contexts > 0`, slices per-request prompt tokens and calls `_rdma_v2_offload_layer.push_prompt(req_id, tokens)` — the auto-trigger mirror of the in-process draft prefill that the non-offload path runs as the first iteration of the draft loop. |
| `../../llmapi/llm_args.py` | +14 | `DraftTargetDecodingConfig` exposes the same new fields so users can configure everything through standard pydantic config. |

### Examples (`examples/llm-api/rdma/`)

| Path | Change | Why |
|---|---|---|
| `draft_session_protocol.py` | +18 | New `TcpModelInit` / `TcpModelInitAck` dataclasses + `parse()` cases. |
| `draft_tcp_server.py` | **new** ~290 lines | Stateless TCP-based draft server. Control-port thread accepts JSON messages (model_init, prompt_init, cancel). Data-port main loop wraps `_TcpRdmaBackend(is_server=True)` in `SpecDecodeChannel` and runs the same `pump_once_for_bound_request` loop as `draft_rdma_server_v2.py`. Lazy-loads `tensorrt_llm.LLM(model_path)` on `TcpModelInit`. Has a `--backend mock` mode for transport-only smoke tests (no model load needed). |
| `spec_dec_target_main.py` | **new** ~115 lines | Standard target driver: `LLM(target, speculative_config=DraftTargetDecodingConfig(...transport='tcp'...))` + `.generate(prompt)`. Has a `--baseline` flag that runs `LLM(target).generate` directly (no spec-dec) — used to obtain the byte-match ground truth. |
| `run_rdma_doca_container.sh` | **new** ~50 lines | Docker launcher. Adds the production model-path volume mounts (`/scratch.trt_llm_data` and `/scratch.trt_llm_data_ci`). Snapshots survive across restarts via `rdma_doca:pre_phase1`. |

## Testing

### Setup

Container: `rdma_doca` built from `tensorrt_llm/devel:latest-fanrongl` →
`rdma_doca:pre_phase1` snapshot. Inside the container:
`pip install -e .` in the TensorRT-LLM tree (Python only; C++ libs in
`tensorrt_llm/libs/` are pre-built in the image). No further build
needed for phase 1.

Host: 8× B200, ConnectX-7. Picked least-busy 2 GPUs each run.

Models (mounted from `/scratch.trt_llm_data/llm-models/Qwen3/`):

* target: `Qwen3-4B` (full bf16)
* draft: `Qwen3-0.6B` (full bf16; `Qwen3-1.7B` listed on the share but
  had an empty directory)

### Test 1 — mock-backend transport correctness

Draft runs `--backend mock`, which returns the deterministic sequence
`[(round_seq * 1000 + i) for i in 1..max_draft_len]` regardless of
input. Target should still produce identical output to a fresh greedy
LLM.generate, because target's greedy verify will always reject mock
tokens (none match the tokenizer's natural continuation) and fall back
to target's own predictions.

```bash
# draft on GPU 2
GPU_ID=2 python3 examples/llm-api/rdma/draft_tcp_server.py \
  --backend mock --port 47900 --control-port 47901

# target on GPU 3
python3 examples/llm-api/rdma/spec_dec_target_main.py \
  --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \
  --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-1.7B \
  --draft-host 127.0.0.1 --draft-port 47900 --draft-control-port 47901 \
  --prompt "Explain GPUDirect RDMA in one short sentence." \
  --max-tokens 16 --max-draft-len 4
```

Result:

```
=== generated token_ids ===
[23690, 4656, 1226, 37992, 4835, 374, 264, 5440, 429, 6147, 2118, 821, 8317, 1948, 264, 22670]
=== decoded text ===
 GPUDirect RDMA is a technology that allows direct data transfer between a GPU
```

### Test 2 — TRT-LLM draft backend (real spec-dec)

Draft loaded as `LLM(/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B)`
with greedy sampling. K=4 draft tokens per round.

```bash
# draft on GPU 1 (lazy-loads model on first TcpModelInit)
GPU_ID=1 python3 examples/llm-api/rdma/draft_tcp_server.py \
  --backend trtllm --port 47912 --control-port 47913

# target on GPU 2
python3 examples/llm-api/rdma/spec_dec_target_main.py \
  --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \
  --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
  --draft-host 127.0.0.1 --draft-port 47912 --draft-control-port 47913 \
  ...
```

Per-round trace (draft side):

```
[draft] round=1 last_token=23690 tokens=[4656, 1226, 37992, 4835]   # 4/4 match → +4 + bonus
[draft] round=2 last_token=374   tokens=[264, 943, 315, 821]         # 1/4 match
[draft] round=3 last_token=5440  tokens=[429, 6147, 369, 1550]       # 2/4 match
[draft] round=4 last_token=2118  tokens=[821, 8317, 1948, 264]       # 4/4 match
[draft] round=5 last_token=22670                                      # ...
```

Mixed accept rates per round (range 1/4 – 4/4) confirm the verify path
is exercised — neither always-accept nor always-reject would produce
that pattern. Target output is the **same 16-token sequence** as the
mock-backend run, which is identical to the baseline.

### Test 3 — baseline (no spec-dec)

```bash
python3 examples/llm-api/rdma/spec_dec_target_main.py \
  --baseline --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \
  --prompt "Explain GPUDirect RDMA in one short sentence." \
  --max-tokens 16
```

Result: identical token IDs to Tests 1 and 2.

### Pass criterion

All three runs yield:

```
[23690, 4656, 1226, 37992, 4835, 374, 264, 5440, 429, 6147, 2118, 821, 8317, 1948, 264, 22670]
```

→ **Phase 1 spec-decode-over-TCP is byte-identical to non-spec-dec greedy. ✅**

## Known limitations

1. **batch_size=1 only.** Multi-request fan-out / serialization is the
   same constraint the v2 path already had — phase 9-style multi-request
   work is unchanged by this phase.
2. **Qwen3-1.7B not used in tests** because the share's directory is
   empty. Substituting Qwen3-0.6B works and validates the loop; the
   user can re-run with 1.7B once that's populated.
3. **No performance numbers.** Phase 1 is correctness-only. Numbers
   appear in phase 2 (ibverbs) and onward, since the TCP control plane
   isn't representative.
4. **`push_prompt` runs synchronously** inside `_rdma_offload_draft_tokens`
   right before the first round-trip. For an 8B draft this would add
   several seconds to the first token's latency. Phases 2-4 keep the
   same trigger point; aggressive overlap (push during target prefill
   forward) is a future-phase concern.

## What changes in phases 2-4

The four-phase plan keeps the business logic of phase 1 **frozen**.
Each subsequent phase swaps `transport` in
`DraftTargetDecodingConfig.draft_offload_v2_transport` and adjusts the
endpoint:

| Phase | `transport` value | New endpoint backend |
|---|---|---|
| 1 (this) | `tcp` | `_TcpRdmaBackend` |
| 2 | `ibverbs` (default) | `_IbverbsRdmaBackend` (already in tree) |
| 3 | `doca` (new) | host-side wrapper around DOCA CPU\_PROXY (reuses `DocaPollChannel`) |
| 4 | `doca` + GPU-SM-DB handler | `doca_gpu_dev_verbs_*` with `DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB`; requires MLNX\_OFED 25.10+ |

Pass criterion in every phase: byte-identical output to the baseline.
