<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 12 — Swap TCP for ibverbs WRITE\_WITH\_IMM

Second leg of the four-phase progression. Replaces phase 11's TCP
transport with host-initiated ibverbs `RDMA_WRITE_WITH_IMM` while
keeping the rest of the stack frozen. Validates that the
``transport`` field on ``DraftTargetDecodingConfig`` is enough to swap
between TCP and RDMA without touching business logic.

## What changed vs phase 11

| Layer | Phase 11 | Phase 12 |
|---|---|---|
| Target driver script | `--transport tcp` (default) | `--transport ibverbs` (new flag) |
| Target offload backend | `_TcpRdmaBackend` | `_IbverbsRdmaBackend` (already in tree) |
| Draft server entry | `draft_tcp_server.py` (new in P11) | `draft_rdma_server_v2.py` (already in tree) |
| Draft model load timing | Lazy (on `TcpModelInit`) | Eager (CLI `--model`) |
| Prompt push channel | `TcpModelInit` + `TcpPromptInit` on the same control port | `TcpPromptInit` only, on the dedicated `--tcp-prompt-port` |
| Per-round data plane | 100-byte TCP frame (4 imm + 96 payload) | 96-byte RDMA WRITE\_WITH\_IMM (imm carried natively) |

All other code paths — target `LLM(target_model, speculative_config=...)`,
`IbverbsDraftOffloadLayer`, `SpecDecodeChannel`, prompt-push auto-trigger
inside `DraftTargetOneModelWorker._rdma_offload_draft_tokens` — are
unchanged across phases.

## Files

| Path | Change | Why |
|---|---|---|
| `examples/llm-api/rdma/spec_dec_target_main.py` | +15 / -8 | New `--transport tcp|ibverbs` flag and `--nic` flag. For `ibverbs`, the script wires `draft_offload_v2_transport='ibverbs'` and `draft_offload_nic_name=<nic>` into the `DraftTargetDecodingConfig`. `draft_offload_v2_model_path` is only set when transport is `tcp` (ibverbs draft is preloaded with `--model`). |
| `docs/source/features/rdma-izzy-refactor/phase12_ibverbs_specdec.md` | new ~100 lines | This document. |

Zero core TRT-LLM code changed in this phase — the phase-11 transport
dispatch was already general enough.

## Testing

### Setup

* Container: same `rdma_doca` as phase 11.
* Models: target Qwen3-4B, draft Qwen3-0.6B (Qwen3-1.7B share dir is
  empty), both from `/scratch.trt_llm_data/llm-models/Qwen3/`.
* GPUs: target on GPU 2, draft on GPU 3, both with ~182 GiB free at
  test time (others tenants idle on this box).
* NIC: `mlx5_0` on both ends (loopback-style — RC QP on the same NIC
  with two distinct QPNs).

### Run line

Important: ibverbs server's `channel.start()` does a blocking TCP accept
for QP-info exchange, so target must connect within
`handshake_timeout_s` (default 120 s). Launch draft, wait ~5 s for it
to begin model loading, then launch target in parallel.

```bash
# Draft (GPU 3 / mlx5_0). RDMA data port 47230, TCP prompt port 47231.
GPU_ID=3 python3 examples/llm-api/rdma/draft_rdma_server_v2.py \
  --backend trtllm \
  --model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
  --nic mlx5_0 --port 47230 --tcp-prompt-port 47231 \
  --max-draft-len 4 --max-num-requests 4096

# Target (GPU 2 / mlx5_0). 4096 slots so TRT-LLM's request_id=2048 fits.
GPU_ID=2 python3 examples/llm-api/rdma/spec_dec_target_main.py \
  --transport ibverbs --nic mlx5_0 --max-num-requests 4096 \
  --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \
  --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
  --draft-host 127.0.0.1 --draft-port 47230 --draft-control-port 47231 \
  --prompt "Explain GPUDirect RDMA in one short sentence." \
  --max-tokens 16 --max-draft-len 4
```

### Result

Target stdout:

```
=== generated token_ids ===
[23690, 4656, 1226, 37992, 4835, 374, 264, 5440, 429, 6147, 2118, 821, 8317, 1948, 264, 22670]
=== decoded text ===
 GPUDirect RDMA is a technology that allows direct data transfer between a GPU
```

Draft per-round log:

```
[draft-v2] round=1 req=2048 slot=2048 round_seq=1 last_token=23690 → tokens=[4656, 1226, 37992, 4835]
[draft-v2] round=2 req=2048 slot=2048 round_seq=2 last_token=374   → tokens=[264, 943, 315, 821]
[draft-v2] round=3 req=2048 slot=2048 round_seq=3 last_token=5440  → tokens=[429, 6147, 369, 1550]
[draft-v2] round=4 req=2048 slot=2048 round_seq=4 last_token=2118  → tokens=[821, 8317, 1948, 264]
[draft-v2] round=5 req=2048 slot=2048 round_seq=5 last_token=22670 → tokens=[323, 13940, 2041, 1667]
```

Per-round accept count from target's `last_token` trail (round R's
`last_token` is target's chosen continuation after round R-1 accepted +
bonus): 4 / 1 / 2 / 4 — same mixed pattern as phase 11. Output exactly
matches phase 11 and the no-spec-dec baseline. ✅

### Pass criterion

Byte-identical to baseline `LLM(target).generate(prompt, max_tokens=16)`:
**16 / 16 token IDs match.**

## Known gotchas

1. **`max_num_requests` must be ≥ `request_id + 1`.** TRT-LLM allocates
   `request_id=2048` for the first generate. The wire slot is
   `request_id & 0xFFF`, so the draft's receive buffer pool must have
   at least 2049 slots. Default in `draft_rdma_server_v2.py` was 64 —
   needs an explicit bump to e.g. 4096 (RAM cost: 96 B × 4096 ≈ 384 KiB,
   negligible).
2. **Launch order matters.** Draft's ibverbs accept is blocking — if
   target doesn't connect within 120 s, it errors out with
   `kEndpointError` (`TimeoutError` in the trace log
   `/tmp/tllm_rdma_worker_trace.log`). Launch target while draft is
   still loading the model.
3. **Same NIC loopback works.** Two RC QPs on `mlx5_0` exchanging
   metadata over the loopback TCP handshake establish a normal RC
   connection; the NIC handles the local DMA path natively. Cross-NIC
   topologies are equally valid (target on `mlx5_0`, draft on
   `mlx5_1`).

## What's next (phases 3-4)

| Phase | Transport | Status |
|---|---|---|
| 3 | DOCA CPU_PROXY (GPU kernel issues WQE, CPU progress rings doorbell) | Next. Reuses the `DocaPollChannel` from `phase10_doca_cuda_graph.md`, wraps as an `Endpoint`. Expected to hit the torch×DOCA GPU memory pool conflict — fix path is host-pinned DOCA buffers. |
| 4 | DOCA GPU SM doorbell (`GPU_SM_DB` handler) | Blocked on MLNX_OFED 25.10+; the production host has 25.07. Doc-only this round. |
