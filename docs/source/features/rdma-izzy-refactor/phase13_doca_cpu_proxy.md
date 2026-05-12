<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 13 — DOCA CPU\_PROXY transport for DraftTarget offload

Third leg of the four-phase progression. GPU kernel issues the RDMA
WQE; a CPU-side progress thread rings the NIC doorbell ("CPU\_PROXY"
handler in DOCA terminology). The izzy ``doca_rdma`` extension does
the heavy lifting via the patches landed in
``phase10_doca_cuda_graph.md`` and the example wrapper in
``phase10`` 增补 2.

This phase delivers a **complete implementation** but the end-to-end
spec-decode run with TRT-LLM as the target is **blocked** by a torch ×
DOCA GPU memory pool conflict on the target side (task #35). The
implementation is committed so the resolution path is unblocked: once
the pool conflict is fixed, the rest of the stack already works.

## What changed vs phase 12

| Layer | Phase 12 | Phase 13 |
|---|---|---|
| Target driver script | `--transport ibverbs` | `--transport doca` (new) |
| Target offload backend | `_IbverbsRdmaBackend` | `_DocaEndpointBackend` (new) — GPU-initiated WQE via izzy `doca_rdma` extension |
| Wire frame | 96-byte protocol payload, imm_data carried natively by NIC WRITE_WITH_IMM | 104-byte frame: 4 imm + 96 payload + 4 polling counter; uses plain RDMA_WRITE (no IMM) because CPU_PROXY can't post recv WRs on the external-datapath QP |
| Doorbell | NIC SQ doorbell rung directly by host | GPU writes producer counter; CPU progress thread reads it, writes DBR + UAR doorbell on behalf of GPU |
| Draft server entry | `draft_rdma_server_v2.py` | `draft_doca_server.py` (new) |
| Model load timing | Eager (CLI `--model`) | Lazy (target → draft `TcpModelInit`) |

The Endpoint protocol abstraction (defined back in phase 11) doesn't
change. ``IbverbsDraftOffloadConfig.transport`` learns one more value
(``"doca"``).

## Files

| Path | Change | Why |
|---|---|---|
| `tensorrt_llm/_torch/speculative/doca_endpoint.py` | **new** ~260 lines | `_DocaEndpointBackend` wrapping izzy's `doca_rdma` extension. Allocates two 4 KiB GPU buffers (send / recv) via `DocaRdmaEndpoint.alloc_doca_gpu_buffer`; wraps each as a torch view. `send()` packs 4-byte imm + 96 payload + 4-byte polling counter, issues a single `torch.ops.doca_rdma.write_with_imm(..., imm=0, ...)` (our patch interprets `imm=0` as plain RDMA_WRITE) for 104 bytes. Two daemon threads — one calls `cpu_proxy_progress` to drive the doorbell, one polls the recv signal and pushes EndpointPackets into the inbox. Connect-side retries until `handshake_timeout_s` since the draft side may still be opening the DOCA accept port. |
| `tensorrt_llm/_torch/speculative/ibverbs_draft_offload.py` | +18 | Adds `transport == "doca"` branch that builds a `_DocaEndpointBackend`. `_ensure_started` pushes `TcpModelInit` for both `"tcp"` and `"doca"` transports (`"ibverbs"` retains the pre-loaded-via-CLI pattern). |
| `tensorrt_llm/llmapi/llm_args.py` | +1 | `draft_offload_v2_transport: Literal["ibverbs", "tcp", "doca"]`. |
| `examples/llm-api/rdma/spec_dec_target_main.py` | +6 / -4 | `--transport` choices grows `doca`; doca path wires both `draft_offload_v2_model_path` (for TcpModelInit) and `draft_offload_nic_name`. |
| `examples/llm-api/rdma/draft_doca_server.py` | **new** ~280 lines | DOCA-side draft server. Same structure as `draft_tcp_server.py`: JSON control listener on `--control-port`, lazy LLM load on `TcpModelInit`, then `SpecDecodeChannel(_DocaEndpointBackend(is_server=True))` data plane. |

## How the implementation works (end-to-end happy path)

```
target process (GPU A + NIC A)              draft process (GPU B + NIC B)
LLM(target_model, speculative_config=        draft_doca_server.py --backend trtllm
   DraftTargetDecodingConfig(
     draft_offload_v2_transport='doca',
     draft_offload_v2_model_path=<draft>,
     ...
   ))
   ↓ first forward (context phase)
DraftTargetOneModelWorker.forward
   ↓ enters offload branch
IbverbsDraftOffloadLayer._ensure_started
   ↓ transport='doca':
   _push_tcp_model_init() → JSON over TCP →    JSON listener
                                                lazy LLM(draft_model)
                                                ack
   ↓ ack received
   _DocaEndpointBackend.start():
     ep.init(0, NIC); alloc_doca_gpu_buffer
     ep.connect_to_peer(host, data_port)   ←→  ep.accept_peer
     ep.exchange_buffer_info               ←→  same
     start progress + poll threads
   ↓
   prime_recv
   _rdma_offload_draft_tokens (auto trigger):
     push_prompt(req_id, prompt_tokens)    ←→  TcpPromptInit / Ack
     layer.forward — issues GPU WQE        ←→  DOCA recv polling thread
                                                pump_once_for_bound_request
                                                backend.step (LLM.generate K)
                                                channel.send_for_request    ←→  target recv poll
   ↓ K draft tokens received
   target verifies (greedy + exact match)
   ↓ loop
```

## Testing

### Pre-step: DOCA SDK + izzy build

Same as phase 11/12 environment. The izzy build artifacts live at
``/opt/doca_rdma_izzy/build/``:

* ``doca_rdma.cpython-312-x86_64-linux-gnu.so`` (pybind module)
* ``doca_rdma_ops.so`` (PyTorch CUDA ops library)

Required env on the target process:

```bash
DOCA_RDMA_BUILD=/opt/doca_rdma_izzy/build
GLUE_RDMA_ALLOW_PROXY_HANDLERS=1
GLUE_RDMA_NIC_HANDLERS=1   # 1 == CPU_PROXY
```

### Test 1 — mock backend on both ends (no model load)

```bash
GPU_ID=0 NIC=mlx5_0 PORT=47310 CTRL=47311 \
  python3 examples/llm-api/rdma/draft_doca_server.py \
    --backend mock --nic mlx5_0 --port 47310 --control-port 47311 \
    --max-num-requests 4096

# (separate terminal — but skip the target LLM load to avoid torch×DOCA
# collision. Need a small client that drives the channel directly,
# which we used during phase 10 verification. Not part of this commit;
# see phase 10's doca_pingpong_smoke.py for the pattern.)
```

This path is exercised by phase 10's smoke harness and is known
working (2 GPUs × 2 NICs, byte-identical 96 B payload). It is not
the focus of phase 13.

### Test 2 — TRT-LLM target + DOCA + mock draft (where we got stuck)

Same run line as phase 12 but with `--transport doca`:

```bash
# draft on GPU 0 / mlx5_0 (mock — no model load)
GPU_ID=0 DOCA_RDMA_BUILD=/opt/doca_rdma_izzy/build \
  GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 GLUE_RDMA_NIC_HANDLERS=1 \
  python3 examples/llm-api/rdma/draft_doca_server.py \
    --backend mock --nic mlx5_0 --port 47310 --control-port 47311 \
    --max-num-requests 4096

# target on GPU 1 / mlx5_1 (loads Qwen3-4B via TRT-LLM)
GPU_ID=1 DOCA_RDMA_BUILD=/opt/doca_rdma_izzy/build \
  GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 GLUE_RDMA_NIC_HANDLERS=1 \
  python3 examples/llm-api/rdma/spec_dec_target_main.py \
    --transport doca --nic mlx5_1 --max-num-requests 4096 \
    --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \
    --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
    --draft-host 127.0.0.1 --draft-port 47310 --draft-control-port 47311 \
    --prompt "Explain GPUDirect RDMA in one short sentence." \
    --max-tokens 16 --max-draft-len 4
```

### What works

* Target loads Qwen3-4B successfully.
* Target → draft TcpModelInit JSON round-trip succeeds.
* Target → draft DOCA `connect_to_peer` + `exchange_buffer_info` succeed
  (after the connect-side retry we added — draft is still opening the
  data port when target first probes).
* Target → draft auto prompt push via `TcpPromptInit` succeeds.
* Target's first round of WQE issuance (`write_with_imm` with imm=0
  → plain RDMA_WRITE) **reaches draft**. Draft logs:
  ```
  [draft][doca] round=1 req=2048 last_token=23690 → tokens=[1001, 1002, 1003, 1004]
  ```
  The `last_token=23690` is target's correct first greedy token from
  Qwen3-4B prefill, so the target → draft direction of the wire frame
  is correctly carrying the 96-byte payload + 4-byte imm_data.

### What's broken

Target times out after 2 s waiting for draft's reply:

```
RequestError: Timed out waiting for draft offload responses (pending=[2048], timeout_s=2.0)
```

Draft `send_for_request` returns OK (no error CQE) but target's recv
buffer never shows the new poll counter — the same failure mode we
observed in phase 10 增补 3 with the standalone DocaPollChannel
when a transformers backend was loaded alongside the DOCA endpoint
on the **same** process and GPU.

### Root cause

Per phase 10 ‘增补 3’ + task #35: the target process loads TRT-LLM, which
uses ``torch``'s caching allocator (and on recent PyTorch versions
``cudaMallocAsync`` mempools). DOCA's ``doca_gpu_mem_alloc`` claims
CUDA memory through a different pool. Both share the same CUDA
virtual-address space; when ``LLM.generate`` grows the torch pool,
the underlying physical pages backing DOCA's recv buffer can be
re-mapped, and the NIC's dma write — which used the physical address
captured at MR-registration time — lands in pages no longer owned by
DOCA. Result: target observes an unchanging recv signal even though
the wire transmission succeeded.

target → draft happens to keep working because the **send** buffer
on the target side is owned by DOCA and read by the local NIC at
transmit time. The collision only manifests on the receive side
(local DMA destination that may have been silently re-mapped).

### Pass criterion — not met

The byte-identical baseline test
``[23690, 4656, 1226, 37992, 4835, 374, 264, 5440, 429, 6147, 2118,
821, 8317, 1948, 264, 22670]`` could not be reproduced over DOCA
CPU_PROXY in this phase. The phase-11 (TCP) and phase-12 (ibverbs)
runs **do** produce that sequence; the wire format and verify logic
are unchanged here, so once the memory-pool issue is solved we expect
the same byte-match.

## Fix paths (any one unblocks phase 3)

In rough effort order:

1. **Host-pinned DOCA buffers.** Patch izzy's
   ``alloc_doca_gpu_buffer`` to allocate via
   ``cudaHostAlloc(cudaHostAllocMapped|Portable)`` +
   ``cudaHostGetDevicePointer`` (host pinned RAM, GPU-visible
   address). Cost: ~200 ns per WQE write across PCIe — negligible at
   spec-decode rates. Cleanest and most robust because torch's
   allocator never touches host pinned memory. Issue is purely a
   build change in izzy.
2. **Pre-grow torch caching allocator before DOCA init.** Allocate a
   ~4 GiB throwaway tensor on the target GPU, free it (caching
   allocator keeps the slab), then let DOCA take fresh pages outside
   that slab. Zero runtime cost but fragile: torch internals change
   between releases, and if subsequent allocations exceed the
   pre-grown slab the trick stops working.
3. **Two-process split.** Run target's LLM in process A and the
   DocaEndpoint in process B, with IPC (shared memory ring + uds
   socket) between them. Total isolation. Cleanest semantically but
   adds 20-50 µs IPC per round; would also require duplicating the
   spec-decode hot path across the IPC boundary, which is a much
   bigger refactor.

## What's next

* **This commit:** ship the implementation + this honest write-up so
  the next person picks up from a known clean state.
* **Task #35** tracks the actual fix.
* **Phase 14 (GPU SM doorbell)** depends on (a) MLNX_OFED ≥ 25.10 on
  the host, which we don't have, and (b) phase 13 unblocked. Phase
  14 is doc-only this round.
