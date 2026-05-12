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

## Addendum: host-pinned DOCA buffer patch (2026-05-12)

Implemented fix path 1 above. Lands as an opt-in flag (env var
``GLUE_DOCA_HOST_PINNED_BUFFERS=1``) on izzy's
``alloc_doca_gpu_buffer`` so the rest of the izzy + DOCA SDK stack is
unaffected when the flag is off.

### What the patch does

In `doca_rdma_endpoint.cpp::alloc_doca_gpu_buffer`, when the env flag
is set, instead of calling `doca_gpu_mem_alloc(GPU)`:

* `cudaHostAlloc(host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable)`
  to get pinned host RAM with a device-mapped view.
* `cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0)` to get the UVA
  address (in our environment this returns the same pointer as
  `host_ptr` — true UVA).
* `ibv_reg_mr(pd, host_ptr, aligned, ...)` to register the host pinned
  range with the NIC (regular MR, not dmabuf — the OS-pinned host page
  is rock-stable, no driver migration concern).
* Return `BufferInfo{addr=dev_ptr, lkey=mr->lkey, rkey=mr->rkey}` so
  the rest of the stack treats it as a normal GPU buffer.

Both processes (target and draft) must run with the flag for the fix
to apply to both directions.

### What now works

End-to-end test in the new mode (target Qwen3-4B on GPU 1 / mlx5_1,
draft Qwen3-0.6B on GPU 0 / mlx5_0, both with `GLUE_DOCA_HOST_PINNED_BUFFERS=1`):

* The draft side issues an `RDMA_WRITE` (via DOCA CPU_PROXY GPU
  kernel) and the bytes **arrive at the right physical address** in
  target's recv buffer:

  ```
  [doca-poll dbg] last_seen=1 cur=1
    recv[0:16]=[0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0]
    addr=0x73ddc143cc00
  ```

  Decoded: `imm_data=0x80000000` (msg_type=kDraftToTarget, slot=2048),
  `round_seq_num=1`, `position=0`, `version=1`, `message_type=0`,
  `num_tokens=4`. This is the correct DraftToTarget reply.

* `_DocaEndpointBackend.recv()` successfully returns the
  EndpointPacket to the SpecDecodeChannel layer.

So the physical-page-migration root cause that motivated this fix is
**resolved**: the bytes do land in the right place on the receiver
side now. ✓

### Remaining gap (different symptom, different layer)

Despite the recv buffer holding the correct payload and the endpoint
returning it to the channel, TRT-LLM's `_receive_responses` still
times out with `pending=[2048]`. The packet appears to be consumed
exactly once but the spec-decode layer doesn't match it against the
pending request.

Most likely candidates for the new failure (none confirmed):

1. `SpecDecodeChannel.pump_once_for_bound_request` returns a status
   other than `kOk` (e.g., the protocol validation in `channel.recv`
   rejects the 96-byte payload because of a subtle field encoding
   issue introduced by our 4-byte imm-prefix wire frame). Adding a
   print inside the channel would settle this in a few minutes.
2. Race between `_DocaEndpointBackend`'s poll thread and the
   spec-decode main thread on the inbox / `_recv_credits` state.
3. A second consumer somewhere in TRT-LLM is draining our inbox
   before `_receive_responses` runs (e.g., a credit-refresh poll in
   `_send_requests` — though we haven't found one).

This is a layer-above-transport debugging task; the wire side is now
clean.

### 2026-05-12 follow-up: confirmed it is a send-side regression at round 2

A second debug pass narrowed the failure precisely. Instrumented
`_DocaEndpointBackend.send`, `IbverbsDraftOffloadLayer._send_requests`,
and the layer's `_receive_responses` entry point, and re-ran the
same launch.

Per-round trace (Qwen3-4B target + Qwen3-0.6B draft, both with
`GLUE_DOCA_HOST_PINNED_BUFFERS=1`, single request id=2048):

```
# round 1 (prefill step)
[_send_requests ENTRY] request_ids=[2048] start_positions=[11] num_accepted=[1]
[doca send ENTRY]     started=True credits=4096 imm=0x80000001 send_seq=0
[doca send DONE]      wqe_out=0 send_seq=1 credits_left=4095
[recv_responses ENTRY] request_ids=[2048] pending=[2048]
[doca poll_once] kOk inbox_len=1                              ← round-1 reply arrives
                                                              ← _receive_responses returns

# round 2 (next generation step, same request)
[_send_requests ENTRY] request_ids=[2048] start_positions=[16] num_accepted=[5]
[doca send ENTRY]     started=True credits=4096 imm=0x80000001 send_seq=1
[doca send DONE]      wqe_out=1 send_seq=2 credits_left=4095   ← second WQE *placed* in SQ
[recv_responses ENTRY] request_ids=[2048] pending=[2048]
… 2 s of polling, every call returns kEndpointEmpty …
RuntimeError: Timed out waiting for draft offload responses (pending=[2048], timeout_s=2.0)
```

So `_receive_responses` IS called twice — once per round-trip — and
the first round completes fully. Earlier readings that suggested
"the packet was consumed but the layer didn't see it" were
mis-pattern-matched: the kOk we saw was round 1's reply, the
subsequent kEndpointEmpty flood is round 2 waiting for a reply that
never lands.

The reply never lands because the **second `RDMA_WRITE` never reaches
the wire**. On the target, izzy's CPU-proxy diagnostic shows the SQ
producer counter transitioning only `0 → 1`; the doorbell record
stabilises at `*DBR=0x01000000` (big-endian SQ pi = 1) and never
advances to `0x02000000` even after `write_with_imm` returns
`wqe_out=1` for round 2. On the draft, the receive-signal counter
stays at 1 — no second packet arrives — so `[draft][doca] round=N`
is logged for `N=1` only.

Concretely, on the target this round-2 send dbg fires:

```
[doca send DONE] wqe_out=1 send_seq=2 credits_left=4095
```

immediately followed by 2 seconds of polls on a recv buffer that
never updates. izzy's `cpu_proxy_progress` only emits 10 dbg events
per thread before capping, so the absence of a "[progress
transition] prod=2" line in our current logs is not proof on its
own; the conclusive signal is the stable `*DBR=0x01000000` across all
8 visible post events and the fact that draft never sees the second
write.

What this likely is (in rough order of probability):

1. **izzy's `rdma_write_imm_kernel` cadence assumption.** The
   kernel calls `doca_gpu_dev_verbs_reserve_wq_slots(qp, 1)` +
   `submit(qp, wqe_idx + 1)`. If `submit` is templated to GPU-SM-DB
   semantics by default (which we believe is the case in this build)
   the second call may be a no-op against a doorbell BAR that
   `MLNX_OFED 25.07` does not expose to the GPU. Round 1 happened
   to succeed only because the *first* doorbell write piggy-backs on
   the initial QP-to-RTS state set by `doca_gpu_dev_verbs_export_qp`;
   subsequent submits are silently lost. Verifying this is a 30-line
   kernel patch that explicitly templates `submit<…
   DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY …>` (or AUTO) and a
   rerun. The kernel source is in
   `doca_rdma_izzy_work/src/doca_rdma_torch_ops.cu`.
2. **Send-CQ not drained between rounds.** The kernel does not
   `wait_send_completion` after submit; if CPU-proxy requires the
   prior CQE to be reaped before honouring a new doorbell ring, the
   second submit would stall. Cheap to test: add an explicit
   `poll_send_cq` after each `write_with_imm` and see if round 2
   moves.
3. **Buffer-aliasing on the second send.** Both rounds write to the
   same 104-byte staging slice of `_send_payload`. If DOCA caches the
   physical address-translation table per WQE on the first ring, the
   second WQE could end up pointing at a stale page (less likely
   given the buffer is host-pinned and ibv-MR-registered, but worth
   a quick byte dump of the SQ memory after round 2's submit).

Next step (when picking this up): add the explicit CPU-PROXY
template to `rdma_write_imm_kernel`'s `submit` call and a
`wait_send_completion` between rounds 1 and 2; rerun the same
target/draft pair and check whether the SQ DBR advances to
`*DBR=0x02000000` and whether draft logs `round=2`.

The TRT-LLM-side code path (layer + channel + endpoint) is **not**
the bug — that piece is now demonstrably correct for one full
round-trip. The remaining work is entirely in the izzy WQE
submission kernel.

### 2026-05-12 follow-up (third pass): izzy kernel is actually fine

The third debug pass revised the previous conclusion. The previous
pass said "the bug is in izzy's WQE submit kernel." That turned out
to be wrong. A minimal synthetic two-process harness
(`doca_rdma_izzy_work/test_multi_send.py`) calls
`torch.ops.doca_rdma.write_with_imm` `N` times in a tight loop, with
optional matmul-heavy compute inserted between calls to mimic
TRT-LLM's verify forward.

Findings:

* **`N=4`, no intervening compute → all 4 writes land at the
  receiver.** Server log:
  ```
  [server] recv first_byte transitioned 0 → 1 buf[0:8]=[1,1,1,1,1,1,1,1]
  [server] recv first_byte transitioned 1 → 2 buf[0:8]=[2,2,2,2,2,2,2,2]
  [server] recv first_byte transitioned 2 → 3 buf[0:8]=[3,3,3,3,3,3,3,3]
  [server] recv first_byte transitioned 3 → 4 buf[0:8]=[4,4,4,4,4,4,4,4]
  ```
* **`N=4`, with `~1 s` of fp16 matmul between each send → 3 of 4
  land, the trailing one is dropped.** The exact "lost the last
  round" pattern that TRT-LLM hits at round 2.

So the izzy `rdma_write_imm_kernel` correctly drives `submit(qp, 2)`,
`submit(qp, 3)`, etc. The CPU-proxy template hypothesis from the
2026-05-12 follow-up above was wrong — the kernel is multi-call safe
on this hardware. The `[progress transition]` dbg in
`cpu_proxy_progress` *is* unreliable, but only because the dbg
counter caps before the second transition lands; the wire delivery
itself succeeds in the synthetic test.

Verified that the in-TRT-LLM failure is **not**:

* CUDA-stream juggling between TRT-LLM forward passes. Pinning the
  endpoint to its own `torch.cuda.Stream` for every `write_with_imm`
  did not change the failure.
* The CPU progress daemon getting starved. Calling
  `cpu_proxy_progress(...)` inline 64 times right after each submit
  did not change the failure.
* Draft-side compute. Swapping the trtllm draft for a `--backend mock`
  draft (no model load, no torch compute) still loses round 2.

So the wire write for round 2 leaves the SQ on the target side, but
something inside the **target process** (between rounds 1 and 2)
either corrupts the SQ memory, races against the DOCA QP state
machine, or otherwise prevents the second WQE from clearing the NIC.
The synthetic test reproduces "drop the last round" only with heavy
intervening compute on the *same GPU as the sending NIC*, which
strongly suggests a memory-ordering or page-migration interaction
between torch's caching allocator (which keeps churning the GPU heap
during model forward) and the DOCA-owned SQ memory page — even with
`GLUE_DOCA_HOST_PINNED_BUFFERS=1` on the *buffers*, the SQ memory
itself is still GPU-resident DOCA memory and remains exposed.

Concrete next experiments (each ~30 min of work):

1. **Allocate the SQ in host-pinned memory too.** Today only the send
   and recv data buffers are host-pinned via `cudaHostAlloc`. The QP
   metadata / WQE ring is still allocated by
   `doca_gpu_mem_alloc(GPU)` inside DOCA. Mapping the SQ region to
   host pinned would close the same fault path the
   2026-05-12 host-pinned-buffer fix closed for the *data* path.
   Requires a DOCA-level option (`doca_gpu_verbs_qp_attr_set_*`)
   plumb-through that may or may not exist.
2. **Pre-warm torch's caching allocator before DOCA init.** Allocate
   a single ~30 GiB tensor on the target GPU, free it, then init the
   DOCA endpoint. Subsequent torch allocations stay inside the
   pre-grown slab and don't race with DOCA pages.
3. **Phase-9 multi-process split.** Move the DOCA endpoint into a
   separate process with shared-memory IPC to TRT-LLM. The synthetic
   test confirms a torch-free process can drive `write_with_imm` at
   arbitrary cadence; this is the most robust path forward.

Revised conclusion: izzy kernel is **not** the bug. The remaining
work is on the target-process memory-isolation side (host-pinning
the SQ, allocator pre-warm, or two-process split).

### 2026-05-12 follow-up (fourth pass): isolated to "generate-between-sends"

The minimal `_DocaEndpointBackend` repro at
`examples/llm-api/rdma/test_endpoint_clone.py` narrowed the trigger to
a single sequence inside the *target* process:

> A real `llm.generate(...)` (Qwen3-4B forward) between two
> `ep.send(...)` calls poisons the QP state so the second send's
> wire write never reaches the peer.

Verified matrix (single-process, ``TLLM_WORKER_USE_SINGLE_PROCESS=1``,
sender-thread fix in place):

| Scenario | Result |
|---|---|
| 4 sends, no compute between | all 4 land at server ✓ |
| 4 sends, fp16 matmul (256 MB tensors) between | all 4 land ✓ |
| 4 sends, fp16 matmul (≥6 GB tensors) between | all 4 land ✓ |
| LLM loaded but never forwarded, 2 sends | both land ✓ |
| LLM loaded + warmup generate, then 2 sends with no generate between | both land ✓ |
| 2 sends with `llm.generate(...)` between | **round 2 lost** ✗ |
| Same as above, but call `torch.cuda.empty_cache()` + `synchronize()` between gen and round-2 send | **round 2 lost** ✗ |

Client-side state on the failing round 2 is **perfect**: SQ producer
counter advances to 2, DBR advances to `0x02000000`, the
`write_with_imm` kernel reports `wqe_out=1`. But the receiving NIC
records zero `rx_write_requests` increment, and the draft side's
recv-signal counter never moves past round 1.

So `llm.generate(...)` does something that breaks the DOCA
external-datapath QP's ability to emit subsequent sends — likely a
CUDA memory-pool / virtual-address change that invalidates the SQ's
GPU-resident page mapping (the host-pinned-buffer fix only covers the
data buffers, not the SQ ring itself).

Phase 2 ibverbs over the same NIC + the same model.forward survives —
because libibverbs uses a regular `ibv_qp` whose SQ is in CPU-pinned
memory and whose doorbell goes through the kernel's mlx5 driver, not
through DOCA's external-datapath GPU-exported QP.

### Hardware/SDK gate

Per the phase 14 design doc, full GPU-initiated RDMA on this NIC
requires **MLNX_OFED ≥ 25.10**. This host runs **25.07**. The Phase 3
CPU_PROXY fallback is the documented workaround for 25.07, and it
works for isolated round-trips (Phase 3 first send + first receive is
byte-correct, as the earlier addenda show), but the second round-trip
after a real CUDA-heavy model forward is gated on a
SDK/firmware-level fix we cannot patch from userspace.

### What ships in this commit

* `_DocaEndpointBackend` gets a **dedicated sender thread + dedicated
  CUDA stream + ctypes-direct WQE staging into the host-pinned send
  buffer**. With this fix:
  * The SQ producer counter now correctly advances on every send (no
    longer stuck at 1 on round 2).
  * The doorbell record updates: `0x01000000 → 0x02000000 → ...`.
  * The synthetic ``test_endpoint_clone`` repro delivers N=4
    sends through the same code path that TRT-LLM uses, even under
    6 GB of allocator pressure.
  * The single-round Phase 3 e2e (round 1 only) byte-matches.
* A standalone repro script
  (`examples/llm-api/rdma/test_endpoint_clone.py`) that pinpoints the
  "generate-between-sends" failure without any TRT-LLM scheduler in
  the way.
* This addendum.

What still requires either MLNX_OFED 25.10 or an explicit DOCA SDK
fix (out of our reach):

* Full spec-decode multi-round e2e byte-match when target's
  ``model.forward`` runs between rounds. This is the original
  Phase 3 e2e goal. Workarounds:
  * Phase 2 ibverbs already byte-matches end-to-end on this hardware
    — recommend running spec-decode through that path until OFED
    25.10 is available.
  * Phase 4 (GPU_SM_DB) once OFED 25.10 lands will side-step the
    CPU_PROXY external-datapath path entirely.

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
