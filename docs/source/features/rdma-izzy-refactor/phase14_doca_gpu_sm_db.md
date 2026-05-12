<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 14 — DOCA GPU SM doorbell (`GPU_SM_DB`)

Final leg of the four-phase draft-offload transport progression. The
GPU SM directly writes the NIC doorbell (UAR) — no CPU progress
thread between the GPU and the NIC. This is the "production" target
shape for GPU-initiated RDMA: minimal kernel-launch overhead, no
host involvement on the hot path, and the WQE round-trip can in
principle be captured into a CUDA graph.

**Status:** doc-only this round. The implementation needs the host
kernel / firmware to expose mlx5 BAR1 GPU peer-to-peer mappings to
the DOCA SDK, which only works on MLNX_OFED ≥ 25.10 (current host
runs 25.07). The phase 3 GPU memory pool blocker (task #35) is also
upstream of phase 4.

## What this phase would change

In code terms, very little:

* No new file. `tensorrt_llm/_torch/speculative/doca_endpoint.py`
  gains a flag like `nic_handler: Literal["cpu_proxy", "gpu_sm_db"]`
  that gets passed through to izzy's `load_nic_handler_candidates`
  via the existing `GLUE_RDMA_NIC_HANDLERS` / `GLUE_RDMA_ALLOW_PROXY_HANDLERS`
  env vars (the choice is per-process; we'd just set
  `GLUE_RDMA_NIC_HANDLERS=2` and unset
  `GLUE_RDMA_ALLOW_PROXY_HANDLERS`).
* The CPU progress daemon thread in `_DocaEndpointBackend` becomes a
  no-op (or is gated off entirely) because the doorbell is now rung
  on the GPU SM at WQE-submit time inside the kernel.
* Otherwise the wire frame, draft server, and target driver are
  identical to phase 13.

| Field | Phase 13 (CPU_PROXY) | Phase 14 (GPU_SM_DB) |
|---|---|---|
| Doorbell ring | CPU progress thread reads `*sq_db`, writes UAR | GPU kernel inlines UAR write at submit |
| Per-message host CPU cost | ~1 µs (progress thread spin + UAR write) | 0 |
| WQE → wire latency | dominated by progress poll cadence (~500 ns – 5 µs) | sub-µs (just the kernel-to-NIC DMA path) |
| CUDA graph capturable | No — CPU progress runs outside graph | Yes |
| Requires MLNX_OFED ≥ 25.10 | No | Yes |

## Why it's blocked on this host

From `phase10_doca_cuda_graph.md`:

* DOCA 3.3 `doca_gpu_dev_verbs_export_qp` API succeeds with handler
  id 2 (`GPU_SM_DB`) on this host, **but** the actual NIC doorbell
  write from the GPU SM does not reach the wire. The trace shows the
  QP gets exported, the GPU kernel runs through `submit()`, but
  `rx_write_requests` on the receiving NIC never increments and no
  CQE appears.
* Root cause (per NVIDIA DOCA 3.3 compat matrix): GPU SM doorbell
  needs the mlx5 kernel driver to expose BAR1 with GPU peer mapping
  and the dynamic UAR allocator enabled. Both ship in MLNX_OFED
  25.10+. This host runs MLNX_OFED 25.07.
* CPU_PROXY (phase 13) is the documented fallback for 25.07 and is
  exactly what we use.

## What needs to happen to unblock

In rough order:

1. **MLNX_OFED ≥ 25.10 on the host.** Sysadmin/install task; needs
   reboot or at minimum mlx5 module reload. We can't do this from
   inside the container.
2. **Task #35 (phase 3 memory pool conflict).** Phase 14 inherits
   the exact same conflict between torch's caching allocator and
   DOCA's GPU memory pool — the doorbell handler change doesn't
   touch buffer ownership. So phase 14 e2e cannot beat phase 13 e2e
   until task #35 is resolved.
3. **Tiny code patch** to flip
   `GLUE_RDMA_NIC_HANDLERS` from `1` (CPU_PROXY) to `2` (GPU_SM_DB)
   and skip the CPU progress thread spawn. We are deliberately
   **not** landing this patch now — landing it before (1) and (2)
   ship would create a transport that fails silently with no logs
   on this hardware, which is worse than a config-time error.

## Expected gains once unblocked

For batch-1 spec-decode with a small draft (Qwen3-0.6B) and target
(Qwen3-4B), the round-trip is currently dominated by the
target/draft model forwards (single-digit ms each). Doorbell
latency saving from CPU_PROXY → GPU_SM_DB is sub-microsecond, so
the *per-token* speedup is in the noise. The real win is:

1. **CUDA-graph capture** of the full target verify + draft request
   step. With CPU progress out of the loop, the WQE submit + NIC
   completion stays inside one graph instance — meaningful for very
   small draft models where per-step overhead dominates.
2. **Multi-stream / multi-request scaling.** Each request's progress
   thread is one host CPU consumed full-blast. With GPU_SM_DB the
   CPU is freed up for scheduling, which matters at higher
   concurrencies.

Both gains kick in at >>1 request/iter or with sub-ms model steps;
they're not measurable on our current batch=1 Qwen3-4B verify path.

## Pass criterion (when ready)

Same byte-identical baseline as phases 1-3:

```
[23690, 4656, 1226, 37992, 4835, 374, 264, 5440, 429, 6147, 2118,
 821, 8317, 1948, 264, 22670]
```

Plus a `nvidia-smi nvlink` / `rdma statistic` increment matching the
per-round volume, and the absence of `cpu_proxy_progress` activity
in `/proc/<pid>/status` (just the spin thread that should now be a
no-op).

## What this commit ships

Just this document — phase 14 has no source changes pending. Once
the host kernel is on MLNX_OFED 25.10 and task #35 is closed:

1. Add `nic_handler` flag to `DocaEndpointConfig` (default keeps
   CPU_PROXY for safe-by-default behaviour).
2. Plumb it through `IbverbsDraftOffloadConfig` and
   `DraftTargetDecodingConfig` so users can request
   `--nic-handler gpu_sm_db`.
3. Re-run phase-1 byte-match harness; expect identical output.

The complete diff for that is well under 50 lines based on phase
13's structure.
