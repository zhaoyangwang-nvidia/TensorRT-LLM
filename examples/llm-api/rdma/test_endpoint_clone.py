# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Minimal repro for the Phase 3 DOCA CPU_PROXY round-2-wire-loss.

Drives ``_DocaEndpointBackend`` directly (same code path as TRT-LLM
spec-dec) without any TRT-LLM scheduler / spec-decode layer. Tests
whether successive ``ep.send`` calls land at the peer, optionally
with a real ``llm.generate(...)`` (Qwen3-4B forward) inserted
between each send to mimic the spec-decode verify step.

Findings (recorded in ``phase13_doca_cpu_proxy.md``):

* ``N=4`` sends + heavy fp16 matmul (≥6 GB allocator pressure) between
  rounds → server gets all 4. The endpoint code path itself is fine.
* ``N=2`` sends + ``llm.generate(...)`` between rounds → server gets
  round 1 only. Round 2 client-side: SQ producer counter advances to
  2, doorbell record advances to ``0x02000000``; on the wire, the
  packet never arrives.
* Phase 2 ibverbs over the same NIC + same model.forward delivers
  every round, so the failure is specific to DOCA's external-datapath
  QP after a CUDA-heavy LLM forward.

Run as two processes:

    # server (idle receiver on GPU 0)
    GLUE_DOCA_HOST_PINNED_BUFFERS=1 GLUE_RDMA_NIC_HANDLERS=1 \\
    GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 GPU_ID=0 NIC=mlx5_0 PORT=47600 \\
    N=2 python3 test_endpoint_clone.py server

    # client on GPU 1, with Qwen3-4B loaded + generate between sends
    GLUE_DOCA_HOST_PINNED_BUFFERS=1 GLUE_RDMA_NIC_HANDLERS=1 \\
    GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 TLLM_WORKER_USE_SINGLE_PROCESS=1 \\
    CUDA_VISIBLE_DEVICES=1 GPU_ID=0 NIC=mlx5_1 HOST=127.0.0.1 PORT=47600 \\
    LOAD_LLM=1 GENERATE_BETWEEN=1 N=2 python3 test_endpoint_clone.py client

Env vars:

* ``LOAD_LLM=1`` — load Qwen3-4B in the client process.
* ``WARMUP_GENERATE=1`` — call ``llm.generate`` once *before*
  endpoint init (does not fix the failure on its own).
* ``GENERATE_BETWEEN=1`` — call ``llm.generate`` between each send;
  this is the trigger that reproduces the round-2 wire-loss.
* ``HEAVY=1`` — substitute a 6 GB fp16 allocation + matmul instead
  of ``llm.generate`` (does *not* reproduce the failure).
"""

import os
import sys
import time

sys.path.insert(0, "/opt/doca_rdma_izzy/build")
import torch  # noqa: E402

from tensorrt_llm._torch.speculative.doca_endpoint import (  # noqa: E402
    DocaEndpointConfig,
    _DocaEndpointBackend,
)
from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.spec_decode_channel import (  # noqa: E402
    EndpointPacket,
    EndpointStatus,
)

ROLE = sys.argv[1]
GPU = int(os.environ["GPU_ID"])
NIC = os.environ["NIC"]
PORT = int(os.environ.get("PORT", "47600"))
HOST = os.environ.get("HOST", "127.0.0.1")
N = int(os.environ.get("N", "4"))
HEAVY = int(os.environ.get("HEAVY", "0"))

torch.cuda.set_device(GPU)
LOAD_LLM = int(os.environ.get("LOAD_LLM", "0"))
WARMUP_GENERATE = int(os.environ.get("WARMUP_GENERATE", "0"))
GENERATE_BETWEEN = int(os.environ.get("GENERATE_BETWEEN", "0"))
LLM_MODEL = os.environ.get("LLM_MODEL", "/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B")

llm = None
if LOAD_LLM and ROLE == "client":
    print(f"[client] loading {LLM_MODEL} ...", flush=True)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    llm = LLM(
        model=LLM_MODEL,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
    )
    print("[client] LLM loaded", flush=True)
    if WARMUP_GENERATE:
        t0 = time.monotonic()
        out = llm.generate(["hi"], sampling_params=SamplingParams(max_tokens=2, temperature=0.0))
        print(
            f"[client] WARMUP generate done in {time.monotonic() - t0:.2f}s "
            f"tokens={list(out[0].outputs[0].token_ids)}",
            flush=True,
        )

cfg = DocaEndpointConfig(
    gpu_id=GPU,
    nic_name=NIC,
    is_server=(ROLE == "server"),
    remote_host=HOST,
    remote_port=PORT,
    bind_port=PORT,
    remote_peer_name="doca_clone",
    recv_queue_depth=4096,
)
ep = _DocaEndpointBackend(cfg)
st = ep.start()
print(f"[{ROLE}] start={st}", flush=True)
ep.prime_recv(4096)

if HEAVY:
    weight_stash = torch.empty(
        (6 * 1024 * 1024 * 1024) // 2, dtype=torch.float16, device=f"cuda:{GPU}"
    )
    big = [torch.randn(4096, 4096, dtype=torch.float16, device=f"cuda:{GPU}") for _ in range(2)]
    torch.cuda.synchronize()
    print(
        f"[client] heavy alloc: {torch.cuda.memory_allocated(GPU) / (1 << 30):.1f} GiB",
        flush=True,
    )

if ROLE == "server":
    for i in range(N):
        deadline = time.time() + 30
        while time.time() < deadline:
            if ep.poll_once() == EndpointStatus.kOk:
                status, pkt = ep.recv()
                if status == EndpointStatus.kOk:
                    ep.prime_recv(1)
                    print(
                        f"[server] got round {i + 1} imm=0x{pkt.imm_data:08x}",
                        flush=True,
                    )
                    break
            time.sleep(0.005)
        else:
            print(f"[server] TIMEOUT waiting round {i + 1}", flush=True)
            break
else:
    for i in range(N):
        payload = bytes([i + 1] * DraftApiProtocol.kMessageBytes)
        pkt = EndpointPacket(imm_data=0x80000000 | (i + 1), payload=payload)
        st = ep.send(pkt)
        print(f"[client] sent round {i + 1} status={st}", flush=True)
        if GENERATE_BETWEEN and llm is not None:
            t0 = time.monotonic()
            from tensorrt_llm import SamplingParams

            outputs = llm.generate(
                ["hello"],
                sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
            )
            print(
                f"[client] llm.generate round {i + 1}: {time.monotonic() - t0:.3f}s "
                f"tokens={list(outputs[0].outputs[0].token_ids)}",
                flush=True,
            )
        elif HEAVY:
            t0 = time.monotonic()
            for _ in range(50):
                _ = big[0] @ big[1]
            torch.cuda.synchronize()
            print(f"[client] heavy round {i + 1}: {time.monotonic() - t0:.3f}s", flush=True)
        time.sleep(0.5)

ep.stop()
print(f"[{ROLE}] done", flush=True)
