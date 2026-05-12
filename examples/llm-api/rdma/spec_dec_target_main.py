# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""DOCA-progression Phase 1 — target driver script.

Standard programmatic ``LLM`` entry point. Enables ``DraftTargetDecoding``
with ``draft_offload_v2=True`` and ``draft_offload_v2_transport='tcp'``,
so the v2 layer pushes prompts + per-token round-trips to a separate
draft process (see ``draft_tcp_server.py``).

The greedy output should be **byte-identical** to a fresh
``LLM(target_model).generate(prompt, max_tokens=...)`` baseline because
both verify and accept under greedy + exact-match semantics.

Usage:

    GPU_ID=4 python3 examples/llm-api/rdma/spec_dec_target_main.py \\
        --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \\
        --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-1.7B \\
        --draft-host 127.0.0.1 --draft-port 47900 --draft-control-port 47901 \\
        --prompt "Explain GPUDirect RDMA in one short sentence." \\
        --max-tokens 64 --max-draft-len 4
"""

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument(
        "--draft-model", required=True, help="Path the target tells the draft server to lazy-load"
    )
    ap.add_argument("--draft-host", default="127.0.0.1")
    ap.add_argument(
        "--draft-port",
        type=int,
        default=47900,
        help="Draft data-plane port (matches --port on the draft server)",
    )
    ap.add_argument(
        "--draft-control-port",
        type=int,
        default=47901,
        help="Draft control-plane port (TcpModelInit / TcpPromptInit)",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-draft-len", type=int, default=4)
    ap.add_argument("--prompt", default="Explain GPUDirect RDMA in one short sentence.")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Skip spec-dec; just run fresh LLM(target).generate for byte-match check.",
    )
    ap.add_argument(
        "--max-num-requests",
        type=int,
        default=16,
        help="Matches draft server --max-num-requests (controls channel slot count).",
    )
    ap.add_argument(
        "--kv-cache-free-fraction",
        type=float,
        default=0.4,
        help="Target-side KV cache free GPU memory fraction.",
    )
    args = ap.parse_args()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

    common_kv = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=args.kv_cache_free_fraction,
    )

    if args.baseline:
        # Non-offload, single-process greedy generate — used to obtain the
        # ground-truth token sequence the spec-dec run must match.
        llm = LLM(
            model=args.target_model,
            kv_cache_config=common_kv,
            cuda_graph_config=None,
        )
    else:
        spec = DraftTargetDecodingConfig(
            max_draft_len=args.max_draft_len,
            speculative_model=args.draft_model,
            draft_offload_enabled=True,
            draft_offload_v2=True,
            draft_offload_server_host=args.draft_host,
            draft_offload_server_port=args.draft_port,
            draft_offload_v2_max_num_requests=args.max_num_requests,
            draft_offload_v2_transport="tcp",
            draft_offload_v2_model_path=args.draft_model,
            draft_offload_v2_kv_cache_free_fraction=args.kv_cache_free_fraction,
            draft_offload_v2_tcp_prompt_port=args.draft_control_port,
        )
        llm = LLM(
            model=args.target_model,
            speculative_config=spec,
            kv_cache_config=common_kv,
            cuda_graph_config=None,
        )

    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_p=1.0,
    )
    outputs = llm.generate([args.prompt], sampling_params=sampling)
    out = outputs[0].outputs[0]
    print("=== generated token_ids ===")
    print(list(out.token_ids))
    print("=== decoded text ===")
    print(out.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
