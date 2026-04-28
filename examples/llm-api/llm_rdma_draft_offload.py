# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Minimal target-side RDMA draft offload example.

Start the fake draft server first, then run this script. The target model is a
real TensorRT-LLM LLM; the draft model is temporarily replaced by the fake RDMA
peer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL = "/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Explain GPUDirect RDMA in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-draft-len", type=int, default=5)
    parser.add_argument("--draft-host", default="127.0.0.1")
    parser.add_argument("--draft-port", type=int, default=47320)
    parser.add_argument("--ib-dev", default="mlx5_0")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU ID for RDMA memory registration (target side)",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="Set CUDA_VISIBLE_DEVICES before importing TensorRT-LLM.",
    )
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-num-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-max-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-free-gpu-memory-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

    spec_config = DraftTargetDecodingConfig(
        max_draft_len=args.max_draft_len,
        draft_offload_enabled=True,
        draft_offload_nic_name=args.ib_dev,
        draft_offload_server_host=args.draft_host,
        draft_offload_server_port=args.draft_port,
        draft_offload_gpu_id=args.gpu_id,
    )
    # Disable CUDA graphs: RDMA draft calls are Python-side operations and
    # would not be re-executed on each CUDA graph replay.
    # cuda_graph_config=None disables CUDA graphs entirely (empty CudaGraphConfig
    # does NOT disable graphs because the validator auto-fills batch sizes).

    llm = LLM(
        model=args.model,
        speculative_config=spec_config,
        disable_overlap_scheduler=True,
        tensor_parallel_size=1,
        cuda_graph_config=None,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        kv_cache_config=KvCacheConfig(
            max_tokens=args.kv_cache_max_tokens,
            free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        ),
    )
    output = llm.generate(
        args.prompt,
        SamplingParams(max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    print(output.outputs[0].text)


if __name__ == "__main__":
    main()
