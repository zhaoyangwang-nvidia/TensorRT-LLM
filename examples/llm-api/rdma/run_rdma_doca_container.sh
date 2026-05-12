#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the rdma_doca development container with the right device, capability,
# and volume mounts for end-to-end speculative-decoding-over-RDMA work.
#
# Mounts:
#   - /home/scratch.zhaoyangw_gpu/rdma       — work tree
#   - /home/scratch.zhaoyangw_gpu/repo       — sister repos (izzy doca_rdma upstream)
#   - /home/scratch.zhaoyangw_gpu/rdma/doca_rdma_izzy_work → /opt/doca_rdma_izzy
#                                              (so izzy source patches survive the container)
#   - /home/scratch.trt_llm_data             → /scratch.trt_llm_data  (Qwen3 models)
#   - /home/scratch.trt_llm_data_ci          → /home/scratch.trt_llm_data_ci
#
# Image defaults to the snapshot we took after installing DOCA SDK 3.3 + izzy
# patches (``rdma_doca:pre_phase1``). Override with $RDMA_DOCA_IMAGE.

set -euo pipefail
IMAGE=${RDMA_DOCA_IMAGE:-rdma_doca:pre_phase1}
NAME=${RDMA_DOCA_NAME:-rdma_doca}

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --gpus all \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK \
  --network host \
  --volume /home/scratch.zhaoyangw_gpu/rdma:/home/scratch.zhaoyangw_gpu/rdma \
  --volume /home/scratch.zhaoyangw_gpu/repo:/home/scratch.zhaoyangw_gpu/repo \
  --volume /home/scratch.zhaoyangw_gpu/rdma/doca_rdma_izzy_work:/opt/doca_rdma_izzy \
  --volume /home/scratch.trt_llm_data:/scratch.trt_llm_data \
  --volume /home/scratch.trt_llm_data_ci:/home/scratch.trt_llm_data_ci \
  "$IMAGE" \
  sleep 86400000

docker exec "$NAME" bash -c 'echo "container up: $(hostname)"; ls /scratch.trt_llm_data/llm-models/Qwen3 | head -5'
