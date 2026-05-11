# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DraftTarget One-Model Speculative Decoding Implementation.

This module implements a one-model approach for DraftTarget speculative decoding,
where the draft and target models share the same model engine. The draft model
layers are integrated into the target model's KV cache and run in a single forward pass.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.sampler import TorchSampler
from .interface import SpecMetadata, SpecWorkerBase
from .mtp import MTPSampler

if TYPE_CHECKING:
    from ...llmapi.llm_args import DraftTargetDecodingConfig


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DraftTargetOneModelSpecMetadata(SpecMetadata):
    """
    Metadata for DraftTarget one-model speculative decoding.

    This class manages the batch information needed for the one-model DraftTarget
    approach where draft and target models share the same model engine.
    Unlike Eagle3/MTP, DraftTarget does not require capturing hidden states
    from the target model to pass to the draft model.
    """

    # The max number of tokens
    max_num_tokens: int = 0
    # The index of the batch inputs
    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

    def prepare(self):
        """Prepare the metadata before model forward."""
        assert self.request_ids is not None
        # Update batch indices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)
        self.num_tokens -= self.num_generations * self.max_draft_len
        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False


class DraftTargetOneModelSampler(MTPSampler):
    """
    Sampler for DraftTarget one-model speculative decoding.

    Inherits from MTPSampler to reuse the speculative decoding sampling logic.
    """

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args, nextn=args.max_draft_len)


class DraftTargetOneModelWorker(SpecWorkerBase):
    def __init__(
        self,
        spec_config: "DraftTargetDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        self._rdma_offload_enabled = bool(
            getattr(spec_config, "draft_offload_enabled", False)
            or _env_enabled("TLLM_DRAFT_RDMA_OFFLOAD")
        )
        self._rdma_offload_v2 = bool(
            getattr(spec_config, "draft_offload_v2", False)
            or _env_enabled("TLLM_DRAFT_RDMA_OFFLOAD_V2")
        )
        self._rdma_draft_client = None
        self._rdma_v2_offload_layer = None
        # Accumulates ALL output tokens across decode rounds so the draft server
        # can reconstruct the full generation context (prompt tokens are prepended
        # by the server side using the known prompt text).
        # Only used by the v1 path.  V2 stateful sessions live on the draft side.
        self._rdma_output_history: list[int] = []
        if self._rdma_offload_enabled:
            if getattr(mapping, "tp_size", 1) != 1 or getattr(mapping, "pp_size", 1) != 1:
                raise RuntimeError(
                    "RDMA draft offload target path currently supports only "
                    "single-rank TP/PP. Disable draft_offload_enabled for "
                    "multi-rank runs."
                )
            if self._rdma_offload_v2:
                # New izzy-compatible 96-byte WRITE_WITH_IMM path.
                from .ibverbs_draft_offload import (
                    IbverbsDraftOffloadConfig,
                    IbverbsDraftOffloadLayer,
                )

                self._rdma_v2_offload_layer = IbverbsDraftOffloadLayer(
                    IbverbsDraftOffloadConfig(
                        nic_name=getattr(spec_config, "draft_offload_nic_name", "mlx5_0"),
                        server_host=getattr(spec_config, "draft_offload_server_host", "127.0.0.1"),
                        server_port=int(getattr(spec_config, "draft_offload_server_port", 47000)),
                        remote_peer_name=getattr(
                            spec_config, "draft_offload_v2_remote_peer_name", "draft_lpu"
                        ),
                        max_num_requests=int(
                            getattr(spec_config, "draft_offload_v2_max_num_requests", 256)
                        ),
                        max_draft_len=int(spec_config.max_draft_len),
                    )
                )
                logger.info(
                    "DraftTarget RDMA-v2 (izzy compatible) enabled: host=%s port=%s nic=%s "
                    "max_num_requests=%s max_draft_len=%s",
                    spec_config.draft_offload_server_host,
                    spec_config.draft_offload_server_port,
                    spec_config.draft_offload_nic_name,
                    spec_config.draft_offload_v2_max_num_requests,
                    spec_config.max_draft_len,
                )
            else:
                from .rdma_draft_offload import RdmaDraftOffloadClient, RdmaDraftOffloadConfig

                self._rdma_draft_client = RdmaDraftOffloadClient(
                    RdmaDraftOffloadConfig(
                        nic_name=getattr(
                            spec_config,
                            "draft_offload_nic_name",
                            os.environ.get("TLLM_DRAFT_RDMA_NIC", "mlx5_0"),
                        ),
                        server_host=getattr(
                            spec_config,
                            "draft_offload_server_host",
                            os.environ.get("TLLM_DRAFT_RDMA_HOST", "127.0.0.1"),
                        ),
                        server_port=int(
                            getattr(
                                spec_config,
                                "draft_offload_server_port",
                                os.environ.get("TLLM_DRAFT_RDMA_PORT", "47320"),
                            )
                        ),
                        gpu_id=getattr(spec_config, "draft_offload_gpu_id", None),
                        max_draft_len=int(spec_config.max_draft_len),
                        buffer_size=int(getattr(spec_config, "draft_offload_buffer_size", 4096)),
                    )
                )
                logger.info(
                    "DraftTarget RDMA-v1 (legacy 4 KiB) enabled: host=%s port=%s nic=%s",
                    self._rdma_draft_client.config.server_host,
                    self._rdma_draft_client.config.server_port,
                    self._rdma_draft_client.config.nic_name,
                )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _prepare_attn_metadata_for_draft_target(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        """
        Save the attention metadata fields modified by DraftTarget.

        During CUDA-graph warmup, kv_lens_cuda is also saved/restored to avoid
        cross-warmup accumulation. During capture and normal inference we keep
        kv_lens_cuda live so the updates persist.
        """
        is_capturing = torch.cuda.is_current_stream_capturing()

        if (
            spec_metadata.is_cuda_graph
            and not is_capturing
            and hasattr(attn_metadata, "kv_lens_cuda")
            and isinstance(attn_metadata.kv_lens_cuda, torch.Tensor)
        ):
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _update_kv_after_first_draft_step(
        self,
        attn_metadata: AttentionMetadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                self.max_draft_len - num_accepted_tokens[num_contexts:batch_size]
            )
            attn_metadata.kv_lens_cuda[:num_contexts] += 1

        # Some attention backends keep extra indexing state derived from
        # seq_lens / kv_lens that must be refreshed for chained drafting.
        attn_metadata.update_for_spec_dec()

    def _update_kv_for_chained_draft_step(
        self,
        attn_metadata: AttentionMetadata,
        batch_size: int,
    ):
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[:batch_size] += 1

        attn_metadata.update_for_spec_dec()

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
        draft_model: nn.Module,
        resource_manager=None,
        is_warmup: bool = False,
    ):
        """
        Technically incorrect at the moment.
        Leverages Eagle3/MTP setup that does this for the context
        input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
        In DraftTarget, we do not want to shift, which necessitates increasing the final chunk of each request by 1
        for the final accepted token.  This creates a big headache since then the kv lens, seq_lens, token counts all
        have to be updated and then reverted when heading back to the target.  TODO: non trivially fix this issue.
        """

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        if self._rdma_offload_enabled:
            if bool(is_warmup):
                # Warmup: initialize RDMA connection and exercise one real
                # round-trip (tokens=[] → draft server sees only the prompt).
                # This pre-warms the QP, GPU buffers, and NIC queues so the
                # first real decode step does not pay connection-setup latency.
                # Output history is NOT updated; the result is discarded.
                self._rdma_draft_client.request(
                    tokens=[],
                    position=0,
                    max_draft_len=self.max_draft_len,
                    device=logits.device,
                )
                next_draft_tokens = torch.zeros(
                    (batch_size, self.max_draft_len), dtype=torch.int32, device=logits.device
                )
            else:
                next_draft_tokens = self._rdma_offload_draft_tokens(
                    accepted_tokens=accepted_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                    position_ids=position_ids,
                    logits=logits,
                    batch_size=batch_size,
                    request_ids=getattr(spec_metadata, "request_ids", None),
                )
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )
            attn_metadata.use_spec_decoding = True
            return {
                "logits": raw_logits,
                "new_tokens": accepted_tokens,
                "new_tokens_lens": num_accepted_tokens,
                "next_draft_tokens": next_draft_tokens,
                "next_new_tokens": next_new_tokens,
            }

        # Prepare attention metadata for speculative decoding and save state for restore
        self._prepare_attn_metadata_for_draft_target(attn_metadata, spec_metadata)

        # Prepare inputs for the first draft forward
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )

        next_draft_tokens = []
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        # Get the draft KV cache manager if using separate layouts
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            for i in range(self.max_draft_len):
                if i == 0:
                    start_ids_gen = (
                        spec_metadata.batch_indices_cuda[:num_gens] * (self.max_draft_len + 1)
                    ).long()
                    gather_ids_gen = (
                        start_ids_gen
                        + num_accepted_tokens[num_contexts:]
                        - 1
                        + attn_metadata.num_ctx_tokens
                    )
                    gather_ids = torch.concat(
                        [spec_metadata.gather_ids[:num_contexts], gather_ids_gen], dim=0
                    )
                else:
                    gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

                if self.guided_decoder is not None:
                    new_tokens = inputs["input_ids"][gather_ids]
                    self.guided_decoder.add_draft_batch(
                        new_tokens, num_accepted_tokens, draft_step=i
                    )

                if original_all_rank_num_tokens is not None:
                    if i == 0:
                        attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
                    elif spec_metadata.all_rank_num_seqs is not None:
                        attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                hidden_states = draft_model.model(**inputs)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                # Disable spec-dec mode for chained draft steps
                attn_metadata.use_spec_decoding = False

                logits = draft_model.logits_processor(
                    hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
                )
                if self.guided_decoder is not None:
                    d2t = getattr(draft_model.model, "d2t", None)
                    self.guided_decoder.execute_draft_batch(logits, d2t, draft_step=i)

                new_draft_token = self.draft_decoder(logits, draft_model)
                next_draft_tokens.append(new_draft_token)

                # Update inputs and metadata for next draft step
                position_ids = inputs["position_ids"][gather_ids] + 1
                if i == 0:
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                    attn_metadata.on_update()
                    if inputs["attn_metadata"].kv_cache_manager is not None:
                        attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                        attn_metadata.num_contexts = 0
                    self._update_kv_after_first_draft_step(
                        attn_metadata, num_accepted_tokens, num_contexts, batch_size
                    )
                else:
                    self._update_kv_for_chained_draft_step(attn_metadata, batch_size)

                inputs = {
                    "input_ids": new_draft_token,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                    "spec_metadata": spec_metadata,
                }

        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # Restore attention metadata to original state
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        attn_metadata.use_spec_decoding = True

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _rdma_offload_draft_tokens(
        self,
        *,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        logits: torch.Tensor,
        batch_size: int,
        request_ids: Optional[list] = None,
    ) -> torch.Tensor:
        # V2 path: izzy-compatible 96-byte WRITE_WITH_IMM via
        # ``IbverbsDraftOffloadLayer``.  All bookkeeping (round_seq,
        # route binding, response dispatch) happens inside the layer.
        if self._rdma_v2_offload_layer is not None:
            if request_ids is None or len(request_ids) != int(batch_size):
                # Fall back to a synthetic request id; v2 needs one per row.
                request_ids = list(range(int(batch_size)))
            return self._rdma_v2_offload_layer(
                input_ids=accepted_tokens,
                position_ids=(
                    position_ids
                    if position_ids is not None
                    else torch.zeros((batch_size,), dtype=torch.int64, device=logits.device)
                ),
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                batch_size=int(batch_size),
                num_contexts=0,
                request_ids=request_ids,
            )

        if self._rdma_draft_client is None:
            raise RuntimeError("RDMA draft offload client was not initialized")
        if int(batch_size) != 1:
            raise RuntimeError("RDMA draft offload target path currently supports batch_size=1")

        accepted_count = int(num_accepted_tokens[0].detach().cpu().item())
        accepted_count = max(1, min(accepted_count, accepted_tokens.shape[1]))

        if position_ids is None or int(position_ids.numel()) == 0:
            position = 0
        else:
            position = int(position_ids.reshape(-1)[-1].detach().cpu().item())

        # Accumulate all accepted output tokens for full-context draft inference.
        for i in range(accepted_count):
            self._rdma_output_history.append(int(accepted_tokens[0, i].detach().cpu().item()))
        # Cap at MAX_TOKENS (64) to fit in the RDMA buffer.
        tokens_to_send = self._rdma_output_history[-64:]

        logger.info(
            "[RDMA] _rdma_offload_draft_tokens: round=%d pos=%d ctx_len=%d",
            self._rdma_draft_client.round_seq,
            position,
            len(tokens_to_send),
        )
        draft_tokens = self._rdma_draft_client.request(
            tokens=tokens_to_send,
            position=position,
            max_draft_len=self.max_draft_len,
            device=logits.device,
        )
        logger.info("[RDMA] got draft tokens: %s", draft_tokens)
        if not draft_tokens:
            draft_tokens = [0]
        if len(draft_tokens) < self.max_draft_len:
            draft_tokens = draft_tokens + [draft_tokens[-1]] * (
                self.max_draft_len - len(draft_tokens)
            )
        draft_tokens = draft_tokens[: self.max_draft_len]
        return torch.tensor([draft_tokens], dtype=torch.int32, device=logits.device)

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        if spec_metadata.draft_tokens is None:
            draft_tokens = torch.zeros(
                (num_gens, self.max_draft_len), dtype=torch.int, device=logits.device
            )
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, self.max_draft_len)

        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        if num_contexts > 0:
            input_ids_ctx = self._prepare_context_input_ids(
                input_ids,
                attn_metadata.num_ctx_tokens,
                spec_metadata.gather_ids,
                accepted_tokens,
                num_contexts,
            ).to(torch.int32)
        else:
            input_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")

        if num_gens > 0:
            input_ids_gen = accepted_tokens[num_contexts:, :].flatten().to(torch.int32)
        else:
            input_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")

        input_ids = torch.cat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
