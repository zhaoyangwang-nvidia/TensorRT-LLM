# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout-consistency tests for the DSA token-to-request map.

``req_idx_per_token`` maps each token of the flattened batch to its request
index. It is built once per target forward in ``prepare_for_indices_conversion``
and then reused, which is only valid while the batch layout is unchanged. The
MTP-Eagle draft loop rewrites the layout to one token per request mid-forward,
so the map has to be refreshed or downstream consumers mis-address the
indexer-K cache and the top-k -> global index conversion (nvbugs/6463964).

The bug is invisible at batch=1 (the stale map is ``[0]``, which is correct by
accident) and invisible to accuracy tasks (speculative decoding is
output-lossless, so GSM8K/MMLU pass while acceptance rate silently drops), which
is why it needs a dedicated layout-level test.

PORTING NOTE: these tests assert a property of the *map*, not of one particular
implementation. To point them at a different rebuild routine (e.g. the
device-side ``build_req_idx_per_token`` in PR #16925), replace the body of
``_rebuild`` below -- the layout table and the assertions carry over unchanged.
Layouts with a request longer than one token are expected to be rebuilt
correctly by a general implementation; the ``arange`` fast path here skips them,
which ``_FAST_PATH_ONLY`` encodes.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import DSAtrtllmAttentionMetadata

# Sentinel filling the map buffer before each call. Every slot holds a distinct,
# non-zero value so a write to a slot that should not have been touched is
# detectable -- a zero-initialized buffer would hide exactly that.
_POISON_BASE = 1000
_BUF_LEN = 64

# (seq_lens, is_one_token_per_request)
_LAYOUTS = [
    ([1] * 8, True),  # MTP-Eagle draft steps 1..k -- the regressing case
    ([1], True),  # batch=1: stale map is accidentally correct
    ([5] * 8, False),  # MTP target forward, next_n = max_draft_len + 1
    ([4] * 8, False),  # static/dynamic tree drafting, seq_lens filled with K
    ([3, 1, 4, 1], False),  # ragged: prefill / chunked prefill
]

# This PR's refresh is an ``arange`` fast path: it is a no-op unless every
# request contributes exactly one token. A general rebuild has no such gate.
_FAST_PATH_ONLY = True


def _make_metadata(seq_lens: list[int]) -> DSAtrtllmAttentionMetadata:
    """Build the minimal metadata surface the refresh reads.

    Uses ``object.__new__`` to skip the heavyweight ``__post_init__`` (which
    needs a KV cache manager and CUDA buffers); this exercises pure layout
    bookkeeping and runs on CPU.
    """
    metadata = object.__new__(DSAtrtllmAttentionMetadata)
    metadata._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    metadata._num_tokens = int(sum(seq_lens))
    metadata.req_idx_per_token = torch.arange(_BUF_LEN, dtype=torch.int32) + _POISON_BASE
    metadata._arange_req_idx = torch.arange(_BUF_LEN, dtype=torch.int32)
    return metadata


def _rebuild(metadata: DSAtrtllmAttentionMetadata) -> None:
    """Invoke the map rebuild under test. See PORTING NOTE in the module docstring."""
    metadata._refresh_req_idx_per_token()


def _reference(seq_lens: list[int]) -> torch.Tensor:
    """Ground truth: the same formula ``prepare_for_indices_conversion`` uses."""
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32), torch.tensor(seq_lens, dtype=torch.int32)
    )


@pytest.mark.parametrize("seq_lens,one_token_per_request", _LAYOUTS)
def test_map_matches_repeat_interleave(seq_lens, one_token_per_request):
    """The rebuilt map must equal the layout formula it is shortcutting.

    Pinning the equivalence means the fast path and the general formula cannot
    silently diverge.
    """
    if _FAST_PATH_ONLY and not one_token_per_request:
        pytest.skip("arange fast path only covers one-token-per-request layouts")

    metadata = _make_metadata(seq_lens)
    _rebuild(metadata)

    num_tokens = int(sum(seq_lens))
    assert torch.equal(metadata.req_idx_per_token[:num_tokens], _reference(seq_lens))


@pytest.mark.parametrize("seq_lens,one_token_per_request", _LAYOUTS)
def test_rebuild_does_not_touch_beyond_num_tokens(seq_lens, one_token_per_request):
    """Guard the other half: nothing outside the live region may be written.

    The buffer is pre-poisoned with distinct values, so an over-wide write is
    detectable. Starting from zeros would not catch it.
    """
    metadata = _make_metadata(seq_lens)
    before = metadata.req_idx_per_token.clone()

    _rebuild(metadata)

    num_tokens = int(sum(seq_lens))
    assert torch.equal(metadata.req_idx_per_token[num_tokens:], before[num_tokens:])


def test_draft_loop_layout_is_rebuilt_not_reused():
    """The regressing case, spelled out.

    After draft step 0 the batch is one token per request, so the map must be
    ``arange`` -- not the target-forward layout ``[0]*next_n + [1]*next_n + ...``
    that ``prepare()`` left in the buffer.
    """
    batch_size, next_n = 8, 5
    metadata = _make_metadata([1] * batch_size)
    # Simulate what prepare() left behind for the preceding target forward.
    stale = _reference([next_n] * batch_size)
    metadata.req_idx_per_token[: stale.numel()] = stale

    _rebuild(metadata)

    assert torch.equal(
        metadata.req_idx_per_token[:batch_size], torch.arange(batch_size, dtype=torch.int32)
    )


def test_rebuild_handles_empty_batch():
    """No sequences: nothing to map, buffer must stay intact."""
    metadata = _make_metadata([])
    before = metadata.req_idx_per_token.clone()

    _rebuild(metadata)

    assert torch.equal(metadata.req_idx_per_token, before)
