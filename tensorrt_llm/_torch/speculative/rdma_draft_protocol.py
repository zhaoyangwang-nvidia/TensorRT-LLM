# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-size target<->draft token protocol for RDMA bring-up."""

from __future__ import annotations

import struct

BUFFER_SIZE = 4096
MAX_TOKENS = 64
MAGIC = 0x5444524D
VERSION = 1

MSG_TARGET_TO_DRAFT = 1
MSG_DRAFT_TO_TARGET = 2

_HEADER = struct.Struct("<IIIIIIII")
_TOKENS = struct.Struct(f"<{MAX_TOKENS}I")


def _check_tokens(tokens: list[int]) -> None:
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"too many tokens: {len(tokens)} > {MAX_TOKENS}")
    for token in tokens:
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"token out of uint32 range: {token}")


def encode_request(
    *,
    round_seq: int,
    position: int,
    tokens: list[int],
    max_draft_tokens: int,
) -> bytes:
    _check_tokens(tokens)
    if max_draft_tokens <= 0 or max_draft_tokens > MAX_TOKENS:
        raise ValueError(f"invalid max_draft_tokens: {max_draft_tokens}")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        MSG_TARGET_TO_DRAFT,
        int(round_seq),
        int(position),
        len(tokens),
        int(max_draft_tokens),
        0,
    )
    body = _TOKENS.pack(*tokens, *([0] * (MAX_TOKENS - len(tokens))))
    return (header + body).ljust(BUFFER_SIZE, b"\x00")


def decode_message(data: bytes) -> dict:
    if len(data) < _HEADER.size + _TOKENS.size:
        raise ValueError(f"buffer too small: {len(data)}")
    magic, version, msg_type, round_seq, position, num_tokens, arg0, _ = _HEADER.unpack_from(
        data, 0
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic: 0x{magic:08x}")
    if version != VERSION:
        raise ValueError(f"bad version: {version}")
    if num_tokens > MAX_TOKENS:
        raise ValueError(f"bad token count: {num_tokens}")
    all_tokens = list(_TOKENS.unpack_from(data, _HEADER.size))
    return {
        "msg_type": msg_type,
        "round_seq": round_seq,
        "position": position,
        "num_tokens": num_tokens,
        "max_draft_tokens": arg0 if msg_type == MSG_TARGET_TO_DRAFT else None,
        "tokens": all_tokens[:num_tokens],
    }
