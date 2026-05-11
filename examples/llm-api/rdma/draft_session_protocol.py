"""TCP control protocol for stateful draft sessions.

Designed alongside the 96-byte WRITE_WITH_IMM Draft API protocol but
running over a separate TCP connection so it can carry variable-length
prompt token lists (the RDMA path is fixed-length 96B, fine for
per-step token but not for prompt push).

The protocol is line-delimited JSON.  Each line is a single message:

::

    {
        "msg_type": "prompt_init",
        "request_id": 7,
        "prompt_tokens": [128000, 791, ...],
        "max_draft_len": 5,
    }

    {"msg_type": "prompt_init_ack", "request_id": 7, "status": "ok", "last_token_position": 4999}

Future message types will live here too (cancel_request, etc.).
"""

import json
import socket
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class TcpPromptInit:
    request_id: int
    prompt_tokens: List[int]
    max_draft_len: int = 5
    msg_type: str = "prompt_init"


@dataclass
class TcpPromptInitAck:
    request_id: int
    status: str = "ok"
    error: Optional[str] = None
    last_token_position: int = 0
    msg_type: str = "prompt_init_ack"


@dataclass
class TcpCancelRequest:
    request_id: int
    msg_type: str = "cancel_request"


@dataclass
class TcpCancelRequestAck:
    request_id: int
    status: str = "ok"
    msg_type: str = "cancel_request_ack"


def serialize(msg) -> bytes:
    r"""Serialize a dataclass message to a single line of JSON + ``\n``."""
    return (json.dumps(asdict(msg)) + "\n").encode("utf-8")


def parse(raw: bytes):
    """Parse a JSON line into the matching dataclass instance."""
    obj = json.loads(raw.decode("utf-8"))
    mt = obj.get("msg_type")
    if mt == TcpPromptInit.msg_type:
        return TcpPromptInit(
            request_id=int(obj["request_id"]),
            prompt_tokens=list(obj["prompt_tokens"]),
            max_draft_len=int(obj.get("max_draft_len", 5)),
        )
    if mt == TcpPromptInitAck.msg_type:
        return TcpPromptInitAck(
            request_id=int(obj["request_id"]),
            status=str(obj.get("status", "ok")),
            error=obj.get("error"),
            last_token_position=int(obj.get("last_token_position", 0)),
        )
    if mt == TcpCancelRequest.msg_type:
        return TcpCancelRequest(request_id=int(obj["request_id"]))
    if mt == TcpCancelRequestAck.msg_type:
        return TcpCancelRequestAck(
            request_id=int(obj["request_id"]),
            status=str(obj.get("status", "ok")),
        )
    raise ValueError("unknown msg_type: %r" % mt)


def recv_line(sock: socket.socket, timeout_s: float = 30.0) -> bytes:
    r"""Read one ``\n``-terminated line from ``sock``.

    Raises ``ConnectionError`` on EOF, ``socket.timeout`` on timeout.
    """
    sock.settimeout(timeout_s)
    chunks = []
    while True:
        ch = sock.recv(4096)
        if not ch:
            raise ConnectionError("peer closed during line read")
        chunks.append(ch)
        if b"\n" in ch:
            break
    return b"".join(chunks).split(b"\n", 1)[0]
