# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DOCA RDMA target client driving N speculative-decoding rounds.

Connects to a DOCA CPU_PROXY draft endpoint, sends a
TargetToDraft frame each round, then receives the draft's reply.

To keep this harness focused on the transport (rather than a real
verify model), each round we treat the round as fully accepted: we
take the last draft token from round R as our ``last_token`` for round
R+1, and we bump position by ``max_draft_len`` each round.

For a real workload, plug this into ``IbverbsDraftOffloadLayer`` or the
target's verify path — the wire format here is the same one the
production layer speaks.

Run:

    GPU_ID=1 NIC=mlx5_1 PORT=47820 HOST=127.0.0.1 \
        python3 doca_target_client.py --rounds 3 --max-draft-len 4
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
import torch  # noqa: E402

torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
import doca_rdma  # noqa: E402
from doca_poll_channel import DocaPollChannel  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
_SPEC_ROOT = os.path.join(_TRT, "tensorrt_llm", "_torch", "speculative")
sys.path.insert(0, _SPEC_ROOT)
spec = importlib.util.spec_from_file_location(
    "draft_api_protocol", os.path.join(_SPEC_ROOT, "draft_api_protocol.py")
)
draft_api_protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft_api_protocol)
DraftApiProtocol = draft_api_protocol.DraftApiProtocol


def _make_target_to_draft(round_seq, position, last_token):
    """Build a 96-byte TargetToDraft frame."""
    tokens = [int(last_token)] + [0] * (DraftApiProtocol.kMaxTokens - 1)
    return DraftApiProtocol.Message(
        version=DraftApiProtocol.kVersionV1,
        message_type=DraftApiProtocol.MessageType.kTargetToDraft,
        round_seq_num=round_seq,
        position=position,
        num_tokens=1,
        tokens=tokens,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-draft-len", type=int, default=4)
    ap.add_argument("--initial-token", type=int, default=42)
    args = ap.parse_args()

    gpu = int(os.environ["GPU_ID"])
    nic = os.environ["NIC"]
    port = int(os.environ["PORT"])
    host = os.environ["HOST"]
    peer_name = "doca_draft"

    torch.cuda.set_device(gpu)
    ep = doca_rdma.DocaRdmaEndpoint()
    ep.init(gpu, nic)
    chan = DocaPollChannel(ep, peer_name, gpu)
    ep.connect_to_peer(host, port, peer_name)
    ep.exchange_buffer_info(peer_name, chan.send_buf, chan.recv_buf)
    print(f"[target] connected, cpu_proxy_enabled={ep.cpu_proxy_enabled(peer_name)}", flush=True)

    chan.start_progress()

    position = 0
    last_token = args.initial_token
    accepted_history = []
    for r in range(args.rounds):
        req = _make_target_to_draft(round_seq=r, position=position, last_token=last_token)
        status, frame = DraftApiProtocol.encode(req)
        assert status == DraftApiProtocol.Status.kOk, DraftApiProtocol.to_string(status)
        chan.send_msg(frame)
        print(
            f"[target] round {r} sent TargetToDraft pos={position} last_token={last_token}",
            flush=True,
        )

        wire = chan.recv_msg(timeout_s=120.0)
        status, msg = DraftApiProtocol.decode(wire)
        assert status == DraftApiProtocol.Status.kOk, DraftApiProtocol.to_string(status)
        draft_tokens = msg.tokens[: msg.num_tokens]
        accepted_history.extend(draft_tokens)
        print(
            f"[target] round {r} recv DraftToTarget num_tokens={msg.num_tokens} "
            f"draft_tokens={draft_tokens}",
            flush=True,
        )

        # Accept-all verify stub: next round's last_token = last draft
        # token of this round; position advances by max_draft_len.
        position += args.max_draft_len
        last_token = draft_tokens[-1] if draft_tokens else last_token

    chan.stop()
    print(f"[target] DONE accepted={accepted_history}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
