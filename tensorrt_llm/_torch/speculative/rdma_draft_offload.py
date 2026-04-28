# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Minimal target-side GPUDirect RDMA draft offload client.

This is intentionally small and targets the first bring-up case: one target
process connects to a fake draft process, writes a fixed-size request into the
draft GPU buffer, and reads back fixed-size draft tokens.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from .rdma_draft_protocol import (
    BUFFER_SIZE,
    MAGIC,
    MSG_DRAFT_TO_TARGET,
    decode_message,
    encode_request,
)

_libverbs = ctypes.CDLL("libibverbs.so.1", use_errno=True)


def _load_wrapper() -> ctypes.CDLL:
    candidates = []
    env_path = os.environ.get("TLLM_RDMA_IBV_WRAPPER_LIB")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            "/home/scratch.zhaoyangw_gpu/rdma/libibv_wrapper.so",
            os.path.join(os.getcwd(), "libibv_wrapper.so"),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            return ctypes.CDLL(path, use_errno=True)
    raise RuntimeError(
        "Could not find libibv_wrapper.so. Set TLLM_RDMA_IBV_WRAPPER_LIB "
        "or run from /home/scratch.zhaoyangw_gpu/rdma."
    )


_libwrap = _load_wrapper()

IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4
IBV_ACCESS_REMOTE_ATOMIC = 8
IBV_ACCESS_RELAXED_ORDERING = 1 << 20

IBV_QPT_RC = 2
IBV_QPS_INIT = 1
IBV_QPS_RTR = 2
IBV_QPS_RTS = 3

IBV_WR_RDMA_WRITE = 0
IBV_SEND_SIGNALED = 2

_WC_STATUS_NAMES = {
    0: "IBV_WC_SUCCESS",
    1: "IBV_WC_LOC_LEN_ERR",
    2: "IBV_WC_LOC_QP_OP_ERR",
    3: "IBV_WC_LOC_EEC_OP_ERR",
    4: "IBV_WC_LOC_PROT_ERR",
    5: "IBV_WC_WR_FLUSH_ERR",
    6: "IBV_WC_MW_BIND_ERR",
    7: "IBV_WC_BAD_RESP_ERR",
    8: "IBV_WC_LOC_ACCESS_ERR",
    9: "IBV_WC_REM_INV_REQ_ERR",
    10: "IBV_WC_REM_ACCESS_ERR",
    11: "IBV_WC_REM_OP_ERR",
    12: "IBV_WC_RETRY_EXC_ERR",
    13: "IBV_WC_RNR_RETRY_EXC_ERR",
    14: "IBV_WC_LOC_RDD_VIOL_ERR",
    15: "IBV_WC_REM_INV_RD_REQ_ERR",
    16: "IBV_WC_REM_ABORT_ERR",
    17: "IBV_WC_INV_EECN_ERR",
    18: "IBV_WC_INV_EEC_STATE_ERR",
    19: "IBV_WC_FATAL_ERR",
    20: "IBV_WC_RESP_TIMEOUT_ERR",
    21: "IBV_WC_GENERAL_ERR",
}

IBV_QP_STATE = 1 << 0
IBV_QP_ACCESS_FLAGS = 1 << 3
IBV_QP_PKEY_INDEX = 1 << 4
IBV_QP_PORT = 1 << 5
IBV_QP_AV = 1 << 7
IBV_QP_PATH_MTU = 1 << 8
IBV_QP_TIMEOUT = 1 << 9
IBV_QP_RETRY_CNT = 1 << 10
IBV_QP_RNR_RETRY = 1 << 11
IBV_QP_RQ_PSN = 1 << 12
IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13
IBV_QP_MIN_RNR_TIMER = 1 << 15
IBV_QP_SQ_PSN = 1 << 16
IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17
IBV_QP_DEST_QPN = 1 << 20

MR_OFF_LKEY = 36
MR_OFF_RKEY = 40
QP_OFF_QP_NUM = 52
PORT_ATTR_SIZE = 56


class ibv_sge(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
    ]


class ibv_wc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
        ("src_qp", ctypes.c_uint32),
        ("wc_flags", ctypes.c_uint),
        ("pkey_index", ctypes.c_uint16),
        ("slid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("dlid_path_bits", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 2),
    ]


_libverbs.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
_libverbs.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
_libverbs.ibv_free_device_list.restype = None
_libverbs.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libverbs.ibv_get_device_name.restype = ctypes.c_char_p
_libverbs.ibv_get_device_name.argtypes = [ctypes.c_void_p]
_libverbs.ibv_open_device.restype = ctypes.c_void_p
_libverbs.ibv_open_device.argtypes = [ctypes.c_void_p]
_libverbs.ibv_close_device.restype = ctypes.c_int
_libverbs.ibv_close_device.argtypes = [ctypes.c_void_p]
_libverbs.ibv_query_port.restype = ctypes.c_int
_libverbs.ibv_query_port.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint8,
    ctypes.c_void_p,
]
_libverbs.ibv_query_gid.restype = ctypes.c_int
_libverbs.ibv_query_gid.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint8,
    ctypes.c_int,
    ctypes.c_void_p,
]
_libverbs.ibv_alloc_pd.restype = ctypes.c_void_p
_libverbs.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
_libverbs.ibv_dealloc_pd.restype = ctypes.c_int
_libverbs.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
_libverbs.ibv_reg_mr.restype = ctypes.c_void_p
_libverbs.ibv_reg_mr.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
_libverbs.ibv_dereg_mr.restype = ctypes.c_int
_libverbs.ibv_dereg_mr.argtypes = [ctypes.c_void_p]
_libverbs.ibv_create_cq.restype = ctypes.c_void_p
_libverbs.ibv_create_cq.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
_libverbs.ibv_destroy_cq.restype = ctypes.c_int
_libverbs.ibv_destroy_cq.argtypes = [ctypes.c_void_p]
_libverbs.ibv_create_qp.restype = ctypes.c_void_p
_libverbs.ibv_create_qp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_libverbs.ibv_destroy_qp.restype = ctypes.c_int
_libverbs.ibv_destroy_qp.argtypes = [ctypes.c_void_p]
_libverbs.ibv_modify_qp.restype = ctypes.c_int
_libverbs.ibv_modify_qp.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

_libwrap.wrap_ibv_post_send.restype = ctypes.c_int
_libwrap.wrap_ibv_post_send.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_libwrap.wrap_ibv_poll_cq.restype = ctypes.c_int
_libwrap.wrap_ibv_poll_cq.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def _buf(size: int) -> ctypes.Array:
    return (ctypes.c_uint8 * size)()


def _voidp(obj) -> ctypes.c_void_p:
    return ctypes.cast(ctypes.addressof(obj), ctypes.c_void_p)


def _set8(buf, offset: int, val: int) -> None:
    struct.pack_into("B", buf, offset, val & 0xFF)


def _set16(buf, offset: int, val: int) -> None:
    struct.pack_into("<H", buf, offset, val & 0xFFFF)


def _set32(buf, offset: int, val: int) -> None:
    struct.pack_into("<I", buf, offset, val & 0xFFFFFFFF)


def _set64(buf, offset: int, val: int) -> None:
    struct.pack_into("<Q", buf, offset, val & 0xFFFFFFFFFFFFFFFF)


def _set_ptr(buf, offset: int, ptr_val: int) -> None:
    struct.pack_into("<Q", buf, offset, ptr_val & 0xFFFFFFFFFFFFFFFF)


def _check(ret: int, op: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{op} failed: ret={ret} errno={ctypes.get_errno()}")


def _read_u32(ptr: ctypes.c_void_p, offset: int) -> int:
    addr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8))
    return struct.unpack_from("<I", bytes(addr[offset : offset + 4]))[0]


def _mr_lkey(mr: ctypes.c_void_p) -> int:
    return _read_u32(mr, MR_OFF_LKEY)


def _mr_rkey(mr: ctypes.c_void_p) -> int:
    return _read_u32(mr, MR_OFF_RKEY)


def _qp_num(qp: ctypes.c_void_p) -> int:
    return _read_u32(qp, QP_OFF_QP_NUM)


def _find_ib_device(name: str) -> ctypes.c_void_p:
    state_path = f"/sys/class/infiniband/{name}/ports/1/state"
    lid_path = f"/sys/class/infiniband/{name}/ports/1/lid"
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as state_file:
            state = state_file.read().strip()
        lid = "unknown"
        if os.path.exists(lid_path):
            with open(lid_path, encoding="utf-8") as lid_file:
                lid = lid_file.read().strip()
        if not state.startswith("4: ACTIVE"):
            raise RuntimeError(
                f"IB device {name} port 1 is not ACTIVE "
                f"(state={state}, lid={lid}). Choose an active NIC."
            )

    num = ctypes.c_int(0)
    devs = _libverbs.ibv_get_device_list(ctypes.byref(num))
    if not devs:
        raise RuntimeError("ibv_get_device_list returned null")
    found = None
    try:
        for i in range(num.value):
            dev = devs[i]
            dev_name = _libverbs.ibv_get_device_name(dev)
            if dev_name and dev_name.decode() == name:
                found = dev
                break
    finally:
        _libverbs.ibv_free_device_list(devs)
    if found is None:
        raise RuntimeError(f"IB device not found: {name}")
    return found


def _query_port(ctx: ctypes.c_void_p, port: int = 1) -> dict:
    port_attr = _buf(PORT_ATTR_SIZE)
    _check(_libverbs.ibv_query_port(ctx, port, _voidp(port_attr)), "ibv_query_port")
    lid = struct.unpack_from("<H", port_attr, 34)[0]
    active_mtu = struct.unpack_from("<i", port_attr, 8)[0]
    gid_buf = _buf(16)
    _check(_libverbs.ibv_query_gid(ctx, port, 0, _voidp(gid_buf)), "ibv_query_gid")
    return {"lid": lid, "active_mtu": active_mtu, "gid": bytes(gid_buf)}


def _make_qp_init_attr(send_cq, recv_cq) -> ctypes.Array:
    buf = _buf(64)
    _set_ptr(buf, 8, send_cq)
    _set_ptr(buf, 16, recv_cq)
    _set32(buf, 32, 16)
    _set32(buf, 36, 16)
    _set32(buf, 40, 1)
    _set32(buf, 44, 1)
    _set32(buf, 52, IBV_QPT_RC)
    return buf


def _qp_to_init(qp, port: int = 1) -> None:
    attr = _buf(144)
    _set32(attr, 0, IBV_QPS_INIT)
    _set32(
        attr,
        32,
        IBV_ACCESS_LOCAL_WRITE
        | IBV_ACCESS_REMOTE_WRITE
        | IBV_ACCESS_REMOTE_READ
        | IBV_ACCESS_REMOTE_ATOMIC,
    )
    _set16(attr, 120, 0)
    _set8(attr, 129, port)
    mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
    _check(_libverbs.ibv_modify_qp(qp, _voidp(attr), mask), "ibv_modify_qp INIT")


def _qp_to_rtr(
    qp,
    remote_lid: int,
    remote_qpn: int,
    remote_psn: int,
    *,
    path_mtu: int,
    remote_gid: bytes,
    port: int = 1,
) -> None:
    attr = _buf(144)
    _set32(attr, 0, IBV_QPS_RTR)
    _set32(attr, 8, path_mtu)
    _set32(attr, 20, remote_psn)
    _set32(attr, 28, remote_qpn)
    for i, value in enumerate(remote_gid[:16]):
        _set8(attr, 56 + i, value)
    _set8(attr, 76, 0)
    _set8(attr, 77, 64)
    _set16(attr, 80, remote_lid)
    _set8(attr, 85, 1)
    _set8(attr, 86, port)
    _set8(attr, 127, 1)
    _set8(attr, 128, 12)
    mask = (
        IBV_QP_STATE
        | IBV_QP_AV
        | IBV_QP_PATH_MTU
        | IBV_QP_DEST_QPN
        | IBV_QP_RQ_PSN
        | IBV_QP_MAX_DEST_RD_ATOMIC
        | IBV_QP_MIN_RNR_TIMER
    )
    _check(_libverbs.ibv_modify_qp(qp, _voidp(attr), mask), "ibv_modify_qp RTR")


def _qp_to_rts(qp, sq_psn: int) -> None:
    attr = _buf(144)
    _set32(attr, 0, IBV_QPS_RTS)
    _set32(attr, 24, sq_psn)
    _set8(attr, 126, 1)
    _set8(attr, 130, 14)
    _set8(attr, 131, 7)
    _set8(attr, 132, 7)
    mask = (
        IBV_QP_STATE
        | IBV_QP_TIMEOUT
        | IBV_QP_RETRY_CNT
        | IBV_QP_RNR_RETRY
        | IBV_QP_SQ_PSN
        | IBV_QP_MAX_QP_RD_ATOMIC
    )
    _check(_libverbs.ibv_modify_qp(qp, _voidp(attr), mask), "ibv_modify_qp RTS")


def _poll_cq_blocking(cq, timeout_s: float = 10.0) -> ibv_wc:
    wc = ibv_wc()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = _libwrap.wrap_ibv_poll_cq(cq, 1, ctypes.addressof(wc))
        if n > 0:
            return wc
        if n < 0:
            raise RuntimeError("ibv_poll_cq returned error")
        time.sleep(0.00001)
    raise TimeoutError("poll_cq timeout")


@dataclass
class _LocalInfo:
    lid: int
    qpn: int
    psn: int
    gid: bytes
    send_addr: int
    send_lkey: int
    recv_addr: int
    recv_rkey: int

    def to_wire(self) -> dict:
        return {
            "lid": self.lid,
            "qpn": self.qpn,
            "psn": self.psn,
            "gid": self.gid.hex(),
            "send_addr": self.send_addr,
            "send_lkey": self.send_lkey,
            "recv_addr": self.recv_addr,
            "recv_rkey": self.recv_rkey,
        }

    @staticmethod
    def from_wire(raw: dict) -> "_LocalInfo":
        return _LocalInfo(
            lid=int(raw["lid"]),
            qpn=int(raw["qpn"]),
            psn=int(raw["psn"]),
            gid=bytes.fromhex(raw["gid"]),
            send_addr=int(raw["send_addr"]),
            send_lkey=int(raw["send_lkey"]),
            recv_addr=int(raw["recv_addr"]),
            recv_rkey=int(raw["recv_rkey"]),
        )


class _JsonLineSocket:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb", buffering=0)

    def send(self, msg: dict) -> None:
        self.file.write(json.dumps(msg, separators=(",", ":")).encode() + b"\n")

    def recv(self) -> dict:
        line = self.file.readline()
        if not line:
            raise EOFError("RDMA draft control socket closed")
        return json.loads(line.decode())

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()


@dataclass
class RdmaDraftOffloadConfig:
    nic_name: str = "mlx5_0"
    server_host: str = "127.0.0.1"
    server_port: int = 47320
    gpu_id: Optional[int] = None
    max_draft_len: int = 5
    buffer_size: int = BUFFER_SIZE


class RdmaDraftOffloadClient:
    def __init__(self, config: RdmaDraftOffloadConfig):
        self.config = config
        self.started = False
        self.device: Optional[torch.device] = None
        self.send_tensor: Optional[torch.Tensor] = None
        self.recv_tensor: Optional[torch.Tensor] = None
        self.ibctx = None
        self.pd = None
        self.cq = None
        self.qp = None
        self.send_mr = None
        self.recv_mr = None
        self.local_info: Optional[_LocalInfo] = None
        self.remote_info: Optional[_LocalInfo] = None
        self.control: Optional[_JsonLineSocket] = None
        self.round_seq = 1

    def start(self, device: torch.device) -> None:
        if self.started:
            return
        self.device = device
        gpu_id = self.config.gpu_id
        if gpu_id is None:
            gpu_id = device.index
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.send_tensor = torch.empty(self.config.buffer_size, dtype=torch.uint8, device=device)
        self.recv_tensor = torch.empty(self.config.buffer_size, dtype=torch.uint8, device=device)
        self.recv_tensor.zero_()
        torch.cuda.synchronize(device)

        dev_ptr = _find_ib_device(self.config.nic_name)
        self.ibctx = _libverbs.ibv_open_device(dev_ptr)
        if not self.ibctx:
            raise RuntimeError("ibv_open_device failed")
        self.pd = _libverbs.ibv_alloc_pd(self.ibctx)
        if not self.pd:
            raise RuntimeError("ibv_alloc_pd failed")
        self.cq = _libverbs.ibv_create_cq(self.ibctx, 32, None, None, 0)
        if not self.cq:
            raise RuntimeError("ibv_create_cq failed")
        init_attr = _make_qp_init_attr(self.cq, self.cq)
        self.qp = _libverbs.ibv_create_qp(self.pd, _voidp(init_attr))
        if not self.qp:
            raise RuntimeError("ibv_create_qp failed")

        access = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
            | IBV_ACCESS_RELAXED_ORDERING
        )
        self.send_mr = _libverbs.ibv_reg_mr(
            self.pd,
            ctypes.c_void_p(int(self.send_tensor.data_ptr())),
            self.config.buffer_size,
            access,
        )
        if not self.send_mr:
            raise RuntimeError(f"ibv_reg_mr send failed: errno={ctypes.get_errno()}")
        self.recv_mr = _libverbs.ibv_reg_mr(
            self.pd,
            ctypes.c_void_p(int(self.recv_tensor.data_ptr())),
            self.config.buffer_size,
            access,
        )
        if not self.recv_mr:
            raise RuntimeError(f"ibv_reg_mr recv failed: errno={ctypes.get_errno()}")

        port_info = _query_port(self.ibctx)
        self.local_info = _LocalInfo(
            lid=port_info["lid"],
            qpn=_qp_num(self.qp),
            psn=(int(time.time_ns()) ^ int(gpu_id)) & 0xFFFFFF,
            gid=port_info["gid"],
            send_addr=int(self.send_tensor.data_ptr()),
            send_lkey=_mr_lkey(self.send_mr),
            recv_addr=int(self.recv_tensor.data_ptr()),
            recv_rkey=_mr_rkey(self.recv_mr),
        )

        sock = socket.create_connection(
            (self.config.server_host, self.config.server_port), timeout=30
        )
        self.control = _JsonLineSocket(sock)
        self.control.send({"type": "rdma_info", "info": self.local_info.to_wire()})
        msg = self.control.recv()
        if msg.get("type") != "rdma_info":
            raise RuntimeError(f"expected rdma_info, got {msg}")
        self.remote_info = _LocalInfo.from_wire(msg["info"])

        _qp_to_init(self.qp)
        _qp_to_rtr(
            self.qp,
            self.remote_info.lid,
            self.remote_info.qpn,
            self.remote_info.psn,
            path_mtu=port_info["active_mtu"],
            remote_gid=self.remote_info.gid,
        )
        _qp_to_rts(self.qp, self.local_info.psn)
        self.started = True
        logger.info(
            "RDMA draft offload connected: gpu=%s nic=%s remote=%s:%s "
            "local(lid=%s qpn=0x%x psn=0x%x send=0x%x/lkey=0x%x recv=0x%x/rkey=0x%x) "
            "remote(lid=%s qpn=0x%x psn=0x%x recv=0x%x/rkey=0x%x)",
            gpu_id,
            self.config.nic_name,
            self.config.server_host,
            self.config.server_port,
            self.local_info.lid,
            self.local_info.qpn,
            self.local_info.psn,
            self.local_info.send_addr,
            self.local_info.send_lkey,
            self.local_info.recv_addr,
            self.local_info.recv_rkey,
            self.remote_info.lid,
            self.remote_info.qpn,
            self.remote_info.psn,
            self.remote_info.recv_addr,
            self.remote_info.recv_rkey,
        )

    def close(self) -> None:
        if self.control is not None:
            self.control.close()
            self.control = None
        if self.recv_mr:
            _libverbs.ibv_dereg_mr(self.recv_mr)
            self.recv_mr = None
        if self.send_mr:
            _libverbs.ibv_dereg_mr(self.send_mr)
            self.send_mr = None
        if self.qp:
            _libverbs.ibv_destroy_qp(self.qp)
            self.qp = None
        if self.cq:
            _libverbs.ibv_destroy_cq(self.cq)
            self.cq = None
        if self.pd:
            _libverbs.ibv_dealloc_pd(self.pd)
            self.pd = None
        if self.ibctx:
            _libverbs.ibv_close_device(self.ibctx)
            self.ibctx = None
        self.started = False

    def _write_send_tensor(self, payload: bytes) -> None:
        assert self.send_tensor is not None
        padded = payload[: self.config.buffer_size].ljust(self.config.buffer_size, b"\x00")
        cpu = torch.frombuffer(bytearray(padded), dtype=torch.uint8)
        self.send_tensor.copy_(cpu.to(self.send_tensor.device), non_blocking=False)
        torch.cuda.synchronize(self.send_tensor.device)

    def _read_recv_tensor(self) -> bytes:
        assert self.recv_tensor is not None
        torch.cuda.synchronize(self.recv_tensor.device)
        return self.recv_tensor.detach().cpu().numpy().tobytes()

    def _rdma_write_to_peer_recv(self, wr_id: int) -> None:
        assert self.local_info is not None
        assert self.remote_info is not None
        sge = ibv_sge()
        sge.addr = self.local_info.send_addr
        sge.length = self.config.buffer_size
        sge.lkey = self.local_info.send_lkey

        wr = _buf(128)
        _set64(wr, 0, wr_id)
        _set_ptr(wr, 8, 0)
        _set_ptr(wr, 16, ctypes.addressof(sge))
        _set32(wr, 24, 1)
        _set32(wr, 28, IBV_WR_RDMA_WRITE)
        _set32(wr, 32, IBV_SEND_SIGNALED)
        _set32(wr, 36, 0)
        _set64(wr, 40, self.remote_info.recv_addr)
        _set32(wr, 48, self.remote_info.recv_rkey)

        bad_wr = ctypes.c_void_p(0)
        _check(
            _libwrap.wrap_ibv_post_send(self.qp, _voidp(wr), ctypes.byref(bad_wr)),
            "ibv_post_send",
        )
        wc = _poll_cq_blocking(self.cq)
        if wc.status != 0:
            status_name = _WC_STATUS_NAMES.get(wc.status, "UNKNOWN")
            raise RuntimeError(
                "RDMA write failed: "
                f"wc.status={wc.status}({status_name}) "
                f"vendor_err={wc.vendor_err} "
                f"local_lid={self.local_info.lid} "
                f"local_qpn=0x{self.local_info.qpn:x} "
                f"remote_lid={self.remote_info.lid} "
                f"remote_qpn=0x{self.remote_info.qpn:x} "
                f"remote_addr=0x{self.remote_info.recv_addr:x} "
                f"remote_rkey=0x{self.remote_info.recv_rkey:x}"
            )

    def _notify(self, msg_type: str, **fields) -> None:
        assert self.control is not None
        self.control.send({"type": msg_type, **fields})

    def _wait_for(self, msg_type: str) -> dict:
        assert self.control is not None
        while True:
            msg = self.control.recv()
            if msg.get("type") == msg_type:
                return msg

    def _wait_recv_magic(self, timeout_s: float = 5.0) -> bytes:
        deadline = time.time() + timeout_s
        magic_bytes = int(MAGIC).to_bytes(4, "little")
        last = b""
        while time.time() < deadline:
            last = self._read_recv_tensor()
            if last[:4] == magic_bytes:
                return last
            time.sleep(0.001)
        raise TimeoutError(f"RDMA draft response did not arrive; last_head64={last[:64].hex()}")

    def request(
        self,
        *,
        tokens: list[int],
        position: int,
        max_draft_len: int,
        device: torch.device,
    ) -> list[int]:
        self.start(device)
        round_seq = self.round_seq
        self.round_seq += 1
        payload = encode_request(
            round_seq=round_seq,
            position=position,
            tokens=tokens,
            max_draft_tokens=max_draft_len,
        )
        self._write_send_tensor(payload)
        self._rdma_write_to_peer_recv(round_seq)
        self._notify("request_done", round_seq=round_seq)
        self._wait_for("response_done")
        response = decode_message(self._wait_recv_magic())
        if response["msg_type"] != MSG_DRAFT_TO_TARGET:
            raise RuntimeError(f"unexpected RDMA draft response: {response}")
        return [int(t) for t in response["tokens"][:max_draft_len]]
