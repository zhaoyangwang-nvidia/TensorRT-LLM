# RDMA Spec-Decode 当前状态报告

更新时间：2026-05-11

## 一句话状态

主线（Phase 0-9）**全部跑通**——target 和 draft 都用 TRT-LLM、走 96B
WRITE_WITH_IMM RDMA 协议、在 docker 容器里 e2e 验证、输出字节匹配 step-3
baseline。Phase 10 GPU-initiated CUDA graph 路径**软件层就绪**，卡在
host OFED / NIC firmware 升级。

## 项目地图

```
TensorRT-LLM/tensorrt_llm/_torch/speculative/
  draft_api_protocol.py        Phase 1, 96B 字节兼容 izzy
  spec_decode_channel.py       Phase 2, imm_data 路由
  ibverbs_endpoint.py          Phase 2-3, InMem + 真 libibverbs backend
  ibverbs_draft_offload.py     Phase 4+7, nn.Module 业务层 + push_prompt
  draft_target.py              Phase 5 modified, draft_offload_v2 分支
  rdma_draft_offload_legacy.py Phase 0, step-3 baseline 备份

TensorRT-LLM/examples/llm-api/rdma/
  draft_rdma_server_v2.py      Phase 6, mock/transformers/trtllm 三 backend
  draft_session_protocol.py    Phase 7, TCP prompt-init 协议
  build_doca_extension.sh      Phase 10, izzy ext build 脚本

TensorRT-LLM/docs/source/features/rdma-izzy-refactor/
  bug_fixes/bug_fix_0[1-6].md  6 个容器侧 e2e 发现的 bug 修复记录
  phase10_doca_cuda_graph.md   Phase 10 完整记录

scripts/
  smoke_write_with_imm.py      Phase 3, 30 行 WRITE_WITH_IMM 单包
  smoke_ibverbs_endpoint.py    Phase 3, 跨 NIC 多 slot loopback
  smoke_draft_rdma_server_v2.py Phase 6, server v2 多 req
  smoke_tcp_prefill.py         Phase 7, TCP push + RDMA round-trip
  smoke_phase6_e2e.py          Phase 6, 真模型 transformers backend
```

## 已 commit 历史（pre-izzy-refactor tag 之后）

```
2618cbf52 [RDMA-Phase10] update notes: DOCA 3.3 + handler probe unblocks QP export, NIC doorbell still gated
bb79fadcb [RDMA-Phase10] DOCA SDK install + izzy extension build (GPU-export blocked)
704135207 [RDMA-PhaseX] add bug-fix notes for container-validated issues
6daeaef87 [RDMA-PhaseX] add draft_rdma_server_v2 + draft_session_protocol examples
211ff3a64 [RDMA-PhaseX-fixes] container-validated bug fixes for v2 path
1b0876694 [RDMA-Phase9] tests for multi-request, response timeout, monotonic round_seq
3cdfc353b [RDMA-Phase8] tests for _debug_trace and _tensor_row_preview
ca3493417 [RDMA-Phase7] add TCP prompt-init control channel + push_prompt
cfc6bab29 [RDMA-Phase5] wire IbverbsDraftOffloadLayer into draft_target.py (additive v2 path)
4366c2a45 [RDMA-Phase4] add IbverbsDraftOffloadLayer (nn.Module business layer)
f81d8d952 [RDMA-Phase3] hardware tests for libibverbs WRITE_WITH_IMM endpoint
1f951aebe [RDMA-Phase2] add SpecDecodeChannel + InMemoryEndpointBackend + ibverbs backend skeleton
4070f3f10 [RDMA-Phase1] add 96-byte Draft API protocol (izzy byte-compat)
a4b19d2cb [RDMA-Phase0] save step-3 baseline as rdma_draft_offload_legacy.py
```

13 个 commit，4479+ 行新增代码，DCO 签名，全部过 pre-commit。

---

# Phase 0-9：主线（已完成）

## Phase 摘要

| Phase | 内容 | 验证 |
|---|---|---|
| 0 | snapshot step-3 baseline | git tag `pre-izzy-refactor` |
| 1 | 96B Draft API 协议（字节级兼容 izzy）| 14 单测，3 组黄金 hex 对照 |
| 2 | imm_data 路由层 + InMem backend | 18 单测 |
| 3 | libibverbs WRITE_WITH_IMM endpoint | 4 单测 + 跨 NIC 真硬件 loopback 1200+ 次 round-trip |
| 4 | `IbverbsDraftOffloadLayer(nn.Module)` | 6 单测 |
| 5 | TRT-LLM 集成（additive `draft_offload_v2` 标志）| 容器 e2e |
| 6 | `draft_rdma_server_v2.py` 三 backend | mock + transformers + trtllm 都验证 |
| 7 | TCP prompt-init 控制信道 + `push_prompt` | 容器 e2e |
| 8 | `_debug_trace` + `_tensor_row_preview` | 5 单测 |
| 9 | 多 request + 超时 + round_seq | 3 单测 |

**50 个单测全绿，4 个真硬件 smoke 全过。**

## v1 / v2 双路径都 e2e 跑通

### v1 路径（legacy 4 KiB protocol）

```
target: TRT-LLM Qwen3-8B (GPU 1, mlx5_0)
        │
        │ rdma_draft_offload.py (old 4 KiB buf, output_history 同步)
        ▼
draft:  real_draft_rdma_server.py + transformers Qwen3-0.6B (GPU 2, mlx5_1)
```

容器验证：

```
$ TLLM_DRAFT_RDMA_OFFLOAD=1 python3 examples/llm-api/llm_rdma_draft_offload.py \
    --model Qwen3-8B ... --draft-port 47320
[TRT-LLM] [INFO] DraftTarget RDMA-v1 (legacy 4 KiB) enabled
...
GPUDirect RDMA enables direct data transfer between GPUs and other
devices like InfiniBand networks or NVMe SSDs, bypassing the CPU and
system
```

输出**字节级匹配** PROJECT_NOTES.md §4 step-3 baseline。

### v2 路径（izzy 字节兼容 96B WRITE_WITH_IMM）

```
target: TRT-LLM Qwen3-8B (GPU 1, mlx5_0)
        │
        │ IbverbsDraftOffloadLayer (96B WRITE_WITH_IMM, slot routing)
        ▼
draft:  draft_rdma_server_v2.py --backend trtllm + TRT-LLM Qwen3-0.6B (GPU 2, mlx5_1)
        │
        │ TRT-LLM 内部 KV cache block reuse
        ▼
        17 req/s per round, 5 draft tokens 一发
```

容器验证：

```
$ TLLM_DRAFT_RDMA_OFFLOAD_V2=1 python3 examples/llm-api/llm_rdma_draft_offload.py \
    --model Qwen3-8B ... --draft-port 47000
[TRT-LLM] [INFO] DraftTarget RDMA-v2 (izzy compatible) enabled
[draft-v2] server listening: nic=mlx5_1 port=47000 max_requests=16 backend=_TrtLlmBackend
[draft-v2] round=1 req=1 slot=1 round_seq=1 last_token=23690 → tokens=[4656,1226,...]
...
GPUDirect RDMA enables direct data transfer between GPUs and other
devices like InfiniBand networks or NVMe SSDs, bypassing the CPU and
system
```

**target 和 draft 双 TRT-LLM 进程通过 RDMA 实现投机解码** —— 用户最初提出的
目标已经达成。

## 容器侧 e2e 发现并修的 6 个 bug

| # | bug | 严重度 | commit |
|---|---|---|---|
| 1 | DynamicCache.crop 不是 tuple unpack —— Qwen3 上 KV truncate 100% raise | FATAL | `211ff3a64` |
| 2 | TCP push 用 logical request_id，server 看 RDMA slot —— session 查不到 | FATAL | `211ff3a64` |
| 3 | handshake_timeout=10s 太短，target load model 40s 期间 server 挂 | BLOCKING | `211ff3a64` |
| 4 | v2 path warmup 阶段访问 `_rdma_draft_client.request` (None) → NPE | FATAL | `211ff3a64` |
| 5 | TRT-LLM 给 per-token position_ids，layer expects per-request —— shape mismatch | FATAL | `211ff3a64` |
| 6 | `draft_rdma_server_v2.py` importlib stub 拦住真 `tensorrt_llm.LLM` import | DESIGN | `6daeaef87` |

详细记录在 `TensorRT-LLM/docs/source/features/rdma-izzy-refactor/bug_fixes/`。

---

# Phase 10：DOCA GPU-initiated CUDA graph（卡在硬件层）

## Phase 10 想做什么

主线已经把 RDMA 跑通了，但 RDMA 是 **CPU initiated**：Python ctypes 调
`ibv_post_send` → 写 doorbell → NIC 取 WR。每个 decode step 都要 GPU ↔ CPU
往返一次。

Phase 10 用 DOCA GPUNetIO，让 **GPU kernel 自己** 构造 WQE + ring NIC
doorbell：

```
                现在（Phase 3-9，ibverbs，CPU-initiated）
                ─────────────────────────
Python/CPU      每 decode step：构造 WR、写 doorbell、等 CQE
                 │
                 ▼
GPU             空闲等结果

                目标（Phase 10，DOCA，GPU-initiated）
                ─────────────────────────
Python/CPU      只做一次性 setup
                 │
                 ▼
GPU             kernel 自己构造 WQE + ring doorbell + 等 CQE
                 │  （整个 RDMA round-trip 在 CUDA graph 里录制）
                 ▼
NIC             收 doorbell → DMA → 发包
```

主要收益：消除每 decode step 的 Python ↔ GPU 同步开销，让 graph 路径生效。

## 技术栈分层

DOCA GPUNetIO 比普通 RDMA 多出"GPU 直接控制 NIC"这条路，分 7 层：

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: Python 业务层                                          │
│    target / draft 进程，调 push_prompt, layer.forward()           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: TRT-LLM extension                                      │
│    doca_rdma.so (pybind, DocaRdmaEndpoint)                       │
│    doca_rdma_ops.so (torch.ops.doca_rdma.write_with_imm)         │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: DOCA userspace SDK                                     │
│    libdoca_gpunetio.so → doca_gpunetio_verbs_export_qp(...)      │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: GPU device-memory QP mapping                           │
│    QP descriptor 从 host memory 重映射到 GPU BAR1 地址空间         │
│    (需要 nvidia_peermem + PCIe peer-to-peer)                      │
├──────────────────────────────────────────────────────────────────┤
│  Layer 5: mlx5 内核驱动                                          │
│    DEVX ioctl + dynamic UAR allocation + doorbell page mapping    │
│    版本：当前 OFED 25.07-0.9.7（DOCA 3.3 期望 25.10+）              │
├──────────────────────────────────────────────────────────────────┤
│  Layer 6: NIC firmware                                           │
│    ConnectX-7 FW 28.47.2526（理论支持），需要 DYNAMIC_UAR=1 等 flag│
├──────────────────────────────────────────────────────────────────┤
│  Layer 7: NIC PCIe DMA engine                                    │
│    doorbell ring → 从 GPU memory 取 WQE → 发包到对端 NIC          │
└──────────────────────────────────────────────────────────────────┘
```

## 每层的状态

### ✅ Layer 1-3 全部通

- DOCA SDK 3.3.0 装进 TRT-LLM 容器
- izzy 全部 C++/CUDA extension（3311 行）在 CUDA 13.1 + PyTorch 2.11 +
  GCC 13 上编出来
- 4 个产物：`doca_rdma.so`（pybind），`doca_rdma_ops.so`（torch ops），
  `libdoca_rdma_kernels.a`，`lpu_rdma_server`
- 改了 2 处 izzy 源码 bug 才编过：

  - `src/doca_rdma_endpoint.cpp:1324` 用 `::listen` 避开和成员函数 `listen` 名冲突
  - `CMakeLists.txt` 加 `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)`，
    pybind11 的 LTO 把 CUDA fatbin 符号 gc 掉的问题

- Python import + `torch.ops.doca_rdma.write_with_imm` 注册成功
- `DocaRdmaEndpoint.init/alloc_doca_gpu_buffer` 单 endpoint 工作

### ⚠️ Layer 4 部分通（需要 env override）

`doca_gpunetio_verbs_def.h` 定义 5 种 NIC handler：

| handler id | 名字 | 说明 | 这台机器 |
|---|---|---|---|
| 0 | AUTO | SDK 自动选最快可用 | ✅ export OK |
| 1 | CPU_PROXY | CPU 帮 GPU ring doorbell | ✅ export OK |
| 2 | GPU_SM_DB | GPU SM 直接 ring 标准 doorbell | ❌ "DOCA Driver call failure" |
| 3 | GPU_SM_BF | GPU SM 用 BlueFlame doorbell | ❌ 同上 |
| 4 | GPU_SM_NO_DBR | 不更新 DBREC（最快）| ❌ 同上 |

izzy 默认只试 id=2（最严格、最快）。这台机器 id=2/3/4 全 fail。设：

```bash
export GLUE_RDMA_NIC_HANDLERS=0          # 试 AUTO
export GLUE_RDMA_ALLOW_PROXY_HANDLERS=1  # 允许 CPU_PROXY 入候选
```

之后 handler=0 成功 export QP 到 GPU，两端 `connect_to_peer` 和
`exchange_buffer_info` 都 True。

### ❌ Layer 5-7 卡死

实际发 WRITE_WITH_IMM 时：

```python
torch.ops.doca_rdma.write_with_imm(qp_ptr, ...)  # ✓ wqe_idx=0
torch.cuda.synchronize()                          # ✓ 返回

torch.ops.doca_rdma.wait_send(qp_ptr, ...)        # ✗ 永远 hang
```

server 端 5 秒后读 recv buffer：

```
recv buf first 32 bytes: [0, 0, 0, 0, 0, 0, 0, 0, ...]   ← 全 0
payload pattern matches expected (0xa0..0xff): False
```

WQE 在 GPU 上构造好了，但 NIC 没去取。doorbell ring 不生效。

## Path 1（容器内换 DOCA）的成果

升级 DOCA 3.2 → 3.3 + 找到 handler 候选 env 之后：

| 测试维度 | path 1 之前 | path 1 之后 |
|---|---|---|
| izzy ext 编译 | DOCA 3.2 ✓ | DOCA 3.3 ✓（零代码改） |
| `verbs_export_qp` 任何 handler | ❌ 全 fail | ✅ handler 0/1 通 |
| QP 真正映到 GPU device space | ❌ | ✅ |
| `get_qp_dev_ptr` 返回有效 pointer | ❌ | ✅ |
| GPU kernel 写 WQE | ❌（没 QP）| ✅ |
| **NIC 取 WQE 发包** | ❌ | ❌ |
| send/recv CQE | ❌ | ❌ |
| CUDA graph 录 RDMA round-trip | ❌ | ❌ |

把卡死的位置从 layer 4 推到 layer 5-7。**软件做完了，剩下的在
host kernel + NIC firmware**。

## Path 1 没解决的问题，为什么是硬件层

DOCA 3.3 release notes 明确 cert OFED 最低 25.10。我们 host 是 25.07-0.9.7，
晚一个 minor。两个版本之间，mlx5_core 改了：

- DEVX ioctl 的 GPU peer-to-peer 接口加了新 cmd
- UAR 分配 dynamic 模式 default off → default on
- BAR1 doorbell page 跟 GPU 共享映射的 ABI 改了

这些**全在 host kernel 里**，容器内动不了。需要：

```bash
# host 上，root 权限
sudo apt install mlnx-ofed-all=25.10-*
sudo /etc/init.d/openibd restart
# 或者 reboot
```

或者换一台已经在跑 25.10+ 的机器。

## 当前硬件 / 软件版本盘点

```
GPU:                 NVIDIA B200, sm_100
NIC:                 ConnectX-7 (vendor_part_id=4129/MT2910)
NIC firmware:        28.47.2526
host kernel:         6.8.0-101-generic
host MLNX OFED:      25.07-0.9.7      ← 卡这里
DOCA SDK (container): 3.3.0
NVIDIA driver:       590.48.01
CUDA driver:         13.1
PyTorch (container): 2.11.0a0+eb65b36914.nv26.02
TRT-LLM:             1.3.0rc13
nvidia_peermem:      loaded ✓
```

---

# 下一步的可能性

## A. 等 sysadmin 升 host OFED 到 25.10+（推荐，最干净）

- **优点**：治本，升上后大概率所有 layer 通
- **缺点**：要 root + 可能 reboot，外部协调成本
- **时间**：取决于 sysadmin 排期
- **风险**：升级可能动 host 别的服务

## B. 给 NV DOCA team 报 ticket

- **内容**：DOCA 3.3 + OFED 25.07 + ConnectX-7 FW 28.47 + B200 sm_100
  组合下，handler 0/1 export OK 但 doorbell 不 trigger；handler 2/3/4
  直接 export fail
- **优点**：拿到官方 cert matrix；NV 可能给 mlxconfig flag 解决
- **缺点**：等回复 1-2 周
- **时间**：异步

## C. 不等硬件，先把 software 层准备好（~3 小时）

- 写 `tensorrt_llm/_torch/speculative/doca_endpoint.py`（~200 行）
  - 实现 `EndpointBackend` Protocol（跟 `_IbverbsRdmaBackend` 一样接口）
  - runtime 检测 DOCA SDK / izzy ext 是否可用
  - 不可用时 `start()` 直接 `return EndpointStatus.kUnsupportedBackend`
- 在 `IbverbsDraftOffloadConfig` 加 `transport: Literal["ibverbs","doca"]="ibverbs"`
- 恢复 `_run_batch1_graph_fast_path`（v2 layer 里现在 return None）：
  - 用 `torch.ops.doca_rdma.write_with_imm` + `wait_send` + `poll_recv`
  - 录入 `torch.cuda.graph` capture
- **优点**：OFED 升上来时 toggle 一下就启用；不浪费当前等待时间
- **缺点**：没硬件 e2e 验证，只能静态测 + 单测
- **风险**：API 跟实际硬件 work 时的 spec 可能有偏差

## D. 换机器

- 找一台已经在跑 OFED 25.10+ 或者 NIM 镜像默认 DOCA-ready 的
- **优点**：直接绕过
- **缺点**：项目本来的硬件就这台

# 总结

| 项 | 状态 |
|---|---|
| Phase 0-9 主线（ibverbs 4 层架构）| ✅ 100% 完成 + 容器 e2e 验证 |
| v1 path（legacy 4 KiB）| ✅ 输出字节匹配 baseline |
| v2 path（izzy 96B WRITE_WITH_IMM）| ✅ 输出字节匹配 baseline |
| draft 端 TRT-LLM backend | ✅ 17 req/s, 真模型跑通 |
| Phase 10 软件层（DOCA SDK + izzy ext + torch ops）| ✅ 编译、加载、注册全 OK |
| Phase 10 layer 4 QP export | ✅ 需要 env override |
| Phase 10 layer 5-7 NIC doorbell + DMA | ❌ host OFED 限制 |
| Phase 10 CUDA graph fast path | ❌ 依赖上面 |

主线目标——**两个 TRT-LLM 程序通过 RDMA 投机解码**——**已经达成**。
Phase 10 是优化路径，软件就绪，等 host OFED 升级即可启用。

推荐 A + C 并行：升级是治本，并行写 software 层不浪费等待时间。

---

## 增补 (post-debug)：Phase 10 根因确认

经多层 debug + 查 NVIDIA DOCA 3.3 release notes，**Phase 10 卡点是 OFED 版本**：

- **Certified**: DOCA 3.3 + ConnectX-7 + B200 sm_100 需要 **MLNX_OFED 25.10**
- **本机**: MLNX_OFED **25.07**
- **缺失**: 25.07 没包含 sm_100 GPU Register 直接访问 flag —— 这正是
  `doca_gpu_verbs_export_qp(GPU_SM_DB)` 的依赖
- **结论**: 不是软件 bug、不是配置错误、不是 SDK 版本问题（DOCA 3.2/3.3 试过都失败）；
  唯一治本路径是 host **MLNX_OFED 25.07 → 25.10**

详细见 `phase10_doca_cuda_graph.md` 文末"官方兼容矩阵确认"段。

debug 期间的关键证据：

1. `ibv_reg_mr(GPU_ptr)` 成功 — 数据 P2P (NIC ↔ GPU memory) 已通，nvidia_peermem OK
2. `doca_gpu_verbs_export_qp(GPU_SM_DB)` 失败 — 控制路径 (GPU SM → NIC BAR) 因 OFED
   差 minor 版本而 broken
3. GPU memory dump 显示 WQE 被 GPU kernel 正确写入（producer index +1）
4. NIC `rx_write_requests` counter 测试前后一致 — NIC 完全不知道有 WQE 等着
5. 不是 PCIe peer-to-peer 整体不通；是非常具体的"GPU 反向写 NIC doorbell"这一根线
