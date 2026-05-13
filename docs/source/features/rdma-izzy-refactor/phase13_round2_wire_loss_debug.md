<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 3 DOCA CPU_PROXY: round-2 wire-loss 调试日志

本文档记录 2026-05-12 这一轮针对 Phase 3 spec-decode e2e 多轮 byte-match
的全部调试过程。问题简单清楚 — "round 2 的 RDMA write 没到 draft" —
但找根因花了好几轮逐层剥离。

## 一、目标

**Phase 3 = DOCA CPU_PROXY transport spec-decode e2e**：

* target 跑独立 TRT-LLM 进程，加载 Qwen3-4B，开启 DraftTargetDecoding
  spec-decode，draft_offload_v2 + transport=doca。
* draft 跑独立 TRT-LLM 进程（或 mock），加载 Qwen3-0.6B。
* 两进程通过 DOCA RDMA CPU_PROXY 来回传递 prompt / draft tokens。
* 输出 token 序列必须 **byte-identical** 于
  `LLM(target).generate(prompt)` 的 baseline。

## 二、起点

进入这一轮调试前：

* Phase 1（TCP）、Phase 2（ibverbs）都已 byte-match 通过。
* Phase 3 单进程 host-pinned buffer fix 已经 commit (`46aba346b`)。
* 已知 round 1 端到端成功，round 2 死在 target 等不到 draft 回包。
* 三轮 doc 写出来的 "izzy kernel bug" 推测被本轮证伪。

## 三、最终定位（先放结论）

经过最小复现脚本一轮一轮排除，trigger 收敛到：

> **在 target 进程内，一次 `ep.send(...)` 之后跑一次真实的
> `model.forward(...)` (Qwen3-4B)，紧接着的下一次 `ep.send(...)` 客户端侧
> 一切正常（SQ producer counter 升到 2、DBR 升到 `0x02000000`、
> `wqe_out=1`），但 NIC 不把这个 WQE 真的 emit 到 wire。**
> 用 `rdma statistic` 看 draft 一侧 `rx_write_requests` 永远不增加，
> draft 的 recv-signal 永远停在 round 1。

* 这跟 Phase 2 ibverbs 完全不一样 — 同样 NIC、同样 model.forward，Phase 2
  byte-match 通过。所以问题在 DOCA external-datapath QP，不在 NIC 硬件、
  也不在 izzy WQE submit kernel、也不在 TRT-LLM layer。
* phase14 doc 早就说过这台机器 MLNX_OFED 25.07 不足以支持完整
  GPU-initiated RDMA，需要 25.10。这次的现象跟那个限制是同一类。

## 四、现象时序

target 进程 single-process LLM 模式下的具体痕迹：

```text
# round 1
[doca send ENQ]   my_seq=1 imm=0x80000001
[doca send WORKER] my_seq=1 wqe_out=0
[progress snapshot] v0=1 last_seen=1 *DBR=0x01000000   ← SQ pi 升到 1
[doca recv PUSHED] cur=1 imm=0x80000000               ← target 收到 draft round-1 回包
                                                       ← layer._receive_responses 拿到 kOk

# model.forward (target verify next step) — Qwen3-4B inference

# round 2
[doca send ENQ]   my_seq=2 imm=0x80000001
[doca send WORKER] my_seq=2 wqe_out=1                 ← kernel 报告 SQ slot 1 被占用
[progress snapshot] v0=2 last_seen=2 *DBR=0x02000000   ← SQ pi 升到 2，doorbell record 也升了

# 永远停在这里：
[doca recv snapshot] cur=1 last_seen=1                 ← target 永远等不到 round-2 回包
RuntimeError: Timed out waiting for draft offload responses
```

draft 一侧：

```text
[draft][doca] round=1 req=2048 last_token=23690 → tokens=[1001, 1002, 1003, 1004]
[doca recv snapshot] cur=1 last_seen=1                 ← round 2 永远不到
[doca recv snapshot] cur=1 last_seen=1
...
```

`rdma statistic show link mlx5_0` 上 draft 那侧 `rx_write_requests`
每次实验只 +1，从来不 +2。

## 五、尝试矩阵

下面所有实验都在同一台 B200 + ConnectX-7、CUDA 13、DOCA 3.3、
MLNX_OFED 25.07 上跑。

### 5.1 izzy kernel 单纯 N 连发

用 `doca_rdma_izzy_work/test_multi_send.py` 直接调
`torch.ops.doca_rdma.write_with_imm` N 次：

| 场景 | 结果 |
|---|---|
| N=4，无中间活动 | server 收齐 4 个 ✓ |
| N=4，1 s fp16 matmul（256 MB tensors）在每两次 send 之间 | 3/4，丢最后一发 ✗ |

→ 推翻第二轮 doc 里"izzy kernel 不支持多次 submit"的推断。
kernel 本身 work。问题在更上层。

### 5.2 用我们的 `_DocaEndpointBackend` 完整代码路径

`examples/llm-api/rdma/test_endpoint_clone.py` 跑：

| 场景 | 结果 |
|---|---|
| N=4，无中间活动 | 全到 ✓ |
| N=4，256 MB tensor + fp16 matmul 在 send 之间 | 全到 ✓ |
| N=4，6 GB tensor + fp16 matmul 在 send 之间 | 全到 ✓ |

→ 我们的 endpoint 代码本身也 work。普通"GPU 内存压力 + 计算"
**不是 trigger**。

### 5.3 单进程 + 加载 Qwen3-4B（不 forward）

`LOAD_LLM=1`，加载 LLM 但永远不 `llm.generate`：

| 场景 | 结果 |
|---|---|
| N=2，加载 LLM、不 forward、不 warmup | 都到 ✓ |
| N=2，加载 LLM、做一次 warmup `llm.generate` 在 endpoint init 之前 | 都到 ✓ |

→ "加载 LLM" 本身**不是 trigger**。Endpoint 是在 LLM 之后 init 都可以。

### 5.4 单进程 + `llm.generate` 在 send 之间（**这就是 trigger**）

| 场景 | 结果 |
|---|---|
| N=2，`ep.send → llm.generate → ep.send`（无 warmup） | round 2 丢 ✗ |
| N=2，warmup `llm.generate` + endpoint init + `ep.send → llm.generate → ep.send` | round 2 丢 ✗ |

→ "**在两个 send 之间跑 model.forward**"是触发条件，跟"endpoint init
何时发生"、"warmup 与否"无关。

### 5.5 各种修复 / 缓解尝试

针对 5.4 这个失败场景做过的所有尝试：

| 尝试 | 修改 | 结果 |
|---|---|---|
| Wait send CQ (GPU) | `torch.ops.doca_rdma.wait_send` 在 write_with_imm 之后 | GPU kernel 10 M poll 超时（CPU_PROXY mode 下 send CQE 不写）；round 1 之后 wire 不出去 ✗ |
| Wait send CQ (CPU) | `poll_send_cq_cpu` spin 等 `wqe_counter == ours` | send CQ 全是 `op_own=0xF1` (CQE_INVALID 占位) — CPU_PROXY mode 下 send CQE 不真写。spin 5 s 超时 ✗ |
| sleep 1 ms 在 send 之后 | 直接 `time.sleep(0.001)` | 无改善 ✗ |
| 内联 `cpu_proxy_progress` × 64 | 在 send 之后立即喷 64 次 progress | 无改善 ✗ |
| 专用 `torch.cuda.Stream` | endpoint 用自己的 stream，避免跟 TRT-LLM stream 交叉 | 单独使用无改善；但是 + 下面的 sender thread 之后**起决定性作用** |
| 关 torch caching allocator | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | 无改善 ✗ |
| 关 MPI worker spawn | `TLLM_WORKER_USE_SINGLE_PROCESS=1` | 让 spec_dec_target_main 单进程跑（不 spawn MPI worker），失败模式不变 ✗ |
| 用 mock draft（不跑 torch） | `--backend mock` | 不变；trigger 在 target 一侧 ✗ |
| `CUDA_LAUNCH_BLOCKING=1` | 强制 kernel 同步 launch | 无改善 ✗ |
| 关 KV cache reuse + overlap scheduler | `enable_block_reuse=False`, `disable_overlap_scheduler=True` | 无改善 ✗ |
| ctypes.memmove 直写 host-pinned buffer | 绕过 `Tensor.copy_` + InferenceMode | 单独无改善；但配合 sender thread 之后**起决定性作用** |
| 独立 sender thread + 专用 stream + InferenceMode + ctypes | 见下文 5.6 | **client side 完全修好**；wire 仍丢 ✗ |
| Pre-warm `llm.generate` 在 endpoint init 之前 | 没 generate-between 时全到；有 generate-between 时还丢 | 不是 fix ✗ |
| `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` 在每次 generate 后 | 释放 cuda pool 试图 reset | 无改善 ✗ |

### 5.6 sender thread + 专用 stream + ctypes（部分修复）

`_DocaEndpointBackend.send` 改成：

1. 主线程 `send()` 把 imm + payload + seq 推入内部 queue，然后挂住等
   worker 完成；
2. dedicated daemon `_send_worker_loop` 在自己的 `torch.cuda.Stream`
   上跑；
3. worker 用 `torch.inference_mode()` 包住（这样能 inplace 改在
   InferenceMode 下创建的 `_send_payload` view）；
4. WQE staging buffer 通过 `ctypes.memmove` 直接写到 host-pinned 映射
   的 device pointer（避免 `Tensor.copy_` 路径跟 InferenceMode /
   TRT-LLM stream scheduler 打架）。

**修复效果**：
* 之前 round 2 SQ producer counter 永远停在 1；现在每次 `send` 都正确
  把 producer counter 推到下一个值，doorbell record 也跟着升
  (`0x01000000 → 0x02000000 → 0x03000000 → ...`)。
* `test_endpoint_clone.py` 在 6 GB GPU 内存压力 + heavy fp16 matmul
  之下 N=4 全部到 server。

**没修复的部分**：
* 一旦 send 之间塞进真实的 `llm.generate(Qwen3-4B forward)`，下一次
  send 的 wire 仍然没了。Client 侧所有指标（producer counter、DBR、
  wqe_out）都正常，只是 NIC 不把 WQE emit 出去。

## 六、为什么我们停在这里

`llm.generate(...)` 之后 DOCA external-datapath QP 失去 "下一次 SQ submit
就 emit 一个新包" 的能力，但同样 NIC 上 Phase 2 ibverbs QP 完全正常。
区别只在 QP 是不是 DOCA 的 "external-datapath GPU-exported" QP。

我们能从 userspace 看到的全部状态都对：

* SQ 内的 WQE bytes 写进去了
* SQ producer counter (`qp_cpu[0]`) 升了
* DBR (doorbell record) 升了
* CPU proxy thread 在 tight loop 上跑

但 NIC firmware / DOCA SDK 内部对 external-datapath QP 在 CUDA-heavy
forward 之后处理新 SQ submit 的逻辑，对我们是黑盒。izzy / DOCA 3.3 /
MLNX_OFED 25.07 这套组合上我们用尽了 userspace 能拿到的杠杆，再往下
就是 SDK / firmware 源码层面的事，不在我们能改的范围内。

`phase14_doca_gpu_sm_db.md` 早期记的 "需要 MLNX_OFED ≥ 25.10 才能
完整支持 GPU-initiated RDMA" 跟现在这个症状是一致的 — Phase 3
CPU_PROXY 在 25.07 上对单一 round-trip 是 work 的（host-pinned buffer
fix 之后 round 1 完整通过），但跨 CUDA-heavy model.forward 的连续
round-trip 不可达。

## 七、本轮已 commit 的产出

| commit | 内容 |
|---|---|
| `4aec00813` | `_DocaEndpointBackend` sender thread + 专用 stream + ctypes + InferenceMode fix |
| `2d03105d8` | `test_endpoint_clone.py` 最小复现脚本 + phase13 doc 第 4 轮 addendum |
| `af721b50e` | （上一轮）phase13 doc 第 3 轮 addendum |
| `dfa6c9975` | （上一轮）phase13 doc 第 2 轮 addendum |
| `46aba346b` | （上一轮）host-pinned DOCA buffer fix |

## 八、复现步骤（一行一行可跑）

```bash
# 1. server (idle receiver, GPU 0 / mlx5_0)
docker exec rdma_doca bash -c '
  cd /home/scratch.zhaoyangw_gpu/rdma/TensorRT-LLM/examples/llm-api/rdma && \
  GLUE_DOCA_HOST_PINNED_BUFFERS=1 GLUE_RDMA_NIC_HANDLERS=1 \
  GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 GPU_ID=0 NIC=mlx5_0 PORT=47600 \
  N=2 python3 -u test_endpoint_clone.py server
'

# 2. client (GPU 1 / mlx5_1) with Qwen3-4B + generate between sends
docker exec rdma_doca bash -c '
  cd /home/scratch.zhaoyangw_gpu/rdma/TensorRT-LLM/examples/llm-api/rdma && \
  GLUE_DOCA_HOST_PINNED_BUFFERS=1 GLUE_RDMA_NIC_HANDLERS=1 \
  GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 TLLM_WORKER_USE_SINGLE_PROCESS=1 \
  CUDA_VISIBLE_DEVICES=1 GPU_ID=0 NIC=mlx5_1 HOST=127.0.0.1 PORT=47600 \
  LOAD_LLM=1 GENERATE_BETWEEN=1 N=2 python3 -u test_endpoint_clone.py client
'
```

预期：client 报告 round 1/round 2 全 sent，每次 generate 大约 18-20 s；
server 拿到 round 1 后 30 s 超时 — `[server] TIMEOUT waiting round 2`。

把 `GENERATE_BETWEEN=0 HEAVY=1` 换上来，N=4 + 6 GB matmul 之间，server
能拿齐 4 round —— 证明 endpoint 代码路径本身是好的。

## 九、Phase 3 e2e byte-match 现状

| 子目标 | 当前状态 |
|---|---|
| 单轮 (round 1) DOCA CPU_PROXY 端到端 | ✓ byte-correct |
| Endpoint `send` 在 TRT-LLM 进程内 SQ producer 正确推进 | ✓ 已修复 (本轮 commit `4aec00813`) |
| 多轮 spec-decode e2e byte-match against baseline | ✗ blocked on DOCA SDK / MLNX_OFED 25.07 限制 |

## 十、想继续往下推的话有哪些选项

按"我们能改"/"我们不能改"分两层：

**a) 我们能改的（都已经验证不解决根本问题，但记录在此）**

* 已尝试且无效（详见 5.5）。

**b) 需要环境/SDK 升级的**

1. **升级 MLNX_OFED 到 25.10**（容器宿主侧改，需要 reboot 或者至少 mlx5
   模块 reload）。这是 phase 14 doc 一直在 block 的 dependency。25.10
   之后理论上 Phase 4 GPU_SM_DB 直接可用，跳过 CPU_PROXY external
   datapath QP 这条路径。
2. **DOCA SDK 内部 fix**：如果 NVIDIA 那边能确认是 external-datapath
   QP 在 cuMem*-触发后某个内部 invalidation，让我们能加显式
   "post-forward QP refresh" 调用，那是另一条路。需要 SDK 团队
   review。
3. **走 Phase 2 ibverbs 来满足 byte-match 验收**：Phase 2 ibverbs e2e
   已经 byte-match 通过；如果 spec-decode 实际 deploy 只关心
   "byte-correct + 接近 RDMA 性能"，ibverbs 路径直接可用，等 25.10 / Phase
   4 GPU_SM_DB 再回头切。
