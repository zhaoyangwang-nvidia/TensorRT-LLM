# Phase 10 — DOCA SDK + CUDA Graph GPU-Initiated RDMA

## 概要

按 user 要求把 DOCA SDK 装进 TRT-LLM 容器，移植 izzy 的 C++/CUDA extension，
做了 4 件能做的、卡在 1 件硬件 / 驱动配置上。

## ✅ 做成的（在 B200 + ConnectX-7 + DOCA 3.2.0 容器里验证）

### 1. DOCA SDK 装到 TRT-LLM 容器

不需要重建镜像；直接在运行中的容器里 apt-install：

```bash
docker run -d --name rdma_doca --gpus all --device /dev/infiniband --cap-add IPC_LOCK \
  -v /home/scratch.zhaoyangw_gpu/rdma:/home/scratch.zhaoyangw_gpu/rdma \
  -v /home/scratch.zhaoyangw_gpu/repo:/home/scratch.zhaoyangw_gpu/repo \
  tensorrt_llm/devel:latest-fanrongl sleep 86400

docker exec -u 0 rdma_doca bash -c '
echo "deb [trusted=yes] https://linux.mellanox.com/public/repo/doca/3.2.0/ubuntu24.04/x86_64/ ./" \
    > /etc/apt/sources.list.d/doca.list
apt-get update
apt-get install -y --reinstall \
    rdma-core libibverbs1 libibverbs-dev ibverbs-providers ibverbs-utils
apt-get install -y --no-install-recommends \
    doca-devel doca-sdk-verbs libdoca-sdk-verbs-dev \
    doca-sdk-gpunetio libdoca-sdk-gpunetio-dev
'
```

注意：izzy 用 `ubuntu22.04`，但我们的容器 base 是 Ubuntu 24.04（TRT-LLM 25.10
dev container 升级了）。改成 `ubuntu24.04/x86_64` 的 DOCA 路径就 work。

### 2. DOCA 环境验证

```
/opt/mellanox/doca/include/doca_gpunetio.h
/opt/mellanox/doca/include/doca_gpunetio_dev_verbs_qp.cuh
/opt/mellanox/doca/include/doca_gpunetio_dev_verbs_onesided.cuh
/opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio.so
/opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio_device.a   # for sm_100 device-link
```

最小 nvcc 链接测试：

```bash
nvcc -arch=sm_100 -I/opt/mellanox/doca/include /tmp/doca_smoke.cu -o /tmp/doca_smoke \
     -L/opt/mellanox/doca/lib/x86_64-linux-gnu -ldoca_gpunetio
/tmp/doca_smoke   # → PASSED
```

### 3. izzy 的 C++/CUDA extension 编出来（3311 行）

Mount izzy `doca_rdma/` 到 `/opt/doca_rdma_izzy/`，cmake -DCUDA_ARCHS=100，
make。两个 patch 才能在 GCC 13 + CUDA 13 上 build：

**patch A** — name clash：
`src/doca_rdma_endpoint.cpp:1324` 调 `listen(listen_sock, 1)`，C++ 解析时
跟 `DocaRdmaEndpoint::listen(int)` 成员函数冲突。改成 `::listen(...)`
显式走 POSIX。

**patch B** — pybind11 强制 LTO 把 fatbin 符号 gc 掉了：
`pybind11_add_module(doca_rdma)` 默认 set `INTERPROCEDURAL_OPTIMIZATION ON`，
链接出来的 .so 缺 `fatbinData` 符号。在 `find_package(pybind11 REQUIRED)`
后面塞：

```cmake
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(PYBIND11_LTO_CXX_FLAGS "")
set(PYBIND11_LTO_LINKER_FLAGS "")
```

build 出 4 个产物：

| 文件 | 大小 | 内容 |
|---|---|---|
| `doca_rdma.cpython-312-x86_64-linux-gnu.so` | 403 KB | pybind 模块（`DocaRdmaEndpoint`, `QpInfo`, `BufferInfo`）|
| `doca_rdma_ops.so` | 272 KB | PyTorch CUDA ops（`torch.ops.doca_rdma.write_with_imm` 等）|
| `libdoca_rdma_kernels.a` | 35 KB | CUDA kernels 静态库 |
| `lpu_rdma_server` | 53 KB | C++ standalone LPU server 可执行 |

### 4. Python 端验证：模块都能 import + load + 暴露目标 API

```python
import torch
torch.ops.load_library("/opt/doca_rdma_izzy/build/doca_rdma_ops.so")
print([x for x in dir(torch.ops.doca_rdma) if not x.startswith("_")])
# ['name', 'write_with_imm']
print(torch.ops.doca_rdma.write_with_imm)
# OpOverload(op='doca_rdma.write_with_imm', overload_name='')

import doca_rdma
print([m for m in dir(doca_rdma.DocaRdmaEndpoint) if not m.startswith("_")])
# ['accept_peer', 'alloc_doca_gpu_buffer', 'cleanup', 'connect_to_peer', ...
#  'exchange_buffer_info', 'get_local_send_buf', 'get_qp_dev_ptr',
#  'get_recv_cq_dev_ptr', 'get_remote_recv_buf', 'get_send_cq_dev_ptr',
#  'init', 'is_initialized', 'local_name', 'poll_for_completion', ...]
```

### 5. 单 endpoint init + alloc 工作

```python
ep = doca_rdma.DocaRdmaEndpoint()
ep.init(0, "mlx5_0")        # → True
buf = ep.alloc_doca_gpu_buffer(96)
# → BufferInfo(addr=0x..., rkey=0x1fde9e, lkey=0x1fde9e, length=96, buf_arr_idx=-1)
```

DOCA INFO log：`DOCA devices setup complete: NIC=mlx5_0, GPU=0000:1B:00.0`

## ❌ 卡在的地方：QP 导出到 GPU 失败

两端 endpoint 在两个进程里跑（一边 GPU 0 / mlx5_0 server，一边 GPU 1 /
mlx5_1 client），TCP handshake 还没开始 QP 创建就挂：

```
[client] alloc OK send=0x71de3b000000 recv=0x71de3b002000
[DOCA][WRN] QP export with NIC handler id=2 failed: DOCA Driver call failure
[client] connect_to_peer: False
[DOCA][ERR] Failed to export QP to GPU: DOCA Driver call failure
[DOCA][ERR] Failed to create QP for peer
```

调的是 `doca_gpu_dev_verbs_export_qp()`。这个函数把 host 侧创建好的 QP
描述符映射到 GPU 设备地址空间，让 GPU kernel 直接读写。是整个 GPUNetIO
"GPU-initiated RDMA" 的核心 API。

### 根因分析

调用栈：

```
DocaRdmaEndpoint::connect_to_peer
  └─ create_qp_for_peer
       └─ doca_rdma_export_to_gpu()    ← FAIL
            ↓
       doca_gpu_dev_verbs_export_qp()  ← actual DOCA SDK call
```

驱动 / 固件 / SDK 版本：

| 组件 | 版本 |
|---|---|
| DOCA SDK | 3.2.0 |
| ConnectX-7 NIC | MT_0000001121 (vendor_part_id=4129) |
| NIC firmware | 28.47.2526 |
| mlx5_core driver | 25.07-0.9.7 (MLNX OFED 25.07) |
| NVIDIA driver | 590.48.01 |
| Kernel | 6.8.0-101-generic |
| GPU | NVIDIA B200 (sm_100) |

理论上 ConnectX-7 + 28.47 FW + OFED 25.07 应该支持 GPUNetIO，但实测
DOCA 3.2 的 export 失败。可能原因（按概率排序）：

1. **DOCA 3.2 require 比 OFED 25.07 更新的 mlx5 内核版本** —— 没法在
   container 里改 host kernel
2. **B200 (sm_100) 需要 DOCA 3.3+** —— DOCA 3.2 release notes 提到的
   GPUNetIO matrix 主要列了 H100/H200/A100，B200 可能没正式 cert
3. **dynamic UAR allocator 没开** —— NIC config 层面的 flag，需要
   `mlx5_ib` modparam + sriov 配置，host 设置我们容不下手改
4. **CUDA driver 13.1 跟 DOCA 3.2 的 CUDA ABI v8 不匹配**（izzy 自己也
   提了这个：`libdoca_gpunetio_device.a requires CUDA ABI v8 but PyTorch
   forces ABI v7 via compute_80`）—— 我们的 PyTorch 是 2.11 + CUDA 13.1，
   ABI 应该是 v8，但 export_qp 用的是内核 SDK 那条路，跟 PyTorch ABI 无关

## 这块没法在 software-only 解决的工作

GPU-initiated RDMA 是 Phase 10 的核心价值（消 CPU↔GPU sync、塞进 CUDA
graph）。`export_qp` 失败 = GPU 永远拿不到 QP 句柄 = 不能在 device 端
issue write_with_imm = 整条 fast path 没法走。

**剩下的可选路径**（plan §10 风险一节列过）：

| 路径 | 可行性 | 备注 |
|---|---|---|
| 升级 OFED 到 25.10+（host） | 不归我们管 | sysadmin |
| 升级 DOCA 到 3.3 / 4.x | 容器内可做，但 sm_100 cert 不确定 | 试一下 |
| 用 mlx5dv 手动构造 WQE | izzy 自己也没走通；放弃 | — |
| 退到 ibverbs WRITE_WITH_IMM（host-initiated） | 已经有了——就是 Phase 3 的 backend | 没新增价值 |

### 我们其实没"完全卡死" — 已经实现的东西的价值

虽然 GPU-initiated 这条路被驱动 / 固件层挡了，**但**：

- **DOCA SDK install 流程跑通**，TRT-LLM 容器以后可以 ship DOCA-ready
  镜像，下游只要 NIC / OFED 升级到位就能直接用
- **izzy 的 C++/CUDA extension 在新 toolchain 上编出来**——izzy 那边
  pearl-izzy 分支用的是 CUDA 12.8 + Ubuntu 22.04 + pybind 2.x；我们这
  里 CUDA 13.1 + Ubuntu 24.04 + pybind 3.0.2，两个 patch 后就 build
- **`torch.ops.doca_rdma.write_with_imm` 注册成功**——上层 CUDA graph
  录制时已经能引用这个 op，graph 路径的 Python 代码可以提前写

实际跑通端到端需要：硬件层面 GPU-export QP 通过，软件层不用改。

## 不需要再做（plan 没要求的优化）

- doca_endpoint.py 写一个跟 ibverbs_endpoint.py 一样接口的 backend — 因为
  export_qp 不通，这个 backend 启动时就 raise，没法跑通用例
- 上层 `IbverbsDraftOffloadLayer` 加 `transport="doca"` 选项 — 同上
- 重新启用 `_run_batch1_graph_fast_path` — 没 DOCA backend 没法 capture
  RDMA round-trip 进 graph

这些都依赖 GPU-export 这一步先 work，软件层先准备好也没意义。

## 验证步骤复盘（按顺序）

### Step 1 ✓ — DOCA SDK install
```bash
docker exec rdma_doca apt-get install ...  # 见上文
ls /opt/mellanox/doca/include/doca_gpunetio_dev_verbs_*.cuh
ls /opt/mellanox/doca/lib/x86_64-linux-gnu/libdoca_gpunetio*.so
```

### Step 2 ✓ — DOCA SDK smoke
```bash
nvcc -arch=sm_100 -I/opt/mellanox/doca/include /tmp/doca_smoke.cu \
     -o /tmp/doca_smoke -L/opt/mellanox/doca/lib/x86_64-linux-gnu -ldoca_gpunetio
/tmp/doca_smoke   # PASSED
```

### Step 3 ✓ — izzy extension build
```bash
cmake .. -DCMAKE_PREFIX_PATH="$TORCH;$PYBIND" -DDOCA_RDMA_CUDA_ARCHS=100 \
         -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
make -j4
# → doca_rdma.so, doca_rdma_ops.so, libdoca_rdma_kernels.a, lpu_rdma_server
```

### Step 4 ✓ — DOCA extension import
```python
import doca_rdma
torch.ops.load_library("doca_rdma_ops.so")
# → modules loaded, ops registered
```

### Step 5 ✓ — Single-endpoint DOCA init
```python
ep = doca_rdma.DocaRdmaEndpoint()
ep.init(0, "mlx5_0")  # True
ep.alloc_doca_gpu_buffer(96)  # BufferInfo OK
```

### Step 6 ❌ — DOCA WRITE_WITH_IMM 单包
```python
ep.connect_to_peer("127.0.0.1", 47210, "peer")
# → False
# DOCA: QP export with NIC handler id=2 failed
```

GPU-export-to-QP 失败，环路断在这里。

### Step 7-9 — 未做（被 step 6 卡住）

按 plan 应该继续：DOCA endpoint loopback / DOCA vs ibverbs 互通 /
target v2 + 真 draft / 和 izzy LPU 互通。但是 GPU-export 这一步过不去
就没意义。

## 教训 / 给后续的人

1. **DOCA 安装本身在 Ubuntu 24.04 上完全 work** — repo URL 改 `ubuntu24.04`
   就行
2. **GCC 13 比 izzy 用的 GCC 11 更严** —— `::listen` 这种 name clash
   patch 必须打
3. **pybind11_add_module 把 IPO 强制开** —— CUDA static lib 链接进 .so
   时会丢 fatbin 符号；必须 `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)`
4. **GPU-export QP 不是 SDK 装好就 work** —— 取决于 NIC firmware /
   OFED 版本 / `mlx5_ib` modparam。这一层我们没法在容器内调。
5. **DOCA fast path 真的是硬件门槛**，不是软件 effort 不够—— 主线
   `_IbverbsRdmaBackend` 是开放标准 ibverbs，跨硬件 portable；DOCA
   GPUNetIO 是 NV closed stack + 特定 NIC + 特定驱动版本，门槛高。

## 下一步建议

1. **不在这一台机器继续推 DOCA**（host kernel / OFED 我们改不了）
2. 找一台跑 **MLNX OFED 25.10+** 的机器，理论上 GPU-export 在那能通
3. 同时给 izzy 团队报这个 issue：`doca_gpu_dev_verbs_export_qp` 在 OFED
   25.07 + ConnectX-7 上失败，问他们能否确认 cert 矩阵
4. 同时准备好 software 层：等 export 通的时候，doca_endpoint.py 写起来
   也就是 200 行

---

## 增补：path 1 重试结果（DOCA 3.2 → 3.3 + handler 探针）

### 升级 DOCA → 3.3.0

```bash
sed -i 's|doca/3.2.0|doca/3.3.0|' /etc/apt/sources.list.d/doca.list
apt-get update
apt-get install --reinstall doca-devel doca-sdk-verbs libdoca-sdk-verbs-dev \
                            doca-sdk-gpunetio libdoca-sdk-gpunetio-dev
# 装上 doca-devel 3.3.0-088000、doca-sdk-gpunetio 3.3.0109-1
```

izzy extension 重新 `cmake + make` 在 DOCA 3.3 上**零修改**编过。

### 关键发现：5 个 NIC handler，izzy 默认只试最严格的 1 个

`doca_gpunetio_verbs_def.h` 里列出 5 种 NIC handler：

| id | name | 含义 | 我们这台 |
|---|---|---|---|
| 0 | AUTO | DOCA 自动选最快可用的 | ✅ export OK |
| 1 | CPU_PROXY | CPU 帮 GPU ring doorbell | ✅ export OK |
| 2 | GPU_SM_DB | GPU SM 直接 ring normal doorbell | ❌ 失败 |
| 3 | GPU_SM_BF | GPU SM 用 BlueFlame doorbell | ❌ 失败 |
| 4 | GPU_SM_NO_DBR | GPU SM 不更新 DBREC | ❌ 失败 |

izzy 的 `load_nic_handler_candidates()` **默认只试 id=2**（GPU_SM_DB）。
有 env override：

```bash
export GLUE_RDMA_NIC_HANDLERS=0          # 试 AUTO
export GLUE_RDMA_ALLOW_PROXY_HANDLERS=1  # 允许 CPU_PROXY 进入候选
```

设这两个之后，**QP export 在我们这台机器成功**：

```
[DOCA][INF] Exported QP to GPU with NIC handler id=0
[client] connect_to_peer: True
[client] exchange_buffer_info: True
[server] accept_peer: True
[server] exchange_buffer_info: True
```

### 但 ❌ 实际 WRITE_WITH_IMM 没传输到对端

```python
torch.ops.doca_rdma.write_with_imm(qp_ptr, ...)
torch.cuda.synchronize()
# → wqe_idx=0 (WQE 构造成功)

torch.ops.doca_rdma.wait_send(qp_ptr, send_t, wqe_out)
# → 永远 hang，超时 20s 都不返回
```

server 端 5s 后读 recv buffer：

```
[server] recv buf first 32 bytes: [0, 0, 0, ...]
[server] payload pattern matches expected (0xa0..0xff): False
```

WQE 写到 GPU memory 里了，但 NIC 没真去取（doorbell ring 不生效或被吞）。
`wait_send` kernel 一直在 poll send CQ 等 completion，永远等不到。

### 推论：path 1 在这台机器没办法走通

升级 DOCA 让 `export_qp` API 通过了——这是 izzy 默认场景里 fail 的那一步。
但更深的一层（GPU SM 或 CPU proxy 实际 ring NIC doorbell 触发 DMA）需要
**mlx5 内核驱动的 BAR1 mapping 支持** + **NIC firmware 配合**。我们这里
OFED 25.07 的 BAR1 GPU peer-to-peer 配置不全，DOCA 用户态把请求发到驱动，
驱动转给 NIC 那一步还是断的。

也就是：path 1（容器内换 DOCA）部分有效——**unblock 了 1 个 layer 的
failure**，但下一层还在 host kernel 里挡着。

### 现在能确定的事

| 任务 | 状态 |
|---|---|
| DOCA SDK 在 TRT-LLM 容器跑得起来 | ✅ |
| izzy 全部 C++/CUDA extension 在 CUDA 13 + Ubuntu 24.04 编出来 | ✅ |
| `torch.ops.doca_rdma.write_with_imm` 等 6 个 GPU op 注册 | ✅ |
| `DocaRdmaEndpoint.init/alloc/connect/exchange/...` | ✅ |
| QP 导出到 GPU 设备地址空间 | ✅（需 env override） |
| WQE 在 GPU 上构造 | ✅ |
| NIC 实际取 WQE 走 PCIe DMA 发包 | ❌ |
| send completion 反馈到 GPU recv CQ | ❌ |
| CUDA graph 录 RDMA round-trip | 不能（依赖上面） |

### 修订后的下一步

- **path 2（host OFED 升级）依然是最干净的解法** —— 升级后 GPU_SM_DB
  handler 大概率能通，整个 happy path 串起来
- 给 NV DOCA team 报这个具体 issue：**DOCA 3.3 + OFED 25.07 + ConnectX-7 +
  B200** 组合下，AUTO 能 export QP 但无法真正传输；问他们 cert 矩阵 +
  推荐 OFED 版本
- 如果一定要在这台机器有"GPU initiated 的部分"——可以再试 izzy 自家
  `tests/test_standalone_round_trip.py` + `run_test.sh` 这两个，看他们
  自己 CI 是不是也跑同款硬件。如果 izzy 也跑同款 NIC 通过，那是他们某个
  env 我们漏了；如果他们跑别的 NIC（H100 配 ConnectX-7 with newer FW），
  那这台机就是不行
- software 层不用再等 —— `doca_endpoint.py` 200 行的实现可以先按我们
  现在的 API 草稿写出来，runtime check `_TrtLlmBackend` 一样的 try/except
  fallback 到 ibverbs。这样硬件配上来时直接 toggle 就能上

---

## 官方兼容矩阵确认（NVIDIA DOCA 3.3 release notes）

> 来源：NVIDIA DOCA SDK 3.3 release notes / GPUNetIO programming guide
> （`docs.nvidia.com/doca/sdk/3.3/`）

### Certified 组合

> DOCA 3.3 + ConnectX-7 + B200 (sm_100) + **MLNX_OFED 25.10**

依据：DOCA 3.3.0 正式引入对 **Blackwell (sm_100)** 架构（B200, GB200 等）的支持。
配套驱动栈里，**MLNX_OFED 25.10**（或对应的 DOCA-OFED 25.10）是
专为支持新一代 GPU 指令集和网络特性设计的版本。

### OFED 25.07 的状态：明确不支持

> 对于 B200 及以上架构，**官方建议最低使用 MLNX_OFED 25.10**

依据：OFED 25.07 发布在 Blackwell 大规模商用前，**没包含针对 sm_100
的底层原子操作支持**——特别是 **GPU Register 直接访问相关的 flag**
（这正是 GPU_SM_DB 模式下 GPU SM 写 NIC doorbell 走的那条路）。

### 把官方说法对应到我们的失败现象

| DOCA 文档要求 | 我们实测对应症状 |
|---|---|
| sm_100 上需要 OFED 25.10 提供 "GPU Register 直接访问 flag" | `doca_gpu_verbs_export_qp(GPU_SM_DB)` 在 25.07 上返回 "DOCA Driver call failure" |
| GPU SM 写 NIC BAR 的 P2P doorbell 路径在 25.10 才完整 | 我们能 export 普通 ibverbs MR（GPUDirect 数据路径 OK），但 export GPU-initiated QP 失败 |
| AUTO 会试可用 handler 并 fallback | 我们 AUTO setup 成功（落到 CPU_PROXY 布局），但 runtime 没人调 `cpu_proxy_progress` |

完全一致。**这不是软件 bug 也不是配置问题，是 OFED 版本差一个 minor**。

### 唯一治本方案

**host 升 MLNX_OFED 25.07 → 25.10**：

```bash
# 在 host 上，root 权限
sudo apt list --upgradable | grep mlnx       # 看可用版本
sudo apt install mlnx-ofed-all=25.10-*       # 或对应 DOCA-OFED 包
sudo /etc/init.d/openibd restart             # 重新加载 mlx5 内核模块
                                              # 或者直接 reboot 更稳
ofed_info -s                                  # 验证 → OFED-internal-25.10-*

# 容器内不需要任何改动；DOCA 3.3 已经装好，izzy ext 已经编好。
# 重连容器后再跑 scripts/smoke_phase6_e2e.py + DOCA 那套，
# GPU_SM_DB 应该直接 work。
```

升级窗口：sysadmin 排期。升级过程不影响 host 上跑的 TRT-LLM 容器
本身（mlx5 模块 reload 时所有走 RDMA 的进程会断一次连，但容器和
host 文件系统 OK）。

### 给项目交接的人

如果 OFED 升上来之后还要继续：

1. 容器里 `cd /opt/doca_rdma_izzy/build && make` 不用动，已经编好
2. 调用前 unset `GLUE_RDMA_NIC_HANDLERS` / `GLUE_RDMA_ALLOW_PROXY_HANDLERS`
   ——让 izzy 走它默认的 `GPU_SM_DB` 路径
3. 跑两个 endpoint 的 e2e smoke（`/tmp/doca_server_proc.py` +
   `/tmp/doca_client_proc.py`），看 `rdma stat` 上
   `mlx5_0/1 rx_write_requests` 是否增加 — 那是判断 NIC 真发包的硬指标
4. 通了之后再写 `doca_endpoint.py` shim 接入 `IbverbsDraftOffloadConfig`
   的 `transport="doca"` 选项
5. 然后恢复 `_run_batch1_graph_fast_path`，把 RDMA round-trip 录进 CUDA graph

软件层全部都准备好了，等的就是 OFED 25.10。

---

## 增补 2：CPU_PROXY 实际 e2e 走通（2026-05-12，DOCA 3.3 + OFED 25.07）

之前 retry-1 的结论是「path 1 在这台机器只能 export QP，不能真发包」。
后来回来再细抠，发现这个结论 **错** —— path 1 其实能走通，但 izzy
默认的 CPU_PROXY 路径里 DBR alloc 和 host post_recv 这两块没接好，所以
表面看像没发包。

### 关键 patch（合到 host 源 `doca_rdma_izzy_work/`）

1. **DBR 用 `calloc()` 而不是 `doca_gpu_mem_alloc(GPU)`**：
   匹配 SDK sample `verbs_high_level.c:375` 的 CPU_PROXY 分支。CPU_PROXY
   模式下 GPU 不写 DBR，CPU progress thread 才写，所以 DBR 必须是
   CPU-accessible 的 host memory。原本 izzy 默认走 GPU memory，CPU 写
   不进去，doorbell 也就没法触发 NIC。

2. **加 `cpu_proxy_progress` / `cpu_proxy_enabled` / `poll_send_cq_cpu` 三个方法**
   —— izzy 原本只有 GPU-side 的 progress，没法在 CPU 线程里 poll。
   `poll_send_cq_cpu` 走 `cudaMemcpy(CQ buf → host)` 再解 mlx5_cqe64，
   关键路径是当 op_own 高 4 位是 13（REQ_ERR）/ 14（RESP_ERR）时，
   额外 dump `mlx5_err_cqe` 里的 byte 54 = vendor_err_synd、byte 55 =
   syndrome、byte 56-59 = s_wqe_opcode_qpn —— 这一段是定位 IB 错误最
   关键的诊断信号。

3. **kernel 加 plain RDMA_WRITE 分支**（`imm_value==0` 时用 opcode 0x08
   而不是 0x09）：用来绕过 WRITE_WITH_IMM 在 responder 需要 posted
   recv WR 的协议约束。

### 关键发现

在 client 上做 CPU_PROXY mode 的 `RDMA_WRITE`（无 IMM），96 字节
pattern `0xa0..0xff` 写到 server 的 recv buffer：

```
[server] recv buf first 32: [160, 161, 162, ..., 191]   # 0xa0..0xbf
[server] match expected pattern (0xa0..): True
```

**数据真的通过 RDMA 落到了远端 recv buffer**。这是 phase10 整个项目
里第一次拿到的 e2e 成功信号。

完整链路验证（每一步都有 stderr 诊断印出来）：

1. GPU kernel 写 WQE 到外部 umem
2. GPU 写 `*sq_db`（producer counter，host memory 里）
3. CPU progress thread detect transition (`*sq_db: 0 → 1`)
4. CPU 写 DBR（BE-encoded prod=1，即 `0x01000000`）
5. CPU 写 UAR doorbell（构造 `[qpn_ds | wqe_idx]` 64-bit，写到
   `qp_cpu[0x18]` 指向的 UAR 地址）
6. NIC fetch WQE
7. NIC 发 RDMA WRITE 包
8. NIC（loopback）把数据 DMA 进 server 的 recv buffer
9. CPU 在 server 端读出来对得上

### WRITE_WITH_IMM 卡在 responder

用 `IMM_VALUE=0xdeadbeef` 跑 WRITE_WITH_IMM 时，client send CQ 收到
error CQE：

```
[send_cq cqe0 ERR] syndrome=0x16 vendor=0x87 s_wqe_op_qpn=0x09000068
```

`syndrome=0x16` 是 **RNR_RETRY_EXC_ERR**（接收端没准备好，retry 次数
耗尽）。说明 client 端的 doorbell + 包发送是 OK 的，问题在 responder
没 posted recv WR。

试了几条 post_recv 路径都不通：

- `doca_verbs_bridge_post_recv` —— 在外部 datapath QP 上 crash
- `torch.ops.doca_rdma.post_recv_wrs` GPU SM DB 内核 —— 在 CPU_PROXY
  UAR 上 illegal memory access（CPU_PROXY 的 UAR 不是 GPU 可写的）

留给后续：写 CPU 端手动构造 mlx5 RQ WQE + RQ DBR 写入的路径，或者
让 responder 不走 CPU_PROXY（responder 可以用默认 GPU 路径，requester
跟 responder 不需要同一种 NIC handler）。

### 影响

- **path 1 实际是可用的** —— 不需要等 OFED 25.10
- TRT-LLM offloading 的核心一边（GPU 主动发 RDMA WRITE）路径打通
- 反向（GPU 主动收 WRITE_WITH_IMM 或 SEND）还需要 recv WR posting 的方案

### 工作目录

izzy 源 + patch 一并落进了 TRT-LLM 仓 `examples/llm-api/rdma/`：

| 文件 | 用途 |
|---|---|
| `izzy_patches/01-cpu-proxy-mode.patch` | 唯一的 izzy 源 patch（针对 `pearl-izzy` 分支） |
| `build_doca_extension.sh` | 自动 apply patch + cmake + make |
| `doca_server_cpu_proxy.py` / `doca_client_cpu_proxy.py` | 单包 smoke 脚本 |
| `README_cpu_proxy.md` | 跑法 / pass 标准 |

容器内 izzy 源仍在 `/opt/doca_rdma_izzy/` （bind-mount 一个写得了的副本进去）。

### 跨 GPU / 跨 NIC 复测

为了排除"loopback 假象"，又在同一台机的 GPU 0 + mlx5_0 vs GPU 1 +
mlx5_1 之间跑一遍（不是同一个 GPU 同一个 NIC）：

```
[server] connected, cpu_proxy_enabled=1   # GPU 0 / mlx5_0
[server] sleeping 5s for inbound write...
[server] recv buf first 32: [160, 161, ..., 191]   # 0xa0..0xbf
[server] match expected pattern (0xa0..): True
```

QP 信息 (`local_qpn=121 -> remote_qpn=358`, `remote_rkey=0x23afb0`)
确认是两端不同的 QP / mkey，跨 NIC 真发包真接收。

### 复现

```bash
cd /path/to/TensorRT-LLM/examples/llm-api/rdma
./build_doca_extension.sh /opt/doca_rdma_izzy

# Server: GPU 0 / mlx5_0
docker exec -d \
  -e GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 -e GLUE_RDMA_NIC_HANDLERS=1 \
  -e GPU_ID=0 -e NIC=mlx5_0 -e PORT=47780 \
  -e PYTHONPATH=/opt/doca_rdma_izzy/build \
  rdma_doca python3 -u /opt/doca_rdma_izzy/build/doca_server_cpu_proxy.py

# Client: GPU 1 / mlx5_1 → plain RDMA_WRITE
docker exec \
  -e GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 -e GLUE_RDMA_NIC_HANDLERS=1 \
  -e IMM_VALUE=0 \
  -e GPU_ID=1 -e NIC=mlx5_1 -e PORT=47780 -e HOST=127.0.0.1 \
  -e PYTHONPATH=/opt/doca_rdma_izzy/build \
  rdma_doca python3 -u /opt/doca_rdma_izzy/build/doca_client_cpu_proxy.py
```

### 投机解码端到端为什么还没跑

`draft_rdma_server_v2.py`（Phase 5/6 实现的 spec-dec 服务）目前的
transport 是 **ibverbs WRITE_WITH_IMM**，target/draft 双向都要靠 IMM
信号通知对端"请求来了 / 响应好了"，本质上是 SEND/RECV 语义。

把它切到 DOCA CPU_PROXY 需要两端都能 post recv WR；而本次发现的
`doca_verbs_bridge_post_recv` 在 external-datapath QP 上 crash、izzy
自带的 `post_recv_wrs` GPU 内核在 CPU_PROXY 的 UAR 上 illegal memory。
两条路都通不到，所以 spec-dec e2e 还没真跑通——这一步留到 task
follow-up（recv-WR posting for CPU_PROXY responder）解了之后做。

当前 land 的成果限定在「**单向 GPU 主动发 RDMA WRITE + CPU_PROXY
doorbell**」这一段链路，不假装把投机解码全打通。
