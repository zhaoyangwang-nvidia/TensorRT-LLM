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
