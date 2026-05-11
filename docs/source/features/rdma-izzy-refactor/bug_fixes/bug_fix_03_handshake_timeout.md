# Bug Fix #03 — RDMA handshake timeout 10s 太短

## 触发条件

`draft_rdma_server_v2.py` 作为 server 端等 RDMA TCP 握手时，如果 client（TRT-LLM
target）还在加载模型权重（~40s+），server 在 10s 后超时退出。

## 根因

`IbverbsEndpointConfig.handshake_timeout_s` 默认 `10.0`。
`_IbverbsRdmaBackend._tcp_handshake` 在 server 模式下：

```python
srv.settimeout(self._cfg.handshake_timeout_s)
client, _ = srv.accept()    # 10s 后超时 raise
```

测试场景：

| 进程 | 启动时间 |
|---|---|
| draft v2 server load model | ~1.2s（TRT-LLM block reuse 缓存命中后）|
| draft v2 server channel.start → accept() | t=1.2s |
| TRT-LLM target load model | ~40s |
| TRT-LLM target connect RDMA TCP | t=40s+ |

差距 30s+，server 端 10s 之后就 timeout → server 死 → target 后面才连接，
connection refused → 测试失败。

## 修复

`tensorrt_llm/_torch/speculative/ibverbs_endpoint.py`：

```python
# Default 120s — long enough for the peer process to finish heavy startup
# (e.g. TRT-LLM target loading an 8B model takes ~40s).  The client side
# typically connects within a few seconds; the server side may wait longer.
handshake_timeout_s: float = 120.0
```

10 → 120。任一端都是 120s 上限，已经足够覆盖 TRT-LLM 8B + 9B + 30B 规模的模型加载时间，
同时也不会"永远卡住"等不来的 peer。

## 验证

把 TRT-LLM target 启动时间 (~40s) 跟 draft server 启动时间 (~5s) 串起来跑通：
v2 server 等到 target 连接、QP 建立、输出正确：

```
GPUDirect RDMA enables direct data transfer between GPUs and other devices
like InfiniBand networks or NVMe SSDs, bypassing the CPU and system
```

字节级匹配 step-3 baseline。

## 副作用

- 极端情况下 server 会等 2 分钟才放弃。生产场景调用方应该自己有外部超时控制。
- 不影响 client 路径（client 一般几秒内连上 server）。

## 教训

冒烟测试和真 e2e 的时间尺度差异很大。冒烟用同一 process 启 server + client
（毫秒级），把 default timeout 设得太短没暴露问题。真 e2e 跨进程 + 真模型才
看到 handshake 时间窗的关系。
