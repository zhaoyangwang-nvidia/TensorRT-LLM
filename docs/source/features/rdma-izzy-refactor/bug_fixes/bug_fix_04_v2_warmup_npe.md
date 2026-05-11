# Bug Fix #04 — v2 path 在 warmup 阶段 NPE

## 触发条件

`draft_offload_v2=True`（`TLLM_DRAFT_RDMA_OFFLOAD_V2=1`）+
`is_warmup=True`（TRT-LLM 启动时的预热前向）。

## 错误信息

```
AttributeError: 'NoneType' object has no attribute 'request'
File ".../draft_target.py", line 289, in forward
    self._rdma_draft_client.request(...)
```

## 根因

`DraftTargetOneModelWorker.forward` 进 warmup 分支直接调用 v1 client：

```python
if self._rdma_offload_enabled:
    if bool(is_warmup):
        # Warmup path...
        self._rdma_draft_client.request(...)  # <-- v1 client only
```

但 v2 path 下 `_rdma_draft_client = None`（v2 用的是 `_rdma_v2_offload_layer`），
所以 attribute access 直接 NPE。

## 修复

`tensorrt_llm/_torch/speculative/draft_target.py`：

```python
if self._rdma_offload_enabled:
    if bool(is_warmup):
        # v1 does a real round-trip with empty output_history;
        # v2 only needs ``IbverbsDraftOffloadLayer`` to start its
        # channel (lazy done by the first forward), so we just zero-
        # init the tensor without hitting the v2 layer.
        if self._rdma_draft_client is not None:
            self._rdma_draft_client.request(
                tokens=[], position=0,
                max_draft_len=self.max_draft_len,
                device=logits.device,
            )
        next_draft_tokens = torch.zeros(...)
```

只在 v1 path 下做真实 warmup round-trip。v2 layer 的连接是 lazy 的，
第一次真正 forward 时由 `_ensure_started()` 触发，不需要 warmup 单独
驱动一次。

## 验证

`TLLM_DRAFT_RDMA_OFFLOAD_V2=1` 跑端到端：warmup 不再炸，整个流程跑完，
输出语义正确：

```
GPUDirect RDMA enables direct data transfer between GPUs and other devices
like InfiniBand networks or NVMe SSDs, bypassing the CPU and system
```

## 影响

- pre-fix 严重程度：**FATAL** — v2 path 启动阶段 100% 挂
- 修复后：v2 path 首次启动 lazy 连接，无 warmup round-trip，可能在第一
  次真正 decode step 多一次 connection setup 延迟。生产场景如果敏感，
  应该在 `__init__` 之后显式调用 `layer._ensure_started()`。当前未做。

## 教训

新增 v2 path 时只关注新增 layer 的代码，没审计现有 forward 路径里所有
`self._rdma_draft_client.X` 调用点。下次新增 alternative path 的时候，
最简单的防御是 grep 所有 `_rdma_draft_client.` 调用，给每一处加 None
guard。
