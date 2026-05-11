# Bug Fix #05 — position_ids shape mismatch between TRT-LLM and v2 layer

## 触发条件

`draft_offload_v2=True` + TRT-LLM target 第一次进入 generation phase。

## 错误信息

```
ValueError: position_ids must provide exactly one start position per
request (got 12, expected 1)

raised by IbverbsDraftOffloadLayer._resolve_request_start_positions
```

## 根因

两边对 `position_ids` 张量形状的约定不一样：

| 来源 | shape | 含义 |
|---|---|---|
| TRT-LLM `forward` 传给 worker | `(total_tokens,)` | 每个 token 的位置（context phase = 12 prompt tokens 的 [0,1,...,11]）|
| `IbverbsDraftOffloadLayer.forward` 期望 | `(batch_size,)` | 每个 request 的 **start** position |

v1 path `_rdma_offload_draft_tokens` 自己有处理 (line 460 取
`position_ids.reshape(-1)[-1]`)，v2 path 没做这个转换 → shape mismatch
raise。

## 修复

`tensorrt_llm/_torch/speculative/draft_target.py` `_rdma_offload_draft_tokens` v2
分支：

```python
# IbverbsDraftOffloadLayer.forward expects exactly one start-position per
# request (length == batch_size), but TRT-LLM passes the *full* per-token
# position_ids (length == total prompt+gen tokens in batch).  Build a
# per-request position vector — last position is what the draft side
# needs to know where to resume from.  Only batch_size=1 is exercised
# today (Phase 4 limitation); the multi-request path lands in Phase 9.
if position_ids is None or int(position_ids.numel()) == 0:
    per_request_positions = torch.zeros((int(batch_size),), dtype=torch.int64, device=logits.device)
elif int(position_ids.numel()) == int(batch_size):
    per_request_positions = position_ids.reshape(-1)
else:
    last_pos = int(position_ids.reshape(-1)[-1].detach().cpu().item())
    per_request_positions = torch.tensor([last_pos] * int(batch_size), dtype=torch.int64, device=logits.device)
```

逻辑：
- 如果 numel == 0 → 全 0
- 如果 numel == batch_size → 已经是 per-request 形状，直接 reshape
- 否则取 flat 的最后一个值 broadcast 到 batch_size（仅 batch_size=1 时正确）

注意：`batch_size > 1` 时这个简化是错的——多 request 时应该按每个
request 的 start position 切分。**Phase 9 多 request 时必须重写**。

## 验证

v2 e2e 端到端跑通，output 与 step-3 baseline 一致：

```
GPUDirect RDMA enables direct data transfer between GPUs and other
devices like InfiniBand networks or NVMe SSDs, bypassing the CPU and
system
```

## 教训

跨模块边界（TRT-LLM ↔ 自定义 offload layer）必须明确约定 tensor shape。
这次是因为 v2 layer 的接口设计照搬 izzy（`(batch_size,)` per-request），
但 TRT-LLM 传过来的是 per-token，没人在中间做 reshape。

定位：v1 path 偷偷做了 `position_ids[-1]` 转换却没文档化，导致 v2 设计
时按 layer 的 contract 来期望 per-request shape，结果跟 caller 不匹配。
**API contract 双方应该都写注释**——只在一端写注释救不了 future caller。
