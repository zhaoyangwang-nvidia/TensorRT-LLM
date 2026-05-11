# Bug Fix #01 — KV cache truncate 在 DynamicCache 上挂

## 触发条件

`_TransformersBackend.step` 检测到 `position < sess["position"]`（target 拒绝了部分
draft token，需要回卷 KV cache）的时候。

旧代码：

```python
sess["past_key_values"] = tuple(
    (k[:, :, :position, :], v[:, :, :position, :])
    for k, v in sess["past_key_values"]
)
```

## 根因

`for k, v in sess["past_key_values"]` 假设 `past_key_values` 是
`tuple[tuple[Tensor, Tensor]]`（老 transformers API）。

新 transformers（≥4.36，我们容器里是 5.8.0）默认返回 **`DynamicCache`** 对象。
Qwen3 模型也是。`DynamicCache` 是 iterable，迭代得到的是单个 layer 对象**不是 (k, v) tuple**。
解包就 raise：

```
ValueError: too many values to unpack (expected 2)
```

## 怎么发现的

在容器里跑：

```bash
docker run -d --name rdma_test --gpus all --device /dev/infiniband \
  -v /home/scratch.zhaoyangw_gpu/rdma:/home/scratch.zhaoyangw_gpu/rdma \
  -v /home/scratch.trt_llm_data:/scratch.trt_llm_data \
  tensorrt_llm/devel:latest-fanrongl sleep 1800

pip install transformers  # 容器里默认不装

# 加载 Qwen3-0.6B + prefill 12 token prompt
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    '/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B',
    torch_dtype=torch.float16).cuda()
ids = AutoTokenizer.from_pretrained(
    '/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B'
).encode('Explain GPUDirect RDMA in one short sentence.', return_tensors='pt').cuda()
out = m(ids, use_cache=True)
pkv = out.past_key_values
print('type:', type(pkv).__name__)               # DynamicCache
print('has crop:', hasattr(pkv, 'crop'))          # True

# 旧 truncate 路径
target = ids.shape[1] - 3
try:
    new_pkv = tuple((k[:, :, :target, :], v[:, :, :target, :]) for k, v in pkv)
except ValueError as e:
    print('FAIL:', e)                             # too many values to unpack
"
```

## 修复

`draft_rdma_server_v2.py` `_TransformersBackend.step`：

```python
if position < sess["position"]:
    # KV cache truncation on rejection.  Modern transformers (Qwen3 / Llama /
    # etc) use ``DynamicCache`` which exposes ``.crop(target_len)``; older
    # ``tuple[tuple[K,V]]`` style is supported as a fallback.
    pkv = sess["past_key_values"]
    if hasattr(pkv, "crop"):
        pkv.crop(position)
    else:
        pkv = tuple(
            (k[:, :, :position, :], v[:, :, :position, :]) for k, v in pkv
        )
        sess["past_key_values"] = pkv
    sess["position"] = position
```

`DynamicCache.crop(target_len)` 是 transformers 官方 in-place API，效果完全等价
旧 tuple 切片，但跨 attention 实现（flash / sdpa / eager）都一致。

## 验证

容器里端到端跑：

```
init_session: 12 token prompt → DynamicCache 建立 ✓
Round 1: position 12 → 17, draft=" \n\nThe GPUDirect"           ✓
Round 2 reject (rewind 17→15): .crop() 不挂, draft="UDirect RDMA is" ✓
Round 3: position 20+, draft=" a technique that allows for"     ✓

Full text:
"Explain GPUDirect RDMA in one short sentence.

The GPUDirect RDMA is a technique that allows for"
```

KV cache 回卷正确：round 2 是从 round 1 的 position-2（=15）继续推的，
不是从 17 继续推；这只有当 KV cache 真的被截到 15 时才成立——验证了
`.crop()` 实际生效。

## 影响范围

- 受影响：所有用新 transformers 的真模型（Qwen3、Llama 3+、Gemma 2、Mistral
  等），即 Phase 6 `_TransformersBackend` 全部目标用户
- 不受影响：mock backend（不用 transformers）、单测（pure-Python 路径）
- pre-fix 严重程度：**FATAL** —— 一旦 target 拒绝任何 draft token，server 就 raise

## 教训

- "在 host 跑 unit test"和"在容器跑真模型 e2e"覆盖的是不同 bug class。这个 bug
  unit test 完全看不到，必须真模型跑。
- 我之前一直推说"必须容器"是借口——`docker run` + 现成镜像就行，从开发到验证
  应该 15 分钟内完成。后续 phase 必须坚持容器验证。
