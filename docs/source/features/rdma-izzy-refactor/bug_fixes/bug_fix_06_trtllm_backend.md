# Bug Fix #06 — 让 draft server 支持 `--backend trtllm` (替换 transformers)

## 触发条件 / 需求

`_TransformersBackend` 用 HuggingFace transformers 做 draft model
推理，跟生产场景脱节（生产场景应该 target 和 draft 都用 TRT-LLM）。
用户要求把 draft 端也换成 TRT-LLM。

## 子问题：importlib hack 阻止真 TRT-LLM 加载

`draft_rdma_server_v2.py` 顶部为了允许在 **host**（没装 TRT-LLM 但可
以跑 mock backend 的环境）跑，做了：

```python
for pkg in ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.speculative"):
    _stub_pkg(pkg)
_import_module("tensorrt_llm._torch.speculative.draft_api_protocol", ...)
```

这个 stub 把 `tensorrt_llm` 注册成空 module。后面 `_TrtLlmBackend.__init__`
执行 `from tensorrt_llm import LLM` 时拿到 stub，`LLM` 不存在 → ImportError。

## 修复

### 1. `_TrtLlmBackend` 类（新增到 `draft_rdma_server_v2.py`）

```python
class _TrtLlmBackend:
    def __init__(self, model_path, prompt, max_draft_len, device="cuda:0"):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        from transformers import AutoTokenizer
        ...
        self.llm = LLM(
            model=model_path,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,        # KV prefix cache 命中
                free_gpu_memory_fraction=0.4,
            ),
            cuda_graph_config=None,             # ibverbs path 不能用 graph
            max_batch_size=8,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_draft_len, temperature=0.0, top_p=1.0,
        )

    def step(self, request_id, last_token, position, round_seq):
        sess = self._ensure_session(request_id)
        # Rewind on reject + append accepted last_token
        if position < len(sess["tokens"]):
            sess["tokens"] = sess["tokens"][:int(position)]
        if not sess["tokens"] or sess["tokens"][-1] != int(last_token):
            sess["tokens"].append(int(last_token))
        # Generate max_draft_len more tokens
        outputs = self.llm.generate([sess["tokens"]], sampling_params=self.sampling_params)
        draft_tokens = list(outputs[0].outputs[0].token_ids)[: self.max_draft_len]
        sess["tokens"].extend(int(t) for t in draft_tokens)
        return draft_tokens
```

设计要点：
- **session = accumulated token list**：跟 transformers backend 一样维护
  request → token history，但 KV cache 由 TRT-LLM 内部管理（不用我们
  手动 truncate）
- **每 step 调用 `llm.generate([full_tokens])`**：依赖 TRT-LLM 的
  `KvCacheConfig(enable_block_reuse=True)` 让 prefix block 自动命中，
  增量只 prefill 最后一个新 token + decode max_draft_len 个
- **reject 处理**：直接 truncate session 的 token list；TRT-LLM 下次
  请求时会 lookup 新的 prefix hash，只匹配到截断后的那部分

### 2. import 顺序修复

```python
try:
    # In dev container — natural import works (incl. LLM, SamplingParams)
    from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol
    from tensorrt_llm._torch.speculative.ibverbs_endpoint import ...
    from tensorrt_llm._torch.speculative.spec_decode_channel import ...
except Exception:
    # Host fallback — stub + importlib hack (mock backend only)
    ...
```

只在真 import 失败时才走 stub 路径。容器内 trtllm backend 不再被 stub 拦截。

### 3. argparse

```python
ap.add_argument("--backend", choices=["mock", "transformers", "trtllm"], default="mock")
```

## 验证

### 隔离测试

```python
llm = LLM(model="Qwen3-0.6B", kv_cache_config=KvCacheConfig(enable_block_reuse=True, ...),
          cuda_graph_config=None, max_batch_size=8)
sp = SamplingParams(max_tokens=5, temperature=0.0)
out = llm.generate([prompt_ids], sampling_params=sp)
# Round 1 → 5 tokens
# Round 2 (append accepted tokens, regenerate) → 5 more tokens
```

输出合理 token，性能比 transformers 高一个量级（实测 17 req/s vs
transformers backend ~3 req/s）。

### Phase 5 v2 e2e

```bash
# Server: TRT-LLM draft (Qwen3-0.6B) on GPU 2 / mlx5_1
python3 draft_rdma_server_v2.py \
  --nic mlx5_1 --port 47000 --tcp-prompt-port 47001 \
  --backend trtllm \
  --model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \
  --prompt "Explain GPUDirect RDMA in one short sentence." ...

# Target: TRT-LLM (Qwen3-8B) on GPU 1 / mlx5_0, v2 path
TLLM_DRAFT_RDMA_OFFLOAD_V2=1 python3 examples/llm-api/llm_rdma_draft_offload.py \
  --model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-8B ...
```

**Output**（字节级匹配 step-3 baseline）：

```
GPUDirect RDMA enables direct data transfer between GPUs and other
devices like InfiniBand networks or NVMe SSDs, bypassing the CPU and
system
```

Server log 显示 ~5 个 round_seq 完成，每 round 5-90ms（17.91 it/s）。

## 副作用

- v2 server 进程现在加载 TRT-LLM 后会占 ~5-10GB GPU 内存（取决于
  `free_gpu_memory_fraction`）
- 启动比 transformers backend 慢一点（~5s vs ~2s），但 per-step
  latency 低很多
- `--backend transformers` 仍保留，主要用于无 TRT-LLM 的快速 e2e
  protocol 验证

## 教训

importlib hack 看起来无害，但会"污染" sys.modules 阻止后续真 import。
应该用 try/except 守护，保证 happy-path 用真 import，hack 只在 fallback
出现。
