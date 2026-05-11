# Bug Fix #02 — TCP `request_id` 跟 RDMA `slot` 在 server 端不是同一个 key

## 触发条件

Target 调 `push_prompt(request_id=7, prompt_tokens=...)`，然后通过 RDMA 发包。
RDMA 包的 `imm_data` 里 `slot=0`（因为 request 7 → 默认 route probe 从
`request_id & 0x0FFF` 开始，但例如 batch 第一个 request 经常落在 slot 0；
或者 IbverbsDraftOffloadLayer 主动 bind 到 slot 0）。

## 根因

Server 端的 namespace 是不一致的：

| 路径 | server 用什么当 session key |
|---|---|
| TCP `init_session` | `request_id` from `TcpPromptInit` payload（target 设置的逻辑 id） |
| RDMA main loop | `slot` from `imm_data`（wire 上唯一的 routing key） |

`backend._sessions[request_id]` 在 TCP push 时建立，但 RDMA 流量来时 server 拿
`slot` 去 `_sessions[slot]` 查，**两个 key 不一样** → `_ensure_session` 找不到
session，走 fallback 路径，没有 `--prompt` 时 raise：

```
RuntimeError: no session for request 0 and no fixed prompt set;
send a TcpPromptInit first
```

## 怎么发现的

在容器里跑 `scripts/smoke_phase6_e2e.py`：

```
Pushing prompt: 12 tokens via TCP port 41687
  ack ok, last_pos=11          ← TCP push 成功，server 把 session 存在 key=7
[draft-v2] backend.step failed: RuntimeError(
  'no session for request 0 and no fixed prompt set; send a TcpPromptInit first'
)                              ← RDMA 来时 server 用 slot=0 查不到
```

## 根本设计问题

Server 没有"target 的 request_id"概念——它只看见 wire 上的 `slot`。所以
TCP push 和 RDMA traffic **必须用同一个 namespace** 当 session key。
最自然的就是用 **wire-level slot**。

## 修复

`IbverbsDraftOffloadLayer.push_prompt`：

```python
def push_prompt(self, request_id, prompt_tokens):
    ...
    # Ensure the channel is started and the request has a route bound.
    self.bind_requests([request_id])
    route_status, route = self._channel.route_for_request(int(request_id))
    if route_status != self._channel_cls.Status.kOk or route is None:
        raise RuntimeError(...)
    wire_handle = int(route.slot)        # <-- 用 slot 当 wire-level key

    msg = {
        "msg_type": "prompt_init",
        "request_id": wire_handle,        # <-- TCP payload 里写的是 slot
        ...
    }
```

API 给上层调用者**没变**——还是传 `request_id`；
但内部转成 `slot` 再发 TCP。这样 server 收到的 `request_id` 字段值
其实是 slot，正好跟 RDMA 主循环用的 key 一致。

## 副作用 / 注意点

- `TcpPromptInit.request_id` 字段名跟实际含义不一致了——payload 里塞的
  是 slot 不是 logical id。**短期容忍**，长期看应该把 protocol 字段名
  改成 `slot` 或加一个独立 `slot` 字段。
- Server 端的 `_serve_one` 已经把 `msg.request_id` 当 session key 用，
  不需要改。RDMA main loop 用 `slot` 当 handle 也不需要改。
- `bind_requests` 是幂等的（已绑定就直接返回），所以 `push_prompt` 在
  retire / unbind 之后调用会重新触发 bind。配合 `_unbind_request` 清
  `_prompt_pushed_requests`，retire-rebind 流程正确。

## 验证

容器里跑 `smoke_phase6_e2e.py`（真模型 Qwen3-0.6B + 真 RDMA）：

```
TCP prompt listener up.
Connecting RDMA client to port 51819...
RDMA client connected.
Server in main loop.
Pushing prompt: 12 tokens via TCP port 41687
  ack ok, last_pos=11
  round 1: tokens=[4710, 785, 23690, 4656, 1226] decoded=' \n\nThe GPUDirect'
  round 2: tokens=[37992, 4835, 374, 264, 14762] decoded=' RDMA is a technique'
  round 3: tokens=[429, 6147, 369, 1550, 29599] decoded=' that allows for high-speed'

Full text:
"Explain GPUDirect RDMA in one short sentence.

The GPUDirect RDMA is a technique that allows for high-speed"

Phase 6 transformers backend e2e smoke PASSED
```

3 轮 RDMA round-trip + KV cache 跨轮持续 + 输出 token decode 后是合理英文，
确认 session 在 TCP push 时建立、RDMA 时被复用，跨轮 KV cache 累积。

## 教训

- 单测 mock backend 永远抓不到这个 bug——mock 不关心 session key 是什么。
  必须真模型 + 真 RDMA + 真 TCP 跑通一次 round-trip 才能暴露。
- 设计上：跨进程/跨协议的 channel 之间，**id 的 namespace 必须显式约定**
  在哪一层。这次的问题就是 TCP 用了 logical id，RDMA 用了 slot，谁也没
  写文档说它们要相等。
