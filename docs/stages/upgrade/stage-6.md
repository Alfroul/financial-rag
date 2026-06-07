### 阶段 6：WebSocket Streaming — 实时流式输出

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-6.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-6.md`、`src/api/routes/query.py`、`src/api/app.py`

**目标**：在 FastAPI 中添加 WebSocket endpoint，实现真正的双向流式输出，替代现有 SSE 方案，让 Gradio 前端获得实时打字机效果。

**前置依赖**：阶段 1 已完成（LLM 已切换）。阶段 3（Gradio）最好已完成，否则无法验证前端效果。

**任务清单**：

1. WebSocket Endpoint
   - 修改 `src/api/routes/query.py`：
     - 新增 `@router.websocket("/ws/chat")` endpoint
     - 接收 JSON 消息：`{"query": "...", "options": {...}}`
     - 调用 `RAGPipeline.astream_query()` 流式生成
     - 逐 token 发送 JSON 帧：`{"type": "token", "content": "..."}`
     - 发送元数据帧：`{"type": "sources", "data": [...]}`（检索来源）
     - 发送完成帧：`{"type": "done", "metrics": {...}}`
     - 处理断连：客户端主动断开时清理资源
     - 错误帧：`{"type": "error", "message": "..."}`

2. 消息协议设计
   - 客户端 → 服务端：
     ```json
     {"type": "query", "query": "茅台的ROE是多少", "options": {"top_k": 5}}
     {"type": "cancel"}  // 取消当前生成
     ```
   - 服务端 → 客户端：
     ```json
     {"type": "token", "content": "贵州"}
     {"type": "token", "content": "茅台"}
     {"type": "sources", "data": [{"title": "...", "score": 0.92}]}
     {"type": "metrics", "data": {"latency_ms": 2340, "tokens": 156}}
     {"type": "done"}
     {"type": "error", "message": "..."}
     ```

3. 流式 Pipeline 适配
   - `src/rag_pipeline.py`：确保 `astream_query()` 正确 yield token
   - 检查 `MimoLLM.astream_chat()` 的流式输出格式与 WebSocket 帧匹配
   - 处理流式中断：客户端 disconnect 时停止 LLM 生成（避免浪费 token）

4. Gradio 前端适配
   - `src/ui_gradio/chat.py`：
     - 使用 `gr.Chatbot` 的 `stream_chat` 方法连接 WebSocket
     - 或使用自定义 JS 组件通过 WebSocket 接收流式数据
   - 确保打字机效果平滑，无明显卡顿

5. 连接管理
   - 简单的连接状态管理（无复杂并发需求）
   - 超时处理：连接建立后 60s 无消息则主动断开
   - 并发限制：单 IP 最大 5 个同时连接

6. 测试
   - `tests/test_websocket.py`：
     - 使用 `httpx` 或 `websockets` 库测试 WebSocket 连接
     - 测试正常流式输出（收到 token 帧序列 + done 帧）
     - 测试错误处理（发送无效消息）
     - 测试断连清理
   - 运行 `pytest tests/ -x -q`

**验收标准**：
- WebSocket endpoint 可连接，流式输出 token 逐帧到达
- Gradio 前端展示平滑的打字机效果
- 消息协议完整（token/sources/metrics/done/error）
- 断连不导致服务端异常
- 全量测试通过

**技术备注**：
- FastAPI 原生支持 WebSocket，不需要额外库
- Gradio 4.x 的 `gr.Chatbot` 支持流式，但与 WebSocket 的集成可能需要自定义 JS
- 如果 Gradio WebSocket 集成复杂度太高，退化为 Gradio 原生流式（generator yield）+ FastAPI WebSocket 作为独立 API
- 保留现有 SSE endpoint 不删除，WebSocket 是增量功能
