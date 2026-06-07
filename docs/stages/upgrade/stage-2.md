### 阶段 2：可观测性 — 接入 Langfuse

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-2.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-2.md`、`src/rag_pipeline.py`、`src/metrics/collector.py`

**目标**：接入 Langfuse，为每次 RAG query 建立全链路 trace（检索 → rerank → 纠错 → LLM 生成），替代当前简陋的 MetricsCollector。

**前置依赖**：阶段 1 已完成（LLM 模块已切换到 MiMo）。

**任务清单**：

1. 依赖安装
   - `requirements.txt` 新增 `langfuse>=2.0.0`
   - `config.yaml` 新增 `langfuse` 配置段：
     ```yaml
     langfuse:
       enabled: false
       public_key: ""      # 从环境变量 LANGFUSE_PUBLIC_KEY 读取
       secret_key: ""      # 从环境变量 LANGFUSE_SECRET_KEY 读取
       host: "https://cloud.langfuse.com"  # 或本地部署地址
     ```
   - `src/config.py` 新增 `LangfuseConfig` dataclass 和对应 property

2. Langfuse 封装模块
   - 创建 `src/observability/__init__.py`
   - 创建 `src/observability/langfuse_tracer.py`：
     - `LangfuseTracer` 类，封装 Langfuse SDK 的 trace/span 管理
     - `start_trace(query: str) -> str`：创建新 trace，返回 trace_id
     - `start_span(trace_id, name, input_data) -> str`：创建子 span
     - `end_span(span_id, output_data, metadata)`：结束 span，记录结果
     - `end_trace(trace_id, output, metadata)`：结束 trace
     - `record_llm_call(span_id, model, prompt_tokens, completion_tokens, latency)`：记录 LLM 调用详情
     - `flush()`：强制上传 trace 数据

3. 集成到 RAG Pipeline
   - `src/rag_pipeline.py` 的 `query()` 和 `aquery()` 方法：
     - 入口处 `start_trace(query)`
     - 检索阶段：`start_span("retrieval")` → 检索 → `end_span`，记录检索结果数量和耗时
     - Rerank 阶段：`start_span("rerank")` → 重排 → `end_span`，记录重排前后排序变化
     - 纠错阶段：`start_span("correction")` → 纠错 → `end_span`，记录纠错触发次数和修正内容
     - LLM 阶段：`start_span("generation")` → 生成 → `end_span`，记录 token 消耗
     - 出口处 `end_trace`，记录最终回答和总耗时
   - 通过构造函数注入 `LangfuseTracer`（可选依赖，`None` 时不记录）

4. 集成到 Agent
   - `src/agent/react.py`：在 ReAct 循环的每个 step 创建 span
   - 每个 tool call 记录为独立 span（tool name、input、output）

5. 环境配置
   - `.env.example` 新增 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`
   - `docker-compose.yml` 可选添加 Langfuse 自部署服务（注释掉，默认用 Cloud）

6. 测试
   - `tests/test_langfuse_tracer.py`：
     - 测试 `LangfuseTracer` 的 trace/span 创建和结束
     - 测试配置 `enabled=false` 时 tracer 为 no-op
     - Mock Langfuse SDK，验证调用参数正确
   - 运行 `pytest tests/ -x -q` 确认全量通过

**验收标准**：
- `config.yaml` 中 `langfuse.enabled=true` 时，每次 query 在 Langfuse Dashboard 可见完整 trace
- `langfuse.enabled=false` 时，系统行为与升级前完全一致（零性能开销）
- 每个 trace 包含 4-5 个 span（retrieval/rerank/correction/generation）
- 全量测试通过

**技术备注**：
- Langfuse Python SDK 支持装饰器模式和手动 API，本项目用手动 API（更灵活）
- 如果 Langfuse Cloud 不可用，tracer 应优雅降级（日志警告但不中断查询）
