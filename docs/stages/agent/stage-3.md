### 阶段 3：功能 Slice — ReAct 循环 + 集成

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/agent/stage-3.md，完成所有任务后确认完成`

**目标**：实现完整的 ReAct 循环（思考-行动-观察迭代），集成到 RAGPipeline 和 Streamlit UI。

**前置依赖**：阶段 1（骨架）+ 阶段 2（工具）已完成。

**任务清单**：

- [x] 3.1 实现 `src/agent/react.py` — ReActAgent 完整实现
  - `run(task: str) -> str` — 同步 ReAct 主循环（约 60 行核心逻辑）
  - `arun(task: str) -> str` — 异步版本（与 run 相同逻辑，LLM 调用用 achat）
  - `_build_initial_messages(task)` — 组装 system prompt + few-shot + 用户任务
  - `_parse_action(response) -> tuple | None` — Action 解析
  - `_extract_answer(response)` — 从 LLM 输出提取最终答案
  - `_call_llm(messages)` — 调用 LLM 的统一入口（便于 mock 测试）

- [x] 3.2 重复动作检测
  - 在循环中追踪最近 3 次的 (tool_name, tool_input) 对
  - 如果连续 2 次完全相同 → 追加提示"你已经在重复相同操作，请直接给出回答"
  - 防止 LLM 陷入循环

- [x] 3.3 改造 `src/rag_pipeline.py` — 新增 agent_query()
  - `agent_query(task, max_steps)` — 同步 Agent 查询
  - `aagent_query(task, max_steps)` — 异步 Agent 查询
  - `_create_agent(max_steps)` 方法：从当前组件组装 ReActAgent
  - agent.enabled=false 时 agent_query() 仍可调用（手动调用）

- [x] 3.4 改造 `src/ui/chat_tab.py` — 增加模式切换
  - 侧边栏（`src/ui/sidebar.py`）增加 radio：`["问答模式", "分析模式"]`
  - 问答模式 → `pipeline.query(user_input)`（现有逻辑不变）
  - 分析模式 → `pipeline.agent_query(user_input)`
  - 分析模式下显示 Agent 思考过程（每步 Thought + Action + Observation）
  - 用 `st.expander` 折叠思考过程，默认只显示最终答案

- [x] 3.5 新增 API 端点 — `src/api/routes/query.py`
  - 新增 `POST /api/v1/agent/query`
  - 请求体：`{"task": "...", "max_steps": 6}`
  - 返回：`{"answer": "...", "steps": [...]}`
  - steps 记录每步的 thought / action / observation

- [x] 3.6 编写测试
  - `tests/test_react_loop.py` — 12 项测试
  - `tests/test_agent_integration.py` — 5 项测试

- [x] 3.7 验证
  - `ruff check` 无错误
  - `pytest tests/test_react_loop.py tests/test_agent_integration.py tests/test_agent_tools.py -v` — 31 项全部通过

**验收标准**：
- [x] ReAct 循环完整实现，支持多步推理
- [x] Action 解析有兜底（格式错误不崩溃）
- [x] 步数限制和重复检测正常工作
- [x] Streamlit UI 有模式切换
- [x] API 端点可用
- [x] 所有测试通过

**完成确认**：

- [x] 阶段 3 全部任务完成，已通过验收标准
