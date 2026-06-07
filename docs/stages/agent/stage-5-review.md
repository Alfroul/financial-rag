# Agent 模块全局 Review 报告

> 审查日期：2026-06-01
> 审查范围：`src/agent/` 及其在 API/UI/Pipeline 中的集成

---

## 5.1 安全性审查

### CalculatorTool 沙箱评估

**当前实现** (`src/agent/tools/calculator.py:35`)：
```python
eval(expression, {"__builtins__": {}}, self.ALLOWED_NAMES)
```

**已覆盖的攻击向量**（测试通过）：
- `__import__("os")` — NameError (builtins 已清除)
- `open("/etc/passwd")` — NameError
- `exec("print(1)")` — NameError
- 超长表达式 (>500字符) — 拒绝执行

**已知风险 — MEDIUM**：
`eval` + `{"__builtins__": {}}` 的沙箱在 CPython 中可通过以下方式绕过：
```python
().__class__.__bases__[0].__subclasses__()  # 获取所有已加载的类
[type('X',(),{'__subclasses__':lambda s:[c for c in ...]})]
```
白名单中的 `type` 不在 ALLOWED_NAMES 中，但仍需注意 Python 版本变化。

**建议**：当前风险可接受（工具由 LLM 调用，非直接用户输入）。如需加固，可改用 AST 解析器只允许数字、运算符和白名单函数调用。

### 用户输入验证

- **超长输入**：`_classify_task()` 在 500 字符处截断 ✅
- **空输入**：Pydantic `min_length=1` + 工具层 `not expression.strip()` 双重校验 ✅
- **Prompt Injection**：用户无法直接注入 Action 行。LLM 生成 Action，解析器只读 LLM 输出。低风险 ✅
- **API Schema 缺失上界**：`AgentQueryRequest.task` 无 `max_length` 约束，仅靠 Agent 内部截断。**建议在 Schema 层加 `max_length=2000`**。

### ToolResult 信息泄露 — LOW

`financial_search.py:28` 和 `knowledge_graph.py:33` 在失败时返回 `f"检索失败: {e}"`。异常消息可能包含内部路径或数据库错误信息。**建议仅返回通用错误消息，详细错误写入日志**。

---

## 5.2 代码质量审查

### 正则解析器覆盖度 — GOOD

- `_ACTION_RE = r"Action:\s*(\w+)\((.+?)\)"` 覆盖 `Action: tool(...)` 格式
- `_KWARG_RE = r'(\w+)\s*=\s*"([^"]*)"'` 覆盖 `key="value"` 格式
- 兜底逻辑：无法解析 kwargs 时整个字符串作为 `query` 参数 ✅
- 测试覆盖：标准格式、无 Action、格式错误、多参数 ✅

### 错误传播链 — GOOD

工具失败 → `ToolResult(success=False, output=...)` → Observation 传回 LLM → LLM 可据此调整策略。链路完整 ✅

### 代码重复 — MEDIUM

`run()` (lines 208-257) 和 `arun()` (lines 259-305) 逻辑完全重复，仅同步/异步调用不同。同理 `_handle_edge_case` / `_ahandle_edge_case`。约 100 行纯重复代码。**建议提取公共逻辑到辅助方法**。

### 未使用文件

`src/agent/prompts.py` 定义了 `REACT_SYSTEM_PROMPT`、`REACT_FEW_SHOT`、`STOP_PROMPT`，但 `react.py` 定义了自己的 `_SYSTEM_PROMPT` 和 `_FEW_SHOT_EXAMPLES`，prompts.py 未被任何文件导入。**已删除**。

### 类型注解 — GOOD

- `BaseTool.run(**kwargs) -> ToolResult` ✅
- `ReActAgent.__init__` 参数类型明确 ✅
- `llm: Any` 可接受（接口多样）✅
- `_parse_action` 返回 `tuple[str, dict[str, str]] | None` ✅

---

## 5.3 架构审查

### BaseTool 协议 — GOOD，无过度设计

仅 `name` / `description` / `run()` 三个要求，足够通用。无多余抽象。

### 耦合度 — GOOD

- `ReActAgent` 依赖 `BaseTool` 协议，不依赖具体工具实现
- `RAGPipeline._create_agent()` 负责组装，Agent 不感知 Pipeline
- 工具注册通过构造函数传入，简单直接

### 灵活性 — GOOD

添加新工具只需：继承 BaseTool → 实现 run() → 传入 tools 列表。无框架约束。

---

## 5.4 鲁棒性审查

### LLM 输出格式不稳定 — MEDIUM

- 如果 LLM 输出 `action:` (小写) 而非 `Action:`，正则不匹配，会被视为最终答案
- 如果 LLM 输出 JSON 格式 Action（如 few-shot 中的旧格式），解析失败
- **建议**：`_ACTION_RE` 加 `re.IGNORECASE` 标志

### 工具超时/API 限流 — LOW

工具异常被 `try/except` 捕获，不会导致 Agent 崩溃。但无重试机制，无超时控制。

### 并发安全 — ACCEPTABLE

`_steps` 列表为实例变量，非线程安全。但当前每个请求创建独立 Agent 实例（`_create_agent()` 每次 new），无共享状态。

### 内存增长 — LOW

`messages` 列表在长对话中无限增长。但 `max_steps=6` 限制了最大增长量（约 12 条消息），可接受。

### `_steps` 未重置

如果同一 Agent 实例被复用（当前不会，但 API 层可能），`_steps` 会累积。**建议在 `run()`/`arun()` 开头清空**。

---

## 5.5 测试完整性

| 测试文件 | 覆盖范围 | 评估 |
|----------|---------|------|
| `test_agent_tools.py` (14项) | 三个工具的正常/异常/安全用例 | ✅ 覆盖良好 |
| `test_react_loop.py` (10项) | 单步/多步/超步数/重复动作/解析/提取 | ✅ 覆盖良好 |
| `test_agent_integration.py` (4项) | Pipeline 集成/工具组装/多步推理 | ✅ 基本覆盖 |

**缺失的测试**：
- [ ] `_classify_task` 边界测试（空、超长、闲聊、知识问答）
- [ ] `_extract_answer` 多行 Thought 剥离
- [ ] CalculatorTool 更多绕过尝试（`type()`, `object.__subclasses__`）
- [ ] 异步流程测试（`arun()` 未在任何测试中覆盖）
- [ ] `_build_few_shot_messages` 输出验证

---

## 5.6 文档完整性

| 文档项 | 状态 |
|--------|------|
| CONTEXT.md Agent 术语 | ✅ 包含所有核心术语 |
| docs/adr/ 架构决策 | ✅ `0003-react-agent-hand-written.md` 存在 |
| plan-agent.md 阶段勾选 | ❌ 全部标记"未开始"，实际已完成 |
| API 文档 | ✅ `/agent/query` 端点已在 schemas.py 定义 |

---

## 总结

| 类别 | 评级 | 说明 |
|------|------|------|
| 安全性 | **GOOD** | 沙箱基本可靠，有已知但可接受的 eval 风险 |
| 代码质量 | **GOOD** | 可读性好，有少量代码重复 |
| 架构 | **GOOD** | 职责清晰，耦合度低 |
| 鲁棒性 | **FAIR** | LLM 输出变体处理可改进 |
| 测试 | **GOOD** | 核心路径覆盖完整，异步测试缺失 |
| 文档 | **FAIR** | plan-agent.md 状态未更新 |

**总体评估**：Agent 模块设计合理、实现简洁，作为手写 ReAct 方案满足项目需求。主要改进点为：代码重复消除、plan-agent.md 状态更新、异常信息脱敏。
