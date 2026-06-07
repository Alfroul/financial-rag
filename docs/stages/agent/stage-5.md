### 阶段 5：全局 Review — Agent

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/agent/stage-5.md，对 Agent 模块进行全面的 Code Review`

**目标**：对 Agent 模块进行全局审查，重点关注安全性（Calculator 沙箱）、代码质量和 ReAct 循环的鲁棒性。

**前置依赖**：阶段 1-4 全部完成。

**任务清单**：

- [x] 5.1 安全性审查（最高优先级）
  - CalculatorTool 的沙箱是否可绕过
    - 测试所有已知绕过方式：`__import__`、`getattr`、`type`、`object.__subclasses__`
    - 输入长度限制是否有效
    - 是否有 eval 嵌套风险
  - Agent 模式下用户输入是否经过验证
    - 超长输入是否截断
    - 是否有 prompt injection 风险（用户在任务中注入 Action 行）
  - ToolResult 是否可能泄露内部错误信息

- [x] 5.2 代码质量审查
  - ReAct 循环的可读性——每个方法是否职责单一
  - Action Parser 的正则是否覆盖所有 LLM 输出变体
  - 错误传播链——工具失败是否正确传递给 LLM 做下一步判断
  - 类型注解完整性

- [x] 5.3 架构审查
  - BaseTool 协议是否足够通用，是否过度设计
  - ReActAgent 与 RAGPipeline 的耦合度
  - 工具注册机制是否灵活（能否轻松添加新工具）
  - 是否有不必要的抽象层

- [x] 5.4 鲁棒性审查
  - LLM 输出格式不稳定时的行为
  - 工具超时或 API 限流时的处理
  - 并发请求下的线程安全
  - 内存使用——长对话的 messages 列表是否会无限增长

- [x] 5.5 测试完整性
  - 所有阶段的测试通过
  - 安全测试覆盖率
  - 边界情况覆盖：空输入、超长输入、全英文输入、重复调用
  - 集成测试：端到端 Agent 查询

- [x] 5.6 文档完整性
  - CONTEXT.md 包含所有 Agent 术语
  - docs/adr/ 包含所有架构决策
  - plan-agent.md 所有阶段已勾选
  - API 文档包含 Agent 端点

**完成确认**：

- [x] 阶段 5 全部任务完成，Agent 模块 Review 通过

> Review 完成于 2026-06-01。详细报告见 `docs/stages/agent/stage-5-review.md`。
> 评估结果：安全性 GOOD、代码质量 GOOD、架构 GOOD、鲁棒性 FAIR、测试 GOOD、文档 FAIR。
