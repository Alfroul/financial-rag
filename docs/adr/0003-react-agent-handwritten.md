# ADR 0003: 手写 ReAct 而不用 LangChain/LlamaIndex

## 状态

已接受

## 上下体

金融分析 Agent 需要一个 ReAct（Reasoning + Acting）循环框架。候选方案：

1. **手写 ReAct** — 自己实现 Thought → Action → Observation 循环
2. **LangChain Agent** — 使用 LangChain 的 AgentExecutor + Tool 抽象
3. **LlamaIndex Agent** — 使用 LlamaIndex 的 Agent 框架

## 决策

选择 **手写 ReAct**，核心循环约 60 行代码。

## 原因

### 为什么不用 LangChain

| 维度 | 手写 ReAct | LangChain Agent |
|------|-----------|----------------|
| 核心代码量 | ~60 行 | 框架 2000+ 行 |
| 可控性 | 每一行都能解释 | Prompt 模板藏在框架里 |
| 调试 | 出错就在 60 行里 | stack trace 嵌套 20 层 |
| 工具数量 | 3-4 个（有限，不需要框架级发现） | 框架适合工具多、动态发现的场景 |
| 面试表现 | "我理解 ReAct 原理" | "我用了 LangChain" = "我调了 API" |
| 依赖 | 零额外依赖 | langchain + langchain-core + 多个插件 |

### 关键判断

1. **工具数量有限**（3-4 个），不需要框架级的工具发现、路由、重试机制
2. **面试叙事优先**——这个项目的调性是"自己造轮子、理解原理"，用 LangChain 会破坏叙事
3. **调试体验**——ReAct 循环需要频繁调 Prompt，手写的每一行都能精确控制
4. **依赖最小化**——项目已有 SiliconFlow LLM 封装，不需要 LangChain 的 LLM 抽象层

### 代价与缓解

| 代价 | 缓解措施 |
|------|---------|
| 需要自己调 Prompt 格式稳定性 | few-shot 示例 + 兜底 Action Parser |
| 没有内置 Memory 管理 | 当前不需要多轮 Agent 对话，单任务独立 |
| 没有内置 Observability | 每步 Thought/Action/Observation 已记录 |
| LLM 可能不遵循 Action 格式 | 兜底逻辑：解析失败时直接返回答案 |

## 后果

- 正面：完全可控、零额外依赖、面试加分、调试方便
- 负面：需要自己维护 Action Parser 和 Prompt 稳定性
- 缓解：Action Parser 有兜底逻辑，Prompt 用 few-shot 增强格式遵循
