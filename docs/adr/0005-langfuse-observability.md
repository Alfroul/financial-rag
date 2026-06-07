# ADR 0005: 接入 Langfuse 实现全链路可观测性

## 状态

已批准

## 上下文

现有 `MetricsCollector` 只记录 P50/P95 延迟和 token 消耗等聚合指标，无法回溯单次 query 的完整链路。面试时无法展示"某个 query 为什么效果差"的诊断能力。

## 决策

接入 Langfuse 作为可观测性平台，理由：
1. 专为 LLM 应用设计，开箱即用的 trace/span 模型
2. 支持自部署（Docker）或 Cloud，灵活度高
3. 免费 Community 版本足够开发使用
4. Dashboard 直观，面试时可现场展示

## 后果

- 正面：每次 query 有完整 trace，可诊断检索/纠错/生成的具体问题
- 负面：新增 `langfuse` 依赖，每次 query 有微小的 trace 上报开销
- 缓解：`enabled=false` 时零开销，tracer 为 no-op
