### 阶段 2：功能 Slice — 工具实现

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/agent/stage-2.md，完成所有任务后确认完成`

**目标**：实现 3 个核心工具（FinancialSearchTool、CalculatorTool、KnowledgeGraphTool），每个工具独立可测。

**前置依赖**：阶段 1 已完成（BaseTool 协议已定义）。

**注意**：KnowledgeGraphTool 依赖 Plan A（GraphRAG）的 GraphStore。如果 Plan A 未完成，先实现空壳（run 返回 "GraphStore 未启用"），Plan A 完成后填充实现。

**任务清单**：

- [x] 2.1 创建 `src/agent/tools/financial_search.py` — FinancialSearchTool
  ```python
  class FinancialSearchTool(BaseTool):
      name = "financial_search"
      description = (
          "检索金融文档知识库。当你需要查找公司财报数据、行业分析、"
          "经济指标时使用。输入：query（自然语言查询）。"
      )

      def __init__(self, pipeline):  # RAGPipeline 实例
          self.pipeline = pipeline

      def run(self, query: str, **kwargs) -> ToolResult:
          try:
              result = self.pipeline.query(query)
              return ToolResult(success=True, output=result.answer)
          except Exception as e:
              return ToolResult(success=False, output=f"检索失败: {e}")
  ```
  - 依赖注入：构造函数接收 RAGPipeline 实例
  - 错误处理：任何异常都返回 ToolResult(success=False)，不抛出

- [x] 2.2 创建 `src/agent/tools/knowledge_graph.py` — KnowledgeGraphTool
  ```python
  class KnowledgeGraphTool(BaseTool):
      name = "knowledge_graph"
      description = (
          "查询金融实体关系图谱。当你需要对比两家公司、追踪指标变化、"
          "查找因果关联时使用。输入：entity（实体名称）。"
      )

      def __init__(self, graph_store=None):  # GraphStore 实例（可选）
          self.graph_store = graph_store

      def run(self, entity: str, **kwargs) -> ToolResult:
          if self.graph_store is None:
              return ToolResult(success=False, output="知识图谱未启用")
          try:
              triples = self.graph_store.query_neighbors(entity)
              if not triples:
                  return ToolResult(success=True, output=f"未找到 {entity} 的关联知识")
              lines = [f"- {t.head} {t.relation} {t.tail}" for t in triples]
              return ToolResult(success=True, output="\n".join(lines))
          except Exception as e:
              return ToolResult(success=False, output=f"图谱查询失败: {e}")
  ```
  - graph_store=None 时返回"未启用"，不抛异常
  - 格式化为易读的列表格式

- [x] 2.3 创建 `src/agent/tools/calculator.py` — CalculatorTool
  ```python
  class CalculatorTool(BaseTool):
      name = "calculator"
      description = (
          "执行金融指标计算。当你需要计算比率、增长率、对比数值时使用。"
          "输入：expression（Python 数学表达式）。"
      )

      ALLOWED_NAMES = {
          "abs": abs, "round": round, "min": min, "max": max,
          "sum": sum, "len": len, "sorted": sorted,
          "float": float, "int": int,
      }

      def run(self, expression: str, **kwargs) -> ToolResult:
          try:
              result = eval(expression, {"__builtins__": {}}, self.ALLOWED_NAMES)
              return ToolResult(success=True, output=str(result))
          except Exception as e:
              return ToolResult(success=False, output=f"计算错误: {e}")
  ```
  - **安全设计**：`__builtins__: {}` 禁止所有内置函数
  - 白名单仅暴露数学相关函数
  - 没有 import、open、exec、eval 嵌套
  - 输入长度限制：expression > 500 字符直接拒绝

- [x] 2.4 编写测试 `tests/test_agent_tools.py`
  - `test_financial_search_success` — mock RAGPipeline，验证返回
  - `test_financial_search_failure` — mock 抛异常，验证 ToolResult(success=False)
  - `test_knowledge_graph_success` — mock GraphStore，验证输出格式
  - `test_knowledge_graph_disabled` — graph_store=None 返回"未启用"
  - `test_calculator_basic` — `2 + 3` → `5`
  - `test_calculator_percentage` — `(31.2 - 25.8) / 25.8 * 100` → 正确百分比
  - `test_calculator_security_builtins` — `__import__("os")` → 失败
  - `test_calculator_security_open` — `open("/etc/passwd")` → 失败
  - `test_calculator_security_exec` — `exec("print(1)")` → 失败
  - `test_calculator_security_long_expression` — 超长表达式被拒绝

- [x] 2.5 验证
  - `ruff check src/agent/tools/` 无错误
  - `pytest tests/test_agent_tools.py -v` 全部通过
  - 安全测试全部通过（Calculator 拒绝危险输入）

**验收标准**：
- 3 个工具实现完整，各自独立可测
- CalculatorTool 安全沙箱通过所有安全测试
- KnowledgeGraphTool 在无 GraphStore 时优雅降级
- 所有测试通过

**完成确认**：

- [x] 阶段 2 全部任务完成，已通过验收标准
