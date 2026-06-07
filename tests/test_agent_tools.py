from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.financial_search import FinancialSearchTool
from src.agent.tools.knowledge_graph import KnowledgeGraphTool

# ── FinancialSearchTool ──────────────────────────────────────────────


class TestFinancialSearchTool:
    def test_success(self):
        pipeline = MagicMock()
        pipeline.query.return_value = {"answer": "贵州茅台2024年营收1680亿", "sources": []}
        tool = FinancialSearchTool(pipeline)

        result = tool.run(query="贵州茅台营收")

        assert result.success is True
        assert "1680亿" in result.output
        pipeline.query.assert_called_once_with("贵州茅台营收")

    def test_failure(self):
        pipeline = MagicMock()
        pipeline.query.side_effect = RuntimeError("LLM timeout")
        tool = FinancialSearchTool(pipeline)

        result = tool.run(query="test")

        assert result.success is False
        assert "检索失败" in result.output

    def test_empty_query(self):
        pipeline = MagicMock()
        tool = FinancialSearchTool(pipeline)

        result = tool.run(query="")

        assert result.success is False
        assert "不能为空" in result.output


# ── KnowledgeGraphTool ───────────────────────────────────────────────


class TestKnowledgeGraphTool:
    def test_success(self):
        t1 = MagicMock()
        t1.head = "贵州茅台"
        t1.relation = "营收"
        t1.tail = "1680亿"
        t2 = MagicMock()
        t2.head = "贵州茅台"
        t2.relation = "属于"
        t2.tail = "白酒行业"

        store = MagicMock()
        store.query_neighbors.return_value = [t1, t2]
        tool = KnowledgeGraphTool(graph_store=store)

        result = tool.run(entity="贵州茅台")

        assert result.success is True
        assert "- 贵州茅台 营收 1680亿" in result.output
        assert "- 贵州茅台 属于 白酒行业" in result.output

    def test_no_results(self):
        store = MagicMock()
        store.query_neighbors.return_value = []
        tool = KnowledgeGraphTool(graph_store=store)

        result = tool.run(entity="未知实体")

        assert result.success is True
        assert "未找到" in result.output

    def test_disabled(self):
        tool = KnowledgeGraphTool(graph_store=None)

        result = tool.run(entity="贵州茅台")

        assert result.success is False
        assert "未启用" in result.output

    def test_query_failure(self):
        store = MagicMock()
        store.query_neighbors.side_effect = RuntimeError("db error")
        tool = KnowledgeGraphTool(graph_store=store)

        result = tool.run(entity="贵州茅台")

        assert result.success is False
        assert "图谱查询失败" in result.output


# ── CalculatorTool ───────────────────────────────────────────────────


class TestCalculatorTool:
    def test_basic(self):
        tool = CalculatorTool()
        result = tool.run(expression="2 + 3")
        assert result.success is True
        assert result.output == "5"

    def test_percentage(self):
        tool = CalculatorTool()
        result = tool.run(expression="(31.2 - 25.8) / 25.8 * 100")
        assert result.success is True
        expected = str((31.2 - 25.8) / 25.8 * 100)
        assert result.output == expected

    def test_security_builtins(self):
        tool = CalculatorTool()
        result = tool.run(expression='__import__("os")')
        assert result.success is False

    def test_security_open(self):
        tool = CalculatorTool()
        result = tool.run(expression='open("/etc/passwd")')
        assert result.success is False

    def test_security_exec(self):
        tool = CalculatorTool()
        result = tool.run(expression='exec("print(1)")')
        assert result.success is False

    def test_security_long_expression(self):
        tool = CalculatorTool()
        long_expr = "1 + " * 200 + "1"
        result = tool.run(expression=long_expr)
        assert result.success is False
        assert "过长" in result.output

    def test_empty_expression(self):
        tool = CalculatorTool()
        result = tool.run(expression="")
        assert result.success is False
