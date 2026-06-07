"""MCP Server 测试：Server 启动、Tool schema、Mock 调用。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 辅助：构造 mock pipeline / retriever / graph_retriever
# ---------------------------------------------------------------------------

def _make_retrieval_result(content: str, score: float, source: str = "test.pdf") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.score = score
    r.metadata = {"source": source}
    r.doc_id = "doc-1"
    return r


def _make_triple(head: str, relation: str, tail: str) -> MagicMock:
    t = MagicMock()
    t.head = head
    t.relation = relation
    t.tail = tail
    return t


# ---------------------------------------------------------------------------
# 测试 1：FastMCP 实例创建不报错
# ---------------------------------------------------------------------------

class TestServerCreation:
    def test_import_server_module(self):
        from src.mcp_server.server import mcp
        assert mcp is not None
        assert mcp.name == "financial-rag"


# ---------------------------------------------------------------------------
# 测试 2：Tool schema 定义正确
# ---------------------------------------------------------------------------

class TestToolSchemas:
    @pytest.fixture(autouse=True)
    def _init(self):
        from src.mcp_server.server import mcp
        self.mcp = mcp

    def _get_tool_names(self):
        from src.mcp_server.server import mcp
        tool_manager = mcp._tool_manager
        return {t.name for t in tool_manager._tools.values()}

    def test_financial_search_registered(self):
        from src.mcp_server.server import mcp
        tool_manager = mcp._tool_manager
        assert "financial_search" in tool_manager._tools

    def test_knowledge_graph_query_registered(self):
        from src.mcp_server.server import mcp
        tool_manager = mcp._tool_manager
        assert "knowledge_graph_query" in tool_manager._tools

    def test_financial_analysis_registered(self):
        from src.mcp_server.server import mcp
        tool_manager = mcp._tool_manager
        assert "financial_analysis" in tool_manager._tools

    def test_tool_has_description(self):
        from src.mcp_server.server import mcp
        tool_manager = mcp._tool_manager
        for name in ("financial_search", "knowledge_graph_query", "financial_analysis"):
            tool = tool_manager._tools[name]
            assert tool.description, f"{name} 缺少 description"

    def test_financial_search_params(self):
        from src.mcp_server.server import mcp
        tool = mcp._tool_manager._tools["financial_search"]
        schema = tool.parameters
        props = schema.get("properties", {})
        assert "query" in props
        assert "top_k" in props
        assert schema["properties"]["query"]["type"] == "string"

    def test_knowledge_graph_query_params(self):
        from src.mcp_server.server import mcp
        tool = mcp._tool_manager._tools["knowledge_graph_query"]
        schema = tool.parameters
        props = schema.get("properties", {})
        assert "entity" in props
        assert "relation" in props

    def test_financial_analysis_params(self):
        from src.mcp_server.server import mcp
        tool = mcp._tool_manager._tools["financial_analysis"]
        schema = tool.parameters
        props = schema.get("properties", {})
        assert "question" in props


# ---------------------------------------------------------------------------
# 测试 3：Mock Pipeline 调用返回预期格式
# ---------------------------------------------------------------------------

class TestToolExecutionWithMock:
    """Mock 掉 pipeline，测试 Tool 调用逻辑。"""

    def test_financial_search_returns_results(self):
        from src.mcp_server import tools

        mock_pipeline = MagicMock()
        mock_pipeline._retriever.retrieve.return_value = [
            _make_retrieval_result("茅台2024年营收1680亿", 0.92, "annual_report.pdf"),
            _make_retrieval_result("五粮液2024年营收830亿", 0.85, "annual_report.pdf"),
        ]

        with patch.object(tools, "_get_pipeline", return_value=mock_pipeline):
            result = tools.financial_search("茅台营收", top_k=2)

        assert "茅台2024年营收1680亿" in result
        assert "0.920" in result
        assert "annual_report.pdf" in result

    def test_financial_search_empty_query(self):
        from src.mcp_server import tools

        result = tools.financial_search("")
        assert "不能为空" in result

    def test_financial_search_no_results(self):
        from src.mcp_server import tools

        mock_pipeline = MagicMock()
        mock_pipeline._retriever.retrieve.return_value = []

        with patch.object(tools, "_get_pipeline", return_value=mock_pipeline):
            result = tools.financial_search("不存在的查询")

        assert "未检索到" in result

    def test_knowledge_graph_query_returns_triples(self):
        from src.mcp_server import tools

        mock_gr = MagicMock()
        mock_gr.retrieve.return_value = [
            _make_triple("贵州茅台", "同比增长", "15%"),
            _make_triple("贵州茅台", "属于", "白酒行业"),
        ]

        with patch.object(tools, "_get_graph_retriever", return_value=mock_gr):
            result = tools.knowledge_graph_query("贵州茅台")

        assert "贵州茅台" in result
        assert "同比增长" in result
        assert "2 条" in result

    def test_knowledge_graph_query_with_relation_filter(self):
        from src.mcp_server import tools

        mock_gr = MagicMock()
        mock_gr.retrieve.return_value = [
            _make_triple("贵州茅台", "同比增长", "15%"),
            _make_triple("贵州茅台", "属于", "白酒行业"),
        ]

        with patch.object(tools, "_get_graph_retriever", return_value=mock_gr):
            result = tools.knowledge_graph_query("贵州茅台", relation="同比增长")

        assert "同比增长" in result
        assert "属于" not in result

    def test_knowledge_graph_query_graph_disabled(self):
        from src.mcp_server import tools

        with patch.object(tools, "_get_graph_retriever", return_value=None):
            result = tools.knowledge_graph_query("茅台")

        assert "未启用" in result

    def test_knowledge_graph_query_empty_entity(self):
        from src.mcp_server import tools

        result = tools.knowledge_graph_query("")
        assert "不能为空" in result

    def test_financial_analysis_returns_answer(self):
        from src.mcp_server import tools

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "茅台2024年ROE为30.2%。",
            "sources": ["annual_report.pdf"],
        }

        with patch.object(tools, "_get_pipeline", return_value=mock_pipeline):
            result = tools.financial_analysis("茅台ROE")

        assert "30.2%" in result
        assert "annual_report.pdf" in result

    def test_financial_analysis_empty_question(self):
        from src.mcp_server import tools

        result = tools.financial_analysis("")
        assert "不能为空" in result

    def test_financial_analysis_pipeline_error(self):
        from src.mcp_server import tools

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("LLM 调用失败")

        with patch.object(tools, "_get_pipeline", return_value=mock_pipeline):
            result = tools.financial_analysis("茅台营收")

        assert "分析失败" in result
