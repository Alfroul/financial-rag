
from src.config import GraphConfig
from src.graph.graph_store import NetworkxGraphStore
from src.graph.triple import Triple


def _make_pipeline_for_routing(graph_store=None, graph_config=None, graph_retriever=None):
    """Create a minimal RAGPipeline with mocked dependencies for routing tests."""
    from unittest.mock import MagicMock

    from src.rag_pipeline import RAGPipeline

    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_config = MagicMock()
    mock_config.max_context_tokens = 4000

    return RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=mock_config,
        graph_store=graph_store,
        graph_config=graph_config,
        graph_retriever=graph_retriever,
    )


class TestRouteQuery:
    def test_route_comparison(self):
        store = NetworkxGraphStore()
        store.add_triples([
            Triple("贵州茅台", "营收", "1680亿", "test"),
            Triple("五粮液", "营收", "800亿", "test"),
        ])
        pipeline = _make_pipeline_for_routing(graph_store=store)
        route = pipeline._route_query("贵州茅台和五粮液对比营收")
        assert route == "graph_comparison"

    def test_route_causal(self):
        store = NetworkxGraphStore()
        store.add_triples([Triple("贵州茅台", "营收", "1680亿", "test")])
        pipeline = _make_pipeline_for_routing(graph_store=store)
        route = pipeline._route_query("为什么贵州茅台涨了")
        assert route == "graph_causal"

    def test_route_rag(self):
        pipeline = _make_pipeline_for_routing()
        route = pipeline._route_query("GDP是什么")
        assert route == "rag"

    def test_route_comparison_needs_two_entities(self):
        store = NetworkxGraphStore()
        store.add_triples([Triple("贵州茅台", "营收", "1680亿", "test")])
        pipeline = _make_pipeline_for_routing(graph_store=store)
        route = pipeline._route_query("贵州茅台对比什么")
        assert route == "rag"

    def test_route_rag_when_graph_disabled(self):
        pipeline = _make_pipeline_for_routing()
        route = pipeline._route_query("贵州茅台和五粮液对比")
        assert route == "rag"


class TestTryGraphQuery:
    def test_returns_none_when_disabled(self):
        pipeline = _make_pipeline_for_routing()
        result = pipeline._try_graph_query("贵州茅台和五粮液对比")
        assert result is None

    def test_returns_none_when_no_retriever(self):
        config = GraphConfig(enabled=True)
        pipeline = _make_pipeline_for_routing(graph_config=config)
        result = pipeline._try_graph_query("贵州茅台和五粮液对比")
        assert result is None


class TestBuildGraphPrompt:
    def test_graph_prompt_format(self):
        triples = [
            Triple("贵州茅台", "营收", "1680亿", "test"),
            Triple("五粮液", "营收", "800亿", "test"),
        ]
        from src.rag_pipeline import RAGPipeline

        prompt = RAGPipeline._build_graph_prompt(triples, "some rag context")
        assert "[图谱知识]" in prompt
        assert "贵州茅台 营收 1680亿" in prompt
        assert "五粮液 营收 800亿" in prompt
        assert "[文档来源]" in prompt
        assert "some rag context" in prompt

    def test_graph_prompt_no_rag(self):
        triples = [Triple("A", "关系", "B", "test")]
        from src.rag_pipeline import RAGPipeline

        prompt = RAGPipeline._build_graph_prompt(triples, "")
        assert "[图谱知识]" in prompt
        assert "[文档来源]" not in prompt
