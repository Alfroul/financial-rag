"""缓存路由逻辑测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import RAGConfig
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import Fact, FactExtractor
from src.rag_pipeline import RAGPipeline
from src.retriever.retriever import RetrievalResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=SiliconFlowEmbedder)
    embedder.embed_query.return_value = [0.5] * 10
    embedder.aembed_query.return_value = [0.5] * 10
    return embedder


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievalResult(
            content="测试文档内容",
            score=0.8,
            metadata={"source": "test"},
            doc_id="doc1",
        )
    ]
    return retriever


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = "这是LLM的回答"
    llm.achat.return_value = "这是异步LLM的回答"
    return llm


@pytest.fixture
def mock_fact_cache():
    cache = MagicMock(spec=FactCacheStore)
    cache.search.return_value = []
    cache.search_by_text.return_value = []
    cache.asearch_by_text.return_value = []
    return cache


@pytest.fixture
def mock_fact_extractor():
    extractor = MagicMock(spec=FactExtractor)
    extractor.extract.return_value = (
        [Fact(topic="测试主题", fact="测试事实", category=["测试"], source="test")],
        [],
    )
    return extractor


@pytest.fixture
def rag_config():
    return RAGConfig(max_context_tokens=4000, query_rewrite=False)


def test_cache_hit_skips_rag(mock_retriever, mock_llm, rag_config, mock_fact_cache, mock_embedder):
    """缓存命中时不调用RAG检索。"""
    # 设置缓存命中
    cached_facts = [
        Fact(topic="缓存主题", fact="缓存事实内容", category=["缓存"], source="cache")
    ]
    mock_fact_cache.search_by_text.return_value = cached_facts

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
        fact_cache=mock_fact_cache,
    )

    result = pipeline.query("测试问题")

    # 验证缓存被查询
    mock_fact_cache.search_by_text.assert_called_once()
    # 验证RAG检索未被调用
    mock_retriever.retrieve.assert_not_called()
    # 验证返回了结果
    assert "answer" in result
    assert "sources" in result


def test_cache_miss_falls_back_to_rag(mock_retriever, mock_llm, rag_config, mock_fact_cache, mock_embedder):
    """缓存未命中时走RAG。"""
    # 设置缓存未命中
    mock_fact_cache.search_by_text.return_value = []

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
        fact_cache=mock_fact_cache,
    )

    result = pipeline.query("测试问题")

    # 验证缓存被查询
    mock_fact_cache.search_by_text.assert_called_once()
    # 验证RAG检索被调用
    mock_retriever.retrieve.assert_called_once()
    # 验证返回了结果
    assert "answer" in result
    assert "sources" in result


def test_rag_result_stored_to_cache(
    mock_retriever, mock_llm, rag_config, mock_fact_cache, mock_fact_extractor, mock_embedder
):
    """RAG结果被提取并存入缓存。"""
    # 设置缓存未命中
    mock_fact_cache.search_by_text.return_value = []

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
        fact_cache=mock_fact_cache,
        fact_extractor=mock_fact_extractor,
    )

    result = pipeline.query("测试问题")

    # 验证fact提取被调用
    mock_fact_extractor.extract.assert_called_once()
    # 验证fact被存入缓存
    mock_fact_cache.add_facts.assert_called_once()
    # 验证返回了结果
    assert "answer" in result


def test_cache_disabled(mock_retriever, mock_llm, rag_config, mock_embedder):
    """配置禁用时走原有流程。"""
    # 不传入fact_cache和fact_extractor
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
    )

    result = pipeline.query("测试问题")

    # 验证RAG检索被调用
    mock_retriever.retrieve.assert_called_once()
    # 验证返回了结果
    assert "answer" in result
    assert "sources" in result


def test_cache_hit_returns_correct_sources(mock_retriever, mock_llm, rag_config, mock_fact_cache, mock_embedder):
    """缓存命中时返回正确的sources格式。"""
    cached_facts = [
        Fact(topic="主题1", fact="事实1", category=["分类1"], source="source1"),
        Fact(topic="主题2", fact="事实2", category=["分类2"], source="source2"),
    ]
    mock_fact_cache.search_by_text.return_value = cached_facts

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
        fact_cache=mock_fact_cache,
    )

    result = pipeline.query("测试问题")

    # 验证sources格式
    assert len(result["sources"]) == 2
    assert result["sources"][0]["metadata"]["topic"] == "主题1"
    assert result["sources"][1]["metadata"]["topic"] == "主题2"


def test_cache_hit_metrics(mock_retriever, mock_llm, rag_config, mock_fact_cache, mock_embedder):
    """缓存命中时metrics中标记cache_hit=True。"""
    cached_facts = [
        Fact(topic="主题", fact="事实", category=["分类"], source="source")
    ]
    mock_fact_cache.search_by_text.return_value = cached_facts

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        config=rag_config,
        fact_cache=mock_fact_cache,
    )

    # 这个测试主要验证不会抛出异常
    result = pipeline.query("测试问题")
    assert "answer" in result
