"""FactCacheStore 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import Fact


@pytest.fixture
def mock_embedder():
    """创建 mock SiliconFlowEmbedder。"""
    embedder = MagicMock(spec=SiliconFlowEmbedder)
    # 为每个fact生成不同方向的embedding（不是标量倍数）
    def mock_embed_texts(texts):
        embeddings = []
        for i in range(len(texts)):
            # 创建不同方向的向量
            vec = [0.0] * 10
            vec[i % 10] = 1.0  # 不同维度为1
            embeddings.append(vec)
        return embeddings
    embedder.embed_texts.side_effect = mock_embed_texts
    # 为query生成embedding - 与fact方向不同
    def mock_embed_query(text):
        return [0.0] * 10
    embedder.embed_query.side_effect = mock_embed_query
    return embedder


@pytest.fixture
def fact_cache(mock_embedder, tmp_path):
    """创建使用临时目录的 FactCacheStore。"""
    return FactCacheStore(
        embedder=mock_embedder,
        collection_name="test_fact_cache",
        persist_directory=str(tmp_path / "chroma_db"),
    )


@pytest.fixture
def sample_facts():
    """创建测试用 Fact 列表。"""
    return [
        Fact(
            topic="沪深300",
            fact="沪深300指数2024年市盈率为12.5倍，处于历史中位数水平。",
            category=["市场数据"],
            source="test_source",
        ),
        Fact(
            topic="贵州茅台",
            fact="贵州茅台2024年营业收入为1505.6亿元，同比增长16.1%。",
            category=["个股分析"],
            source="test_source",
        ),
    ]


def test_add_and_search(fact_cache, sample_facts):
    """添加fact后能检索到。"""
    fact_cache.add_facts(sample_facts)

    # 使用与fact相似的embedding进行搜索
    query_embedding = [0.1] * 10  # 与第一个fact的embedding相似
    results = fact_cache.search(query_embedding, threshold=0.0)

    assert len(results) > 0
    assert any(f.topic == "沪深300" for f in results)


def test_search_threshold(fact_cache, sample_facts):
    """低于阈值的结果被过滤。"""
    fact_cache.add_facts(sample_facts)

    # 使用完全不同的embedding，设置高阈值
    query_embedding = [0.9] * 10
    results = fact_cache.search(query_embedding, threshold=0.99)

    # 由于embedding差异大，高阈值应该过滤掉大部分结果
    assert len(results) == 0


def test_clear(fact_cache, sample_facts):
    """清空后检索为空。"""
    fact_cache.add_facts(sample_facts)
    fact_cache.clear()

    query_embedding = [0.1] * 10
    results = fact_cache.search(query_embedding, threshold=0.0)

    assert len(results) == 0


def test_stats(fact_cache, sample_facts):
    """统计信息正确。"""
    fact_cache.add_facts(sample_facts)

    stats = fact_cache.stats()

    assert stats["total_facts"] == 2
    assert "test_source" in stats["sources"]
    assert stats["sources"]["test_source"] == 2


def test_duplicate_facts(fact_cache, sample_facts):
    """重复添加不产生重复条目。"""
    fact_cache.add_facts(sample_facts)
    fact_cache.add_facts(sample_facts)  # 重复添加

    stats = fact_cache.stats()

    # 由于使用upsert和MD5去重，应该只有2条
    assert stats["total_facts"] == 2


def test_empty_facts(fact_cache):
    """添加空列表不报错。"""
    fact_cache.add_facts([])

    stats = fact_cache.stats()
    assert stats["total_facts"] == 0


def test_search_empty_cache(fact_cache):
    """空缓存搜索返回空列表。"""
    query_embedding = [0.1] * 10
    results = fact_cache.search(query_embedding, threshold=0.0)

    assert len(results) == 0


def test_stats_empty_cache(fact_cache):
    """空缓存统计信息正确。"""
    stats = fact_cache.stats()

    assert stats["total_facts"] == 0
    assert stats["sources"] == {}
