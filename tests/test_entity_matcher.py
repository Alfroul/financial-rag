"""EntityMatcher 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.entity_matcher import EntityMatcher, _cosine_similarity


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    return embedder


@pytest.fixture
def matcher(mock_embedder):
    return EntityMatcher(mock_embedder)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestBuildIndex:
    def test_build_index_populates(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[0.1], [0.2]]
        matcher.build_index(["茅台", "五粮液"])
        assert matcher._entities == ["茅台", "五粮液"]
        assert len(matcher._embeddings) == 2

    def test_build_index_empty(self, matcher):
        matcher.build_index([])
        assert matcher._entities == []
        assert matcher._embeddings == []

    def test_build_index_replaces(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[0.1], [0.2]]
        matcher.build_index(["A", "B"])
        mock_embedder.embed_texts.return_value = [[0.3]]
        matcher.build_index(["X"])
        assert matcher._entities == ["X"]
        assert len(matcher._embeddings) == 1


class TestMatch:
    def test_match_returns_best_above_threshold(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        matcher.build_index(["茅台", "五粮液"])

        result = matcher.match("贵州茅台", threshold=0.8)
        assert result == "茅台"

    def test_match_returns_none_below_threshold(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[0.0, 1.0, 0.0]]
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        matcher.build_index(["五粮液"])

        result = matcher.match("茅台", threshold=0.8)
        assert result is None

    def test_match_empty_index(self, matcher):
        matcher.build_index([])
        assert matcher.match("茅台") is None

    def test_match_default_threshold(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        matcher.build_index(["茅台"])

        assert matcher.match("茅台") == "茅台"


class TestMatchMany:
    def test_match_many_filters(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        mock_embedder.embed_query.side_effect = None
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        matcher.build_index(["茅台", "五粮液"])

        # query [1,0,0] 与 "茅台" [1,0,0] 匹配，与 "五粮液" [0,1,0] 不匹配
        results = matcher.match_many(["贵州茅台"], threshold=0.85)
        assert results == ["茅台"]

    def test_match_many_empty_input(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        matcher.build_index(["A"])
        assert matcher.match_many([]) == []

    def test_match_many_all_match(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        mock_embedder.embed_query.side_effect = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        matcher.build_index(["茅台", "五粮液"])

        results = matcher.match_many(["茅台", "五粮液"], threshold=0.8)
        assert results == ["茅台", "五粮液"]

    def test_match_many_none_match(self, mock_embedder, matcher):
        mock_embedder.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        mock_embedder.embed_query.return_value = [0.0, 1.0, 0.0]
        matcher.build_index(["茅台"])

        results = matcher.match_many(["不存在"], threshold=0.8)
        assert results == []
