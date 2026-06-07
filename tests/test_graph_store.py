import tempfile

import pytest

from src.graph.graph_store import NetworkxGraphStore
from src.graph.triple import Triple


@pytest.fixture
def store():
    return NetworkxGraphStore()


@pytest.fixture
def sample_triples():
    return [
        Triple("贵州茅台", "营收", "1680亿", "test"),
        Triple("贵州茅台", "属于", "白酒行业", "test"),
        Triple("五粮液", "营收", "800亿", "test"),
        Triple("五粮液", "属于", "白酒行业", "test"),
        Triple("白酒行业", "属于", "消费品", "test"),
    ]


class TestAddTriples:
    def test_add_triples(self, store, sample_triples):
        count = store.add_triples(sample_triples)
        assert count == 5
        assert store._graph.number_of_nodes() == 6
        assert store._graph.number_of_edges() == 5

    def test_add_triples_dedup(self, store, sample_triples):
        store.add_triples(sample_triples)
        count = store.add_triples(sample_triples)
        assert count == 0
        assert store._graph.number_of_edges() == 5

    def test_add_triples_different_relation(self, store):
        store.add_triples([Triple("A", "关系1", "B", "test")])
        count = store.add_triples([Triple("A", "关系2", "B", "test")])
        assert count == 1
        assert store._graph.number_of_edges() == 2

    def test_node_attributes(self, store, sample_triples):
        store.add_triples(sample_triples)
        assert store._graph.nodes["贵州茅台"]["type"] == "entity"
        assert store._graph.nodes["消费品"]["type"] == "entity"

    def test_edge_attributes(self, store, sample_triples):
        store.add_triples(sample_triples)
        edge_data = store._graph["贵州茅台"]["白酒行业"]
        first_edge = list(edge_data.values())[0]
        assert "relation" in first_edge


class TestQueryNeighbors:
    def test_query_neighbors(self, store, sample_triples):
        store.add_triples(sample_triples)
        triples = store.query_neighbors("贵州茅台")
        assert len(triples) >= 1
        entities = {t.head for t in triples} | {t.tail for t in triples}
        assert "贵州茅台" in entities

    def test_query_neighbors_depth(self, store, sample_triples):
        store.add_triples(sample_triples)
        depth1 = store.query_neighbors("贵州茅台", max_depth=1)
        depth2 = store.query_neighbors("贵州茅台", max_depth=2)
        assert len(depth2) >= len(depth1)

    def test_query_neighbors_nonexistent(self, store):
        triples = store.query_neighbors("不存在的实体")
        assert triples == []


class TestQueryPath:
    def test_query_path(self, store, sample_triples):
        store.add_triples(sample_triples)
        paths = store.query_path("贵州茅台", "五粮液")
        assert len(paths) >= 1

    def test_query_path_no_connection(self, store):
        store.add_triples([Triple("A", "关系", "B", "test"), Triple("C", "关系", "D", "test")])
        paths = store.query_path("A", "C")
        assert paths == []

    def test_query_path_nonexistent(self, store):
        paths = store.query_path("不存在A", "不存在B")
        assert paths == []


class TestDeleteBySource:
    def test_delete_by_source(self, store):
        store.add_triples([
            Triple("A", "关系", "B", "source1"),
            Triple("C", "关系", "D", "source2"),
        ])
        count = store.delete_by_source("source1")
        assert count == 1
        assert store._graph.number_of_edges() == 1

    def test_delete_by_source_removes_isolates(self, store):
        store.add_triples([Triple("A", "关系", "B", "source1")])
        store.delete_by_source("source1")
        assert store._graph.number_of_nodes() == 0


class TestStats:
    def test_stats(self, store, sample_triples):
        store.add_triples(sample_triples)
        s = store.stats()
        assert s["nodes"] == 6
        assert s["edges"] == 5
        assert s["sources"] == 1


class TestSaveLoad:
    def test_save_load(self, store, sample_triples):
        store.add_triples(sample_triples)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        store.save(path)

        new_store = NetworkxGraphStore()
        new_store.load(path)
        assert new_store.stats()["nodes"] == store.stats()["nodes"]
        assert new_store.stats()["edges"] == store.stats()["edges"]

    def test_load_nonexistent(self, store):
        store.load("/tmp/nonexistent_graph.pkl")
        assert store.stats()["nodes"] == 0


class TestClear:
    def test_clear(self, store, sample_triples):
        store.add_triples(sample_triples)
        store.clear()
        assert store.stats()["nodes"] == 0
        assert store.stats()["edges"] == 0
