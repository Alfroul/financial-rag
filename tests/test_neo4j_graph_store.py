"""Neo4jGraphStore 单元测试 — 使用 mock driver。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from src.graph.graph_store import Neo4jGraphStore, create_graph_store
from src.graph.triple import Triple

# ---------------------------------------------------------------------------
# Helpers: fake neo4j session/driver
# ---------------------------------------------------------------------------

class FakeRecord:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class FakeResult:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = [FakeRecord(r) for r in records]
        self._idx = 0

    def __iter__(self):
        return iter(self._records)

    def single(self) -> FakeRecord | None:
        if not self._records:
            return None
        return self._records[0]


class FakeSession:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def run(self, query: str, **kwargs) -> FakeResult:
        self.queries.append((query, kwargs))
        return FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self):
        self._session = FakeSession()

    def session(self):
        return self._session

    def close(self):
        pass


def _make_store() -> tuple[Neo4jGraphStore, FakeSession]:
    fake_driver = FakeDriver()
    with patch("src.graph.graph_store.Neo4jGraphStore._ensure_index"):
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store._driver = fake_driver
    return store, fake_driver._session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNeo4jGraphStore:
    def test_add_triples_returns_count(self):
        store, session = _make_store()
        triples = [
            Triple(head="贵州茅台", relation="营收", tail="1680亿", source="test.md"),
            Triple(head="五粮液", relation="营收", tail="830亿", source="test.md"),
        ]
        # Mock the session.run to return a count
        def fake_run(query, **kwargs):
            session.queries.append((query, kwargs))
            if "UNWIND" in query:
                return FakeResult([{"created": 2}])
            return FakeResult([])

        session.run = fake_run
        result = store.add_triples(triples)
        assert result == 2

    def test_add_triples_empty_list(self):
        store, _ = _make_store()
        assert store.add_triples([]) == 0

    def test_query_neighbors_returns_triples(self):
        store, session = _make_store()

        records = [
            {"head": "贵州茅台", "relation": "营收", "tail": "1680亿", "source": "test.md"},
            {"head": "贵州茅台", "relation": "属于", "tail": "白酒", "source": "test.md"},
        ]

        def fake_run(query, **kwargs):
            return FakeResult(records)

        session.run = fake_run
        result = store.query_neighbors("贵州茅台", max_depth=1)
        assert len(result) == 2
        assert result[0].head == "贵州茅台"
        assert result[0].relation == "营收"

    def test_query_path_returns_path(self):
        store, session = _make_store()

        path_data = [
            {"head": "贵州茅台", "relation": "属于", "tail": "白酒", "source": "test.md"},
        ]

        def fake_run(query, **kwargs):
            return FakeResult([{"triples": path_data}])

        session.run = fake_run
        result = store.query_path("贵州茅台", "白酒")
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].head == "贵州茅台"

    def test_query_path_no_path(self):
        store, session = _make_store()

        def fake_run(query, **kwargs):
            return FakeResult([])

        session.run = fake_run
        result = store.query_path("贵州茅台", "不存在的实体")
        assert result == []

    def test_get_entities(self):
        store, session = _make_store()

        def fake_run(query, **kwargs):
            return FakeResult([{"name": "贵州茅台"}, {"name": "五粮液"}])

        session.run = fake_run
        entities = store.get_entities()
        assert "贵州茅台" in entities
        assert "五粮液" in entities

    def test_delete_by_source(self):
        store, session = _make_store()
        call_count = 0

        def fake_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeResult([{"deleted": 3}])
            return FakeResult([])

        session.run = fake_run
        deleted = store.delete_by_source("test.md")
        assert deleted == 3

    def test_clear(self):
        store, session = _make_store()
        store.clear()
        assert any("DETACH DELETE" in q for q, _ in session.queries)

    def test_stats(self):
        store, session = _make_store()
        call_count = 0

        def fake_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeResult([{"c": 10}])
            elif call_count == 2:
                return FakeResult([{"c": 20}])
            else:
                return FakeResult([{"c": 3}])

        session.run = fake_run
        stats = store.stats()
        assert stats == {"nodes": 10, "edges": 20, "sources": 3}

    def test_save_no_op(self):
        store, _ = _make_store()
        store.save("/tmp/test.pkl")  # should not raise

    def test_load_no_op(self):
        store, _ = _make_store()
        store.load("/tmp/test.pkl")  # should not raise


class TestCreateGraphStoreFactory:
    def test_default_returns_networkx(self):
        @dataclass
        class FakeGraphConfig:
            backend: str = "networkx"
            persist_path: str = "nonexistent.pkl"
            neo4j_uri: str = "bolt://localhost:7687"
            neo4j_user: str = "neo4j"
            neo4j_password_env: str = "NEO4J_PASSWORD"
            max_neighbors: int = 50
            max_depth: int = 2

        from src.graph.graph_store import NetworkxGraphStore

        store = create_graph_store(FakeGraphConfig())
        assert isinstance(store, NetworkxGraphStore)

    def test_neo4j_fallback_without_password(self):
        @dataclass
        class FakeGraphConfig:
            backend: str = "neo4j"
            persist_path: str = "nonexistent.pkl"
            neo4j_uri: str = "bolt://localhost:7687"
            neo4j_user: str = "neo4j"
            neo4j_password_env: str = "NONEXISTENT_VAR"
            max_neighbors: int = 50
            max_depth: int = 2

        from src.graph.graph_store import NetworkxGraphStore

        store = create_graph_store(FakeGraphConfig())
        assert isinstance(store, NetworkxGraphStore)

    def test_neo4j_backend_success(self):
        @dataclass
        class FakeGraphConfig:
            backend: str = "neo4j"
            persist_path: str = "nonexistent.pkl"
            neo4j_uri: str = "bolt://localhost:7687"
            neo4j_user: str = "neo4j"
            neo4j_password_env: str = "NEO4J_PASSWORD"
            max_neighbors: int = 50
            max_depth: int = 2

        import sys
        import types

        # Create a fake neo4j module
        fake_neo4j = types.ModuleType("neo4j")
        fake_neo4j.GraphDatabase = MagicMock()
        sys.modules["neo4j"] = fake_neo4j

        try:
            with patch.dict("os.environ", {"NEO4J_PASSWORD": "testpass"}):
                store = create_graph_store(FakeGraphConfig())
                assert isinstance(store, Neo4jGraphStore)
        finally:
            del sys.modules["neo4j"]

    def test_neo4j_connection_failure_fallback(self):
        @dataclass
        class FakeGraphConfig:
            backend: str = "neo4j"
            persist_path: str = "nonexistent.pkl"
            neo4j_uri: str = "bolt://localhost:7687"
            neo4j_user: str = "neo4j"
            neo4j_password_env: str = "NEO4J_PASSWORD"
            max_neighbors: int = 50
            max_depth: int = 2

        import sys
        import types

        from src.graph.graph_store import NetworkxGraphStore

        fake_neo4j = types.ModuleType("neo4j")
        fake_neo4j.GraphDatabase = MagicMock()
        fake_neo4j.GraphDatabase.driver.side_effect = Exception("Connection refused")
        sys.modules["neo4j"] = fake_neo4j

        try:
            with patch.dict("os.environ", {"NEO4J_PASSWORD": "testpass"}):
                store = create_graph_store(FakeGraphConfig())
                assert isinstance(store, NetworkxGraphStore)
        finally:
            del sys.modules["neo4j"]
