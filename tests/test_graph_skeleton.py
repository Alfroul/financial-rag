import pytest
import yaml

from src.config import Config, GraphConfig
from src.graph import GraphStore, NetworkxGraphStore, Triple


class TestTriple:
    def test_triple_creation(self):
        t = Triple(head="贵州茅台", relation="营收", tail="1680亿", source="test.pdf")
        assert t.head == "贵州茅台"
        assert t.relation == "营收"
        assert t.tail == "1680亿"
        assert t.source == "test.pdf"

    def test_triple_frozen(self):
        t = Triple(head="贵州茅台", relation="营收", tail="1680亿", source="test.pdf")
        with pytest.raises(AttributeError):
            t.head = "五粮液"  # type: ignore[misc]

    def test_triple_to_text(self):
        t = Triple(head="贵州茅台", relation="营收", tail="1680亿", source="test.pdf")
        assert t.to_text() == "贵州茅台 营收 1680亿"


class TestGraphStoreInterface:
    def test_graph_store_interface(self):
        assert hasattr(GraphStore, "add_triples")
        assert hasattr(GraphStore, "query_neighbors")
        assert hasattr(GraphStore, "query_path")
        assert hasattr(GraphStore, "get_entities")
        assert hasattr(GraphStore, "delete_by_source")
        assert hasattr(GraphStore, "clear")
        assert hasattr(GraphStore, "stats")
        assert hasattr(GraphStore, "save")
        assert hasattr(GraphStore, "load")

    def test_networkx_store_is_subclass(self):
        assert issubclass(NetworkxGraphStore, GraphStore)

    def test_networkx_store_init(self):
        store = NetworkxGraphStore()
        assert store._graph is not None


class TestGraphConfig:
    def test_graph_config(self):
        cfg = GraphConfig()
        assert cfg.enabled is False
        assert cfg.persist_path == "data/graph_store.pkl"
        assert cfg.max_neighbors == 50
        assert cfg.max_depth == 2

    def test_config_yaml_graph(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "graph": {
                        "enabled": True,
                        "persist_path": "custom/path.pkl",
                        "max_neighbors": 100,
                        "max_depth": 3,
                    }
                }
            ),
            encoding="utf-8",
        )
        config = Config.__new__(Config)
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        config._graph = Config._make(GraphConfig, raw.get("graph") or {})
        assert config.graph.enabled is True
        assert config.graph.persist_path == "custom/path.pkl"
        assert config.graph.max_neighbors == 100
        assert config.graph.max_depth == 3
