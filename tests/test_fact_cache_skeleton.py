from src.config import Config, FactCacheConfig
from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import Fact, FactExtractor


def test_fact_dataclass():
    fact = Fact(topic="GDP", fact="2024年中国GDP增长5.0%", category=["宏观经济"], source="data/test.pdf")
    assert fact.topic == "GDP"
    assert fact.fact == "2024年中国GDP增长5.0%"
    assert fact.category == ["宏观经济"]
    assert fact.source == "data/test.pdf"


def test_fact_dataclass_defaults():
    fact = Fact(topic="CPI", fact="CPI同比上涨0.2%")
    assert fact.category == []
    assert fact.source == ""


def test_fact_extractor_interface():
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.chat.return_value = "[]"
    extractor = FactExtractor(llm=mock_llm)
    assert callable(getattr(extractor, "extract", None))
    result = extractor.extract("some context", "test.pdf")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == []  # facts
    assert result[1] == []  # triples


def test_fact_cache_store_interface(tmp_path):
    from unittest.mock import MagicMock

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = []
    store = FactCacheStore(
        embedder=mock_embedder,
        persist_directory=str(tmp_path / "chroma"),
    )
    assert callable(getattr(store, "add_facts", None))
    assert callable(getattr(store, "search", None))
    assert callable(getattr(store, "clear", None))
    assert callable(getattr(store, "stats", None))

    store.add_facts([])
    assert store.search([0.1, 0.2, 0.3]) == []
    store.clear()
    assert store.stats() == {"total_facts": 0, "sources": {}}


def test_fact_cache_config():
    config = FactCacheConfig()
    assert config.enabled is False
    assert config.collection_name == "fact_cache"
    assert config.similarity_threshold == 0.7
    assert config.max_facts == 10000


def test_fact_cache_config_from_yaml():
    config = Config()
    fc = config.fact_cache
    assert isinstance(fc, FactCacheConfig)
    assert fc.enabled is False
    assert fc.collection_name == "fact_cache"
    assert fc.similarity_threshold == 0.7
    assert fc.max_facts == 10000
