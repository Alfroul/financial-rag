from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.correction.rule_checker import RuleChecker
from src.fact_cache.sync import CacheSynchronizer, FileChange
from src.fact_extractor.extractor import Fact
from src.graph.graph_store import NetworkxGraphStore
from src.graph.triple import Triple

# ---------------------------------------------------------------------------
# RuleChecker + GraphStore
# ---------------------------------------------------------------------------


class TestRuleCheckerGraph:
    def test_entity_check_with_graph(self):
        """图中有实体时，即使不在 source 中也跳过报错。"""
        store = NetworkxGraphStore()
        store.add_triples([Triple("贵州茅台", "营收", "1680亿", "test")])

        checker = RuleChecker(
            financial_terms=["贵州茅台", "五粮液"],
            graph_store=store,
        )
        issues = checker.check(
            answer="贵州茅台营收1680亿，五粮液营收800亿",
            source_texts=["五粮液营收800亿"],
        )
        entity_issues = [i for i in issues if i["type"] == "entity"]
        entity_values = [i["value"] for i in entity_issues]
        assert "贵州茅台" not in entity_values
        assert "五粮液" not in entity_values

    def test_entity_check_without_graph(self):
        """graph_store=None 时完全跳过图验证，行为与原来一致。"""
        checker = RuleChecker(
            financial_terms=["贵州茅台"],
            graph_store=None,
        )
        issues = checker.check(
            answer="贵州茅台营收1680亿",
            source_texts=["其他内容"],
        )
        entity_issues = [i for i in issues if i["type"] == "entity"]
        assert len(entity_issues) == 1
        assert entity_issues[0]["value"] == "贵州茅台"

    def test_entity_not_in_graph_still_raises(self):
        """实体既不在 source 也不在 graph 中，仍然报错。"""
        store = NetworkxGraphStore()
        store.add_triples([Triple("五粮液", "营收", "800亿", "test")])

        checker = RuleChecker(
            financial_terms=["贵州茅台"],
            graph_store=store,
        )
        issues = checker.check(
            answer="贵州茅台营收1680亿",
            source_texts=["其他内容"],
        )
        entity_issues = [i for i in issues if i["type"] == "entity"]
        assert len(entity_issues) == 1
        assert entity_issues[0]["value"] == "贵州茅台"


# ---------------------------------------------------------------------------
# CacheSynchronizer + GraphStore
# ---------------------------------------------------------------------------


def _make_synchronizer(graph_store=None, tmp_path=None):
    """构造带 mock 依赖的 CacheSynchronizer。"""
    mock_cache = MagicMock()
    mock_cache.delete_by_source.return_value = 0
    mock_cache.add_facts.return_value = None

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = (
        [Fact(topic="test", fact="test fact", source="test.txt")],
        [Triple("贵州茅台", "营收", "1680亿", "test.txt")],
    )

    mock_watcher = MagicMock()
    mock_watcher.scan.return_value = []
    mock_watcher.save_hashes.return_value = None

    data_dir = str(tmp_path) if tmp_path else "data/raw"

    return CacheSynchronizer(
        cache=mock_cache,
        extractor=mock_extractor,
        watcher=mock_watcher,
        data_directory=data_dir,
        graph_store=graph_store,
    ), mock_cache, mock_extractor, mock_watcher


class TestGraphSyncAddedFile:
    def test_graph_sync_added_file(self, tmp_path):
        """新文件触发 Triple 提取并写入图谱。"""
        store = NetworkxGraphStore()
        sync, mock_cache, mock_extractor, mock_watcher = _make_synchronizer(
            graph_store=store, tmp_path=tmp_path,
        )
        mock_watcher.scan.return_value = [
            FileChange(path="new.txt", change_type="added"),
        ]

        with patch.object(sync, "_extract_file", return_value=(
            [Fact(topic="t", fact="f", source="new.txt")],
            [Triple("贵州茅台", "营收", "1680亿", "new.txt")],
        )):
            result = sync.sync()

        assert result.added == 1
        assert store.stats()["edges"] == 1
        assert store.stats()["nodes"] == 2


class TestGraphSyncModifiedFile:
    def test_graph_sync_modified_file(self, tmp_path):
        """修改文件触发删除旧 Triple + 重建。"""
        store = NetworkxGraphStore()
        store.add_triples([Triple("旧实体", "关系", "旧值", "mod.txt")])

        sync, mock_cache, mock_extractor, mock_watcher = _make_synchronizer(
            graph_store=store, tmp_path=tmp_path,
        )
        mock_watcher.scan.return_value = [
            FileChange(path="mod.txt", change_type="modified"),
        ]

        with patch.object(sync, "_should_update", return_value=True), \
             patch.object(sync, "_extract_file", return_value=(
                 [Fact(topic="t", fact="f", source="mod.txt")],
                 [Triple("新实体", "关系", "新值", "mod.txt")],
             )):
            result = sync.sync()

        assert result.modified == 1
        entities = store.get_entities()
        assert "旧实体" not in entities
        assert "新实体" in entities


class TestGraphSyncDeletedFile:
    def test_graph_sync_deleted_file(self, tmp_path):
        """删除文件触发 delete_by_source。"""
        store = NetworkxGraphStore()
        store.add_triples([Triple("A", "关系", "B", "del.txt")])

        sync, mock_cache, mock_extractor, mock_watcher = _make_synchronizer(
            graph_store=store, tmp_path=tmp_path,
        )
        mock_watcher.scan.return_value = [
            FileChange(path="del.txt", change_type="deleted"),
        ]

        result = sync.sync()

        assert result.deleted == 1
        assert store.stats()["edges"] == 0


class TestGraphSyncFailureIsolated:
    def test_graph_sync_failure_isolated(self, tmp_path):
        """Triple 同步失败不影响 FactCache 同步。"""
        store = MagicMock()
        store.add_triples.side_effect = RuntimeError("graph broken")

        sync, mock_cache, mock_extractor, mock_watcher = _make_synchronizer(
            graph_store=store, tmp_path=tmp_path,
        )
        mock_watcher.scan.return_value = [
            FileChange(path="new.txt", change_type="added"),
        ]

        with patch.object(sync, "_extract_file", return_value=(
            [Fact(topic="t", fact="f", source="new.txt")],
            [Triple("A", "关系", "B", "new.txt")],
        )):
            result = sync.sync()

        assert result.added == 1
        mock_cache.add_facts.assert_called_once()
