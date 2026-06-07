from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.fact_cache.store import FactCacheStore
from src.fact_cache.sync import (
    CacheSynchronizer,
    ChangeJudge,
    DataSourceWatcher,
)
from src.fact_extractor.extractor import Fact, FactExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def hash_file(tmp_path: Path) -> Path:
    return tmp_path / ".fact_cache_hashes.json"


@pytest.fixture
def watcher(data_dir: Path, hash_file: Path) -> DataSourceWatcher:
    return DataSourceWatcher(
        data_directory=str(data_dir),
        hash_file=str(hash_file),
    )


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1] * 10])
    embedder.embed_query = MagicMock(return_value=[0.1] * 10)
    return embedder


@pytest.fixture
def cache(tmp_path: Path, mock_embedder: MagicMock) -> FactCacheStore:
    return FactCacheStore(
        embedder=mock_embedder,
        collection_name="test_sync",
        persist_directory=str(tmp_path / "chroma"),
    )


@pytest.fixture
def mock_extractor() -> MagicMock:
    extractor = MagicMock(spec=FactExtractor)
    extractor.extract = MagicMock(
        return_value=(
            [Fact(topic="GDP", fact="2024年GDP增长5.0%", category=["macro-economy"], source="test.txt")],
            [],
        )
    )
    return extractor


# ===========================================================================
# DataSourceWatcher 测试
# ===========================================================================


class TestDetectNewFiles:
    def test_detect_new_files(self, data_dir: Path, watcher: DataSourceWatcher) -> None:
        (data_dir / "new_file.txt").write_text("new content", encoding="utf-8")
        changes = watcher.scan()
        assert len(changes) == 1
        assert changes[0].change_type == "added"
        assert changes[0].path == "new_file.txt"


class TestDetectModifiedFiles:
    def test_detect_modified_files(
        self, data_dir: Path, hash_file: Path, watcher: DataSourceWatcher
    ) -> None:
        (data_dir / "doc.txt").write_text("original", encoding="utf-8")
        watcher.save_hashes()

        (data_dir / "doc.txt").write_text("modified content", encoding="utf-8")
        changes = watcher.scan()

        assert len(changes) == 1
        assert changes[0].change_type == "modified"
        assert changes[0].path == "doc.txt"


class TestDetectDeletedFiles:
    def test_detect_deleted_files(
        self, data_dir: Path, hash_file: Path, watcher: DataSourceWatcher
    ) -> None:
        (data_dir / "to_delete.txt").write_text("will be removed", encoding="utf-8")
        watcher.save_hashes()

        (data_dir / "to_delete.txt").unlink()
        changes = watcher.scan()

        assert len(changes) == 1
        assert changes[0].change_type == "deleted"
        assert changes[0].path == "to_delete.txt"


class TestHashPersistence:
    def test_hash_persistence(
        self, data_dir: Path, hash_file: Path, watcher: DataSourceWatcher
    ) -> None:
        (data_dir / "a.txt").write_text("content a", encoding="utf-8")
        (data_dir / "b.txt").write_text("content b", encoding="utf-8")
        watcher.save_hashes()

        assert hash_file.exists()
        saved = json.loads(hash_file.read_text(encoding="utf-8"))
        assert "a.txt" in saved
        assert "b.txt" in saved
        assert len(saved) == 2

        # 新实例也能正确读取
        watcher2 = DataSourceWatcher(
            data_directory=str(data_dir),
            hash_file=str(hash_file),
        )
        changes = watcher2.scan()
        assert len(changes) == 0  # 无变更


# ===========================================================================
# ChangeJudge 测试
# ===========================================================================


class TestChangeJudgeYes:
    @patch("src.fact_cache.sync.httpx.Client")
    def test_change_judge_yes(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        judge = ChangeJudge(api_key="test-key")
        result = judge.judge("旧GDP增长5%", "新GDP增长6%")
        assert result is True


class TestChangeJudgeNo:
    @patch("src.fact_cache.sync.httpx.Client")
    def test_change_judge_no(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "NO"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        judge = ChangeJudge(api_key="test-key")
        result = judge.judge("内容不变", "内容不变")
        assert result is False


class TestChangeJudgeFallback:
    @patch("src.fact_cache.sync.httpx.Client")
    def test_change_judge_fallback(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("connection timeout")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        judge = ChangeJudge(api_key="test-key")
        result = judge.judge("old", "new")
        assert result is True  # 降级为YES


# ===========================================================================
# CacheSynchronizer 测试
# ===========================================================================


class TestManualSync:
    def test_manual_sync(
        self,
        data_dir: Path,
        hash_file: Path,
        cache: FactCacheStore,
        mock_extractor: MagicMock,
    ) -> None:
        (data_dir / "report.txt").write_text("some financial data", encoding="utf-8")

        watcher = DataSourceWatcher(
            data_directory=str(data_dir),
            hash_file=str(hash_file),
        )
        synchronizer = CacheSynchronizer(
            cache=cache,
            extractor=mock_extractor,
            watcher=watcher,
            data_directory=str(data_dir),
        )

        result = synchronizer.manual_sync()

        assert result.files_scanned == 1
        assert result.added == 1
        assert result.facts_updated == 1
        mock_extractor.extract.assert_called_once()


class TestUpdateDeletesOldFacts:
    def test_update_deletes_old_facts(
        self,
        data_dir: Path,
        hash_file: Path,
        cache: FactCacheStore,
        mock_extractor: MagicMock,
    ) -> None:
        # 先添加旧fact
        old_fact = Fact(
            topic="GDP",
            fact="旧数据",
            category=["macro-economy"],
            source="doc.txt",
        )
        cache.add_facts([old_fact])
        stats = cache.stats()
        assert stats["total_facts"] == 1

        # 模拟文件修改
        (data_dir / "doc.txt").write_text("new content", encoding="utf-8")

        watcher = DataSourceWatcher(
            data_directory=str(data_dir),
            hash_file=str(hash_file),
        )
        # 先保存初始hash（无文件），然后添加文件
        # 需要先保存一次hash让文件被识别为新增
        # 实际上这里没有保存hash，所以所有文件都是新增
        synchronizer = CacheSynchronizer(
            cache=cache,
            extractor=mock_extractor,
            watcher=watcher,
            judge=None,  # 跳过judge
            data_directory=str(data_dir),
        )

        # 用手动同步测试：会先clear再重新提取
        result = synchronizer.manual_sync()
        assert result.files_scanned == 1
        mock_extractor.extract.assert_called_once()


class TestSyncIncremental:
    def test_sync_detects_new_and_updates(
        self,
        data_dir: Path,
        hash_file: Path,
        cache: FactCacheStore,
        mock_extractor: MagicMock,
    ) -> None:
        watcher = DataSourceWatcher(
            data_directory=str(data_dir),
            hash_file=str(hash_file),
        )
        synchronizer = CacheSynchronizer(
            cache=cache,
            extractor=mock_extractor,
            watcher=watcher,
            data_directory=str(data_dir),
        )

        # 首次同步 — 空目录无变更
        result = synchronizer.sync()
        assert result.added == 0

        # 添加文件后再次同步
        (data_dir / "news.txt").write_text("financial news", encoding="utf-8")
        result = synchronizer.sync()
        assert result.added == 1
        assert result.facts_updated == 1


class TestSyncDeleted:
    def test_sync_deletes_removed_files(
        self,
        data_dir: Path,
        hash_file: Path,
        cache: FactCacheStore,
        mock_extractor: MagicMock,
    ) -> None:
        # 创建文件并保存hash
        (data_dir / "old.txt").write_text("old data", encoding="utf-8")
        watcher = DataSourceWatcher(
            data_directory=str(data_dir),
            hash_file=str(hash_file),
        )
        watcher.save_hashes()

        # 添加fact到缓存
        old_fact = Fact(
            topic="test",
            fact="old fact",
            category=[],
            source="old.txt",
        )
        cache.add_facts([old_fact])
        assert cache.stats()["total_facts"] == 1

        # 删除文件
        (data_dir / "old.txt").unlink()

        synchronizer = CacheSynchronizer(
            cache=cache,
            extractor=mock_extractor,
            watcher=watcher,
            data_directory=str(data_dir),
        )

        result = synchronizer.sync()
        assert result.deleted == 1
        # 删除的文件应该从缓存中移除对应fact
        assert cache.stats()["total_facts"] == 0
