from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import Fact, FactExtractor
from src.graph.triple import Triple

if TYPE_CHECKING:
    from src.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """表示一个文件变更。"""

    path: str
    change_type: str  # "added" | "modified" | "deleted"


@dataclass
class SyncResult:
    """同步结果。"""

    files_scanned: int = 0
    added: int = 0
    modified: int = 0
    deleted: int = 0
    facts_updated: int = 0
    skipped: int = 0


class DataSourceWatcher:
    """扫描数据目录，检测文件变更。"""

    def __init__(
        self,
        data_directory: str = "data/raw",
        hash_file: str = "data/.fact_cache_hashes.json",
    ) -> None:
        self._data_directory = Path(data_directory)
        self._hash_file = Path(hash_file)
        self._cached_saved: dict[str, dict[str, str]] | None = None

    def scan(self) -> list[FileChange]:
        """扫描目录，与上次hash对比，返回变更列表。"""
        current_data = self._compute_hashes()
        saved_data = self._load_hashes()
        self._cached_saved = saved_data

        changes: list[FileChange] = []

        for path, entry in current_data.items():
            if path not in saved_data:
                changes.append(FileChange(path=path, change_type="added"))
            elif saved_data[path]["hash"] != entry["hash"]:
                changes.append(FileChange(path=path, change_type="modified"))

        for path in saved_data:
            if path not in current_data:
                changes.append(FileChange(path=path, change_type="deleted"))

        return changes

    def save_hashes(self) -> None:
        """将当前文件hash和内容保存到磁盘。"""
        data = self._compute_hashes()
        self._hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._hash_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._cached_saved = None
        logger.info("DataSourceWatcher: saved %d file hashes", len(data))

    def get_old_content(self, rel_path: str) -> str | None:
        """从上次保存的记录中获取文件旧内容（使用缓存避免重复读盘）。"""
        saved = self._cached_saved if self._cached_saved is not None else self._load_hashes()
        entry = saved.get(rel_path)
        if isinstance(entry, dict):
            return entry.get("content")
        return None

    def _compute_hashes(self) -> dict[str, dict[str, str]]:
        """计算数据目录下所有文件的SHA256 hash和内容。"""
        result: dict[str, dict[str, str]] = {}
        if not self._data_directory.exists():
            return result
        for file_path in sorted(self._data_directory.rglob("*")):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(self._data_directory))
                content = self._read_file(file_path)
                result[rel_path] = {"hash": self._sha256(file_path), "content": content}
        return result

    def _load_hashes(self) -> dict[str, dict[str, str]]:
        """从磁盘加载上次保存的hash和内容记录。兼容旧格式（值为纯hash字符串）。"""
        if not self._hash_file.exists():
            return {}
        try:
            with open(self._hash_file, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            # 兼容旧格式：值为纯字符串hash
            result: dict[str, dict[str, str]] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    result[k] = v
                elif isinstance(v, str):
                    result[k] = {"hash": v, "content": ""}
            return result
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("DataSourceWatcher: failed to load hashes: %s", e)
        return {}

    @staticmethod
    def _read_file(file_path: Path) -> str:
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            return ""

    @staticmethod
    def _sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class ChangeJudge:
    """通过本地小模型判断文件变更是否为实质性知识变化。"""

    def __init__(
        self,
        judge_model: str = "mimo-v2-pro",
        judge_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        api_key: str | None = None,
    ) -> None:
        self._model = judge_model
        self._base_url = judge_base_url.rstrip("/")
        self._api_key = api_key

    def judge(self, old_content: str, new_content: str, max_retries: int = 2) -> bool:
        """判断两段内容是否有实质性知识变化。返回True表示需要更新。"""
        prompt = (
            "你是一个金融数据变更检测专家。请判断以下两段内容之间是否有实质性知识变化。\n"
            "实质性知识变化包括：数据更新、新增信息、删除重要信息、事实修正。\n"
            "非实质性变化包括：格式调整、标点修改、换行变化、空格调整。\n\n"
            f"【旧内容】\n{old_content[:2000]}\n\n"
            f"【新内容】\n{new_content[:2000]}\n\n"
            "这两段内容是否有实质性知识变化？只回答YES或NO。"
        )

        for attempt in range(max_retries + 1):
            try:
                response = self._call_llm(prompt)
                answer = response.strip().upper()
                return "YES" in answer
            except Exception as e:
                if attempt < max_retries:
                    logger.info("ChangeJudge: retry %d/%d after failure: %s", attempt + 1, max_retries, e)
                    continue
                logger.warning("ChangeJudge: LLM call failed after %d retries, fallback to YES: %s", max_retries, e)
                return True
        return True  # fallback: assume change if loop exhausts

    def _call_llm(self, prompt: str) -> str:
        """调用OpenAI兼容API。"""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return str(message.get("content", ""))
        return ""


class CacheSynchronizer:
    """缓存同步器，协调变更检测、变更判断和缓存更新。"""

    def __init__(
        self,
        cache: FactCacheStore,
        extractor: FactExtractor,
        watcher: DataSourceWatcher,
        judge: ChangeJudge | None = None,
        data_directory: str = "data/raw",
        judge_api_key: str | None = None,
        judge_model: str = "mimo-v2-pro",
        judge_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        graph_store: GraphStore | None = None,
        graph_persist_path: str = "data/graph_store.pkl",
    ) -> None:
        self._cache = cache
        self._extractor = extractor
        self._watcher = watcher
        self._judge = judge
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._data_directory = Path(data_directory)
        self._last_sync: SyncResult | None = None
        self._graph_store = graph_store
        self._graph_persist_path = graph_persist_path

    @property
    def last_sync(self) -> SyncResult | None:
        return self._last_sync

    def sync(self) -> SyncResult:
        """增量同步：检测变更并更新缓存。"""
        result = SyncResult()
        changes = self._watcher.scan()

        for change in changes:
            result.files_scanned += 1

            if change.change_type == "deleted":
                deleted = self._cache.delete_by_source(change.path)
                self._sync_triples_delete(change.path)
                result.deleted += 1
                result.facts_updated += deleted
                logger.info("Sync: deleted %d facts for %s", deleted, change.path)

            elif change.change_type == "added":
                facts, triples = self._extract_file(change.path)
                if facts:
                    self._cache.add_facts(facts)
                    result.facts_updated += len(facts)
                self._sync_triples_add(triples, change.path)
                result.added += 1
                logger.info("Sync: added %d facts for %s", len(facts), change.path)

            elif change.change_type == "modified":
                if self._should_update(change.path):
                    self._cache.delete_by_source(change.path)
                    self._sync_triples_delete(change.path)
                    facts, triples = self._extract_file(change.path)
                    if facts:
                        self._cache.add_facts(facts)
                    self._sync_triples_add(triples, change.path)
                    result.facts_updated += len(facts)
                    result.modified += 1
                    logger.info("Sync: updated %d facts for %s", len(facts), change.path)
                else:
                    result.skipped += 1
                    logger.info("Sync: skipped %s (no substantive change)", change.path)

        self._watcher.save_hashes()
        self._save_graph()
        self._last_sync = result
        return result

    def manual_sync(self) -> SyncResult:
        """人工触发全量重建缓存。先提取再原子替换，避免中途失败丢失旧数据。"""
        result = SyncResult()
        all_facts: list[tuple[str, list]] = []
        all_triples: list[tuple[str, list[Triple]]] = []

        for file_path in sorted(self._data_directory.rglob("*")):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(self._data_directory))
                facts, triples = self._extract_file(rel_path)
                all_facts.append((rel_path, facts))
                all_triples.append((rel_path, triples))
                result.files_scanned += 1

        self._cache.clear()
        self._clear_graph()
        logger.info("ManualSync: cache cleared, adding %d files", result.files_scanned)

        for rel_path, facts in all_facts:
            if facts:
                self._cache.add_facts(facts)
                result.facts_updated += len(facts)
            result.added += 1

        for _rel_path, triples in all_triples:
            self._sync_triples_add(triples, _rel_path)

        self._watcher.save_hashes()
        self._save_graph()
        self._last_sync = result
        return result

    def _should_update(self, rel_path: str) -> bool:
        """判断修改的文件是否需要更新缓存。"""
        judge = self._ensure_judge()
        if judge is None:
            return True

        file_path = self._data_directory / rel_path
        if not file_path.exists():
            return True

        new_content = DataSourceWatcher._read_file(file_path)
        old_content = self._watcher.get_old_content(rel_path)

        if not old_content:
            return True

        return judge.judge(old_content, new_content)

    def _ensure_judge(self) -> ChangeJudge | None:
        """延迟初始化ChangeJudge（需要api_key时自动创建）。"""
        if self._judge is not None:
            return self._judge
        if self._judge_api_key:
            self._judge = ChangeJudge(
                judge_model=self._judge_model,
                judge_base_url=self._judge_base_url,
                api_key=self._judge_api_key,
            )
            return self._judge
        return None

    def _extract_file(self, rel_path: str) -> tuple[list[Fact], list[Triple]]:
        """对单个文件提取 fact 和 triple。"""
        file_path = self._data_directory / rel_path
        if not file_path.exists():
            return [], []
        try:
            content = DataSourceWatcher._read_file(file_path)
            if not content.strip():
                return [], []
            facts, triples = self._extractor.extract(content, source=rel_path)
            return facts, triples
        except Exception as e:
            logger.error("Sync: failed to extract facts from %s: %s", rel_path, e)
            return [], []

    def _sync_triples_add(self, triples: list[Triple], source: str) -> None:
        """将提取的 Triple 写入图谱。失败不影响 FactCache。"""
        if self._graph_store is None or not triples:
            return
        try:
            added = self._graph_store.add_triples(triples)
            logger.info("Sync: added %d triples for %s", added, source)
        except Exception as e:
            logger.warning("Sync: failed to add triples for %s: %s", source, e)

    def _sync_triples_delete(self, source: str) -> None:
        """删除指定来源的 Triple。失败不影响 FactCache。"""
        if self._graph_store is None:
            return
        try:
            deleted = self._graph_store.delete_by_source(source)
            logger.info("Sync: deleted %d triples for %s", deleted, source)
        except Exception as e:
            logger.warning("Sync: failed to delete triples for %s: %s", source, e)

    def _clear_graph(self) -> None:
        """清空图谱（全量重建前调用）。"""
        if self._graph_store is None:
            return
        try:
            self._graph_store.clear()
        except Exception as e:
            logger.warning("Sync: failed to clear graph: %s", e)

    def _save_graph(self) -> None:
        """持久化图谱。"""
        if self._graph_store is None:
            return
        try:
            self._graph_store.save(self._graph_persist_path)
        except Exception as e:
            logger.warning("Sync: failed to save graph: %s", e)
