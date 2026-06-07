from __future__ import annotations

import contextlib
import hashlib
import logging

import chromadb

from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.fact_extractor.extractor import Fact

logger = logging.getLogger(__name__)


class FactCacheStore:
    """基于ChromaDB的知识缓存存储层。"""

    def __init__(
        self,
        embedder: SiliconFlowEmbedder,
        collection_name: str = "fact_cache",
        persist_directory: str = "data/chroma_db",
    ) -> None:
        self._embedder = embedder
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_facts(self, facts: list[Fact]) -> None:
        if not facts:
            return

        texts = [f"{f.topic} {f.fact}" for f in facts]
        embeddings = self._embedder.embed_texts(texts)

        ids = []
        documents = []
        metadatas = []
        used_embeddings = []

        for fact, embedding in zip(facts, embeddings, strict=True):
            doc_id = self._make_id(fact)
            ids.append(doc_id)
            documents.append(fact.fact)
            used_embeddings.append(embedding)
            metadatas.append(
                {
                    "topic": fact.topic,
                    "category": ",".join(fact.category),
                    "source": fact.source,
                }
            )

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=used_embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        logger.info("FactCacheStore: upserted %d facts", len(ids))

    def search_by_text(self, query: str, threshold: float = 0.7) -> list[Fact]:
        """同步：对查询文本做embedding后检索相似fact。"""
        embedding = self._embedder.embed_query(query)
        return self.search(embedding, threshold=threshold)

    async def asearch_by_text(self, query: str, threshold: float = 0.7) -> list[Fact]:
        """异步：对查询文本做embedding后检索相似fact。"""
        embedding = await self._embedder.aembed_query(query)
        return self.search(embedding, threshold=threshold)

    def search(self, query_embedding: list[float], threshold: float = 0.7) -> list[Fact]:
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=min(count, 20),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        facts: list[Fact] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{} for _ in docs]
        dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists, strict=True):
            similarity = 1.0 - dist
            if similarity < threshold:
                continue
            topic = str(meta.get("topic", "")) if meta else ""
            category_str = str(meta.get("category", "")) if meta else ""
            source = str(meta.get("source", "")) if meta else ""
            category = [c.strip() for c in category_str.split(",") if c.strip()]
            facts.append(Fact(topic=topic, fact=doc or "", category=category, source=source))

        return facts

    def delete_by_source(self, source: str) -> int:
        """删除指定source的所有fact，返回删除数量。"""
        result = self._collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info("FactCacheStore: deleted %d facts for source=%s", len(ids), source)
        return len(ids)

    def clear(self) -> None:
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict:
        count = self._collection.count()
        if count == 0:
            return {"total_facts": 0, "sources": {}}

        result = self._collection.get(include=["metadatas"])
        metas = result.get("metadatas") or []
        sources: dict[str, int] = {}
        for meta in metas:
            src = str((meta or {}).get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1

        return {"total_facts": count, "sources": sources}

    @staticmethod
    def _make_id(fact: Fact) -> str:
        raw = f"{fact.topic}:{fact.fact}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
