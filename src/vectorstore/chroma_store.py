import logging

import chromadb

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    pass


class ChromaStore:
    """ChromaDB 向量存储封装。

    支持预计算向量的存入、相似度检索、元数据过滤和持久化。
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "financial_docs",
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]] | None = None,
    ) -> None:
        """添加文档向量到存储。

        Args:
            ids: 文档唯一标识列表。
            documents: 文档文本列表。
            embeddings: 预计算的向量列表。
            metadatas: 元数据列表，可选。

        Raises:
            VectorStoreError: 输入校验失败或存储异常。
        """
        if len(ids) != len(documents) or len(ids) != len(embeddings):
            raise VectorStoreError(
                f"ids({len(ids)}), documents({len(documents)}), embeddings({len(embeddings)}) 长度必须一致"
            )

        if metadatas is not None and len(metadatas) != len(ids):
            raise VectorStoreError(f"metadatas({len(metadatas)}) 与 ids({len(ids)}) 长度必须一致")

        try:
            kwargs: dict = {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
            }
            if metadatas is not None:
                kwargs["metadatas"] = metadatas

            self._collection.add(**kwargs)
            logger.info("成功存储 %d 条文档到 %s", len(ids), self._collection_name)
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                raise VectorStoreError(f"向量维度不匹配，请检查 Embedding 模型配置: {e}") from e
            raise VectorStoreError(f"添加文档失败: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        try:
            safe_k = min(max(top_k, 1), 50)
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": safe_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                kwargs["where"] = where
            if where_document is not None:
                kwargs["where_document"] = where_document

            results = self._collection.query(**kwargs)

            # 解析 ChromaDB 响应为统一格式
            items: list[dict] = []
            if not results["documents"] or not results["documents"][0]:
                return items

            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{} for _ in docs]
            dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc, meta, dist, doc_id in zip(docs, metas, dists, ids, strict=False):
                if not doc or not doc.strip():
                    continue
                # ChromaDB cosine distance → 转为相似度分数 (0~1)
                # cosine distance = 1 - cosine_similarity, 所以 similarity = 1 - distance
                score = min(1.0, max(0.0, 1.0 - dist))
                items.append(
                    {
                        "content": doc,
                        "metadata": meta or {},
                        "score": score,
                        "id": doc_id,
                    }
                )

            return items

        except Exception as e:
            logger.error("检索失败: %s", e, exc_info=True)
            raise VectorStoreError(f"检索失败: {e}") from e

    def get_documents_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """批量获取文档内容（供 BM25 等模块使用）。"""
        try:
            result = self._collection.get(ids=ids, include=["documents", "metadatas"])  # type: ignore[assignment]
            doc_map: dict[str, dict] = {}
            docs: list = result.get("documents") or []
            metas: list = result.get("metadatas") or [{}] * len(docs)
            res_ids: list = result.get("ids") or []
            for doc_id, doc, meta in zip(res_ids, docs, metas, strict=False):
                if doc:
                    doc_map[str(doc_id)] = {"content": doc, "metadata": meta or {}}
            return doc_map
        except Exception as e:
            logger.error("批量获取文档失败: %s", e)
            return {}

    def get_all_documents(self) -> dict[str, str]:
        """获取所有文档内容（供 BM25 索引构建使用）。

        注意：对于大型知识库（>10000 文档块），此方法会消耗大量内存。
        """
        try:
            count = self._collection.count()
            if count > 10000:
                logger.warning("知识库包含 %d 条文档，get_all_documents 将消耗大量内存", count)
            result = self._collection.get(include=["documents"])
            docs = result.get("documents") or []
            ids = result.get("ids") or []
            return {doc_id: doc for doc_id, doc in zip(ids, docs, strict=False) if doc}
        except Exception as e:
            logger.error("获取所有文档失败: %s", e)
            return {}

    def delete_collection(self) -> None:
        """删除并重建集合。"""
        try:
            self._client.delete_collection(name=self._collection_name)
            logger.info("已删除集合 %s", self._collection_name)
        except Exception:
            # 集合可能不存在，忽略
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("已重建集合 %s", self._collection_name)

    def get_stats(self) -> dict:
        """获取存储统计信息。

        Returns:
            包含集合名称和文档数量的字典。
        """
        return {
            "collection_name": self._collection_name,
            "document_count": self._collection.count(),
        }

    def get_all_ids(self) -> list[str]:
        """获取所有已存储的文档 ID。"""
        try:
            result = self._collection.get(include=[])  # type: ignore[assignment]
            ids: list = result.get("ids", [])
            return [str(i) for i in ids]
        except Exception as e:
            logger.error("获取 ID 列表失败: %s", e)
            return []

    def delete_by_ids(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
            logger.info("已删除 %d 条文档", len(ids))
        except Exception as e:
            raise VectorStoreError(f"删除文档失败: {e}") from e
