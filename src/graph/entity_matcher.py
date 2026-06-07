from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0:
        return 0.0
    return dot / norm


class EntityMatcher:
    def __init__(self, embedder: SiliconFlowEmbedder) -> None:
        self._embedder = embedder
        self._entities: list[str] = []
        self._embeddings: list[list[float]] = []

    def build_index(self, entities: list[str]) -> None:
        self._entities = list(entities)
        if not self._entities:
            self._embeddings = []
            return
        self._embeddings = self._embedder.embed_texts(self._entities)
        logger.info("EntityMatcher index built with %d entities", len(self._entities))

    def match(self, query_entity: str, threshold: float = 0.85) -> str | None:
        if not self._entities or not self._embeddings:
            return None
        query_emb = self._embedder.embed_query(query_entity)
        best_score = -1.0
        best_entity: str | None = None
        for entity, emb in zip(self._entities, self._embeddings):
            score = _cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_entity = entity
        if best_score >= threshold:
            return best_entity
        return None

    def match_many(self, query_entities: list[str], threshold: float = 0.85) -> list[str]:
        matched: list[str] = []
        for qe in query_entities:
            result = self.match(qe, threshold)
            if result is not None:
                matched.append(result)
        return matched
