from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jieba

from src.graph.triple import Triple

if TYPE_CHECKING:
    from src.graph.entity_matcher import EntityMatcher
    from src.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)

_COMPARISON_KEYWORDS = {"对比", "vs", "比较", "相比", "和", "与", "区别", "差异"}


class GraphRetriever:
    def __init__(
        self,
        graph_store: GraphStore,
        entity_matcher: EntityMatcher,
        max_neighbors: int = 50,
        max_depth: int = 2,
    ) -> None:
        self._graph_store = graph_store
        self._entity_matcher = entity_matcher
        self._max_neighbors = max_neighbors
        self._max_depth = max_depth

    def retrieve(self, query: str, mode: str = "neighbors") -> list[Triple]:
        known_entities = self._graph_store.get_entities()
        if not known_entities:
            return []

        candidate_entities = self._extract_entities(query, known_entities)
        matched = self._entity_matcher.match_many(candidate_entities)

        if not matched:
            return []

        if mode == "comparison" and len(matched) >= 2:
            return self._retrieve_comparison(matched[0], matched[1])
        if mode == "path" and len(matched) >= 2:
            return self._retrieve_path(matched[0], matched[1])
        return self._retrieve_neighbors(matched[0])

    def _extract_entities(self, query: str, known_entities: list[str]) -> list[str]:
        known_set = set(known_entities)
        tokens = list(jieba.cut(query))
        candidates: list[str] = []
        for token in tokens:
            token = token.strip()
            if len(token) < 2:
                continue
            if token in known_set:
                candidates.append(token)
        if not candidates:
            for entity in known_entities:
                if entity in query:
                    candidates.append(entity)
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _retrieve_neighbors(self, entity: str) -> list[Triple]:
        triples = self._graph_store.query_neighbors(entity, max_depth=self._max_depth)
        return triples[:self._max_neighbors]

    def _retrieve_comparison(self, entity_a: str, entity_b: str) -> list[Triple]:
        triples_a = self._graph_store.query_neighbors(entity_a, max_depth=1)
        triples_b = self._graph_store.query_neighbors(entity_b, max_depth=1)
        seen: set[tuple[str, str, str]] = set()
        merged: list[Triple] = []
        for t in triples_a + triples_b:
            key = (t.head, t.relation, t.tail)
            if key not in seen:
                seen.add(key)
                merged.append(t)
        return merged[:self._max_neighbors]

    def _retrieve_path(self, entity_a: str, entity_b: str) -> list[Triple]:
        paths = self._graph_store.query_path(entity_a, entity_b)
        if not paths:
            return []
        return paths[0]
