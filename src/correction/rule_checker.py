from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)

# Regex patterns for extracting structured data from text
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_DATE_RE = re.compile(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?")


class RuleChecker:
    """Layer 2: Rule-based pre-check for answer consistency with sources."""

    def __init__(
        self,
        financial_terms: list[str] | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._graph_store = graph_store
        if financial_terms is not None:
            self._financial_terms = financial_terms
        else:
            try:
                from src.retriever.bm25_retriever import _FINANCIAL_TERMS
                self._financial_terms = _FINANCIAL_TERMS
            except ImportError:
                self._financial_terms = []

    def check(self, answer: str, source_texts: list[str]) -> list[dict[str, object]]:
        """Run 4 rule-based checks (number, entity, date, overlap), return issues."""
        combined_sources = " ".join(source_texts)
        issues: list[dict[str, object]] = []

        issues.extend(self._check_numbers(answer, combined_sources))
        issues.extend(self._check_entities(answer, combined_sources))
        issues.extend(self._check_dates(answer, combined_sources))
        issues.extend(self._check_overlap(answer, combined_sources))

        if issues:
            logger.info("RuleChecker found %d issues", len(issues))
        return issues

    def _check_numbers(self, answer: str, source: str) -> list[dict[str, object]]:
        issues: list[dict[str, object]] = []
        for match in _NUMBER_RE.finditer(answer):
            value = match.group()
            if value not in source:
                issues.append({
                    "type": "number",
                    "value": value,
                    "severity": "HIGH",
                    "message": f"Number '{value}' not found in source documents",
                })
        return issues

    def _check_entities(self, answer: str, source: str) -> list[dict[str, object]]:
        issues: list[dict[str, object]] = []
        graph_entities = self._get_graph_entities()
        for term in self._financial_terms:
            if term in answer and term not in source:
                if graph_entities and term in graph_entities:
                    logger.debug("Entity '%s' not in source but found in graph", term)
                    continue
                issues.append({
                    "type": "entity",
                    "value": term,
                    "severity": "MEDIUM",
                    "message": f"Financial term '{term}' not found in source documents",
                })
        return issues

    def _get_graph_entities(self) -> set[str]:
        if self._graph_store is None:
            return set()
        try:
            return set(self._graph_store.get_entities())
        except Exception as e:
            logger.warning("RuleChecker: failed to get graph entities: %s", e)
            return set()

    def _check_dates(self, answer: str, source: str) -> list[dict[str, object]]:
        issues: list[dict[str, object]] = []
        for match in _DATE_RE.finditer(answer):
            value = match.group()
            if value not in source:
                issues.append({
                    "type": "date",
                    "value": value,
                    "severity": "HIGH",
                    "message": f"Date '{value}' not found in source documents",
                })
        return issues

    def _check_overlap(self, answer: str, source: str, threshold: float = 0.3) -> list[dict[str, object]]:
        answer_words = set(answer)
        source_words = set(source)
        if not answer_words:
            return []
        overlap = len(answer_words & source_words) / len(answer_words)
        if overlap < threshold:
            return [{
                "type": "overlap",
                "value": f"{overlap:.2f}",
                "severity": "MEDIUM",
                "message": f"Low vocabulary overlap ({overlap:.1%}) between answer and source",
            }]
        return []
