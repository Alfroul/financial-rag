from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from src.correction.external_verifier import ExternalVerifier
from src.correction.nli_verifier import NLIVerifier
from src.correction.retrieval_gate import RetrievalGate
from src.correction.rule_checker import RuleChecker
from src.correction.types import CorrectionResult

if TYPE_CHECKING:
    from src.config import SelfCorrectionConfig
    from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class SelfCorrectingPipeline:
    """Self-correcting RAG pipeline wrapper.

    Wraps an existing RAGPipeline and adds multi-layer hallucination
    detection and correction:
        Layer 0 – retrieval quality gate
        Layer 2 – rule-based pre-check
        Layer 3 – claim decomposition + NLI
        Layer 4 – external model verification (optional)
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: SelfCorrectionConfig,
        api_key: str | None = None,
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        model: str = "mimo-v2.5-pro",
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._gate = RetrievalGate(
            rerank_threshold_good=config.rerank_threshold_good,
            rerank_threshold_weak=config.rerank_threshold_weak,
        )
        self._rule_checker = RuleChecker()
        self._nli = NLIVerifier()
        self._external: ExternalVerifier | None = None
        if api_key:
            self._external = ExternalVerifier(
                api_key=api_key, base_url=base_url, model=model,
            )

    @property
    def inner_pipeline(self) -> RAGPipeline:
        """Access the underlying RAG pipeline for streaming before correction."""
        return self._pipeline

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        """Query with multi-layer correction.

        Returns the original result dict augmented with a ``correction`` key
        carrying a CorrectionResult when any layer fires.
        """
        raw = self._pipeline.query(question, chat_history=chat_history, **kwargs)
        answer = raw.get("answer", "")
        sources = cast(list[dict], raw.get("sources", []))
        source_texts = [s.get("content", "") for s in sources if isinstance(s, dict)]

        # Layer 0: retrieval quality gate
        retrieval_results = self._extract_retrieval_results(raw)
        quality = self._gate.assess(retrieval_results)

        if quality.level == "WEAK":
            logger.info("Retrieval quality WEAK, skipping correction layers")
            return {
                **raw,
                "correction": CorrectionResult(
                    passed=True,
                    flagged_claims=[],
                    layer_results={"retrieval_quality": quality},
                    confidence=0.2,
                ),
            }

        # Layer 2: rule-based pre-check
        issues = self._rule_checker.check(answer, source_texts)
        if not issues:
            logger.info("Rule check clean, returning result")
            return {**raw, "correction": CorrectionResult(passed=True, confidence=0.9)}

        logger.info("Rule check found %d issues, proceeding to Layer 3", len(issues))

        # Layer 3: claim decomposition + NLI
        claims = self._nli.decompose_claims(answer)
        context = " ".join(source_texts)
        verdicts = self._nli.verify(claims, context)

        unsupported = [v for v in verdicts if not v.supported]
        if not unsupported:
            logger.info("NLI: all claims supported")
            return {
                **raw,
                "correction": CorrectionResult(
                    passed=True,
                    layer_results={"rule_issues": issues, "nli": verdicts},
                    confidence=0.8,
                ),
            }

        # Layer 4: external model verification
        unsupported_claims = [v.claim for v in unsupported]
        if self._external:
            external_verdicts = self._external.verify(unsupported_claims, context)
            still_unsupported = [v for v in external_verdicts if not v.supported]
            if not still_unsupported:
                logger.info("External verifier: all claims supported after re-check")
                return {
                    **raw,
                    "correction": CorrectionResult(
                        passed=True,
                        flagged_claims=[],
                        layer_results={
                            "rule_issues": issues,
                            "nli": verdicts,
                            "external": external_verdicts,
                        },
                        confidence=0.85,
                    ),
                }
            unsupported_claims = [v.claim for v in still_unsupported]

        # Retry generation with feedback
        return self._retry_with_feedback(
            question, chat_history, unsupported_claims, raw, kwargs,
        )

    def stream_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **kwargs,
    ):
        """Stream query: delegates directly to inner pipeline (no correction in streaming)."""
        return self._pipeline.stream_query(question, chat_history=chat_history, **kwargs)

    async def aquery(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        """Async query: delegates to inner pipeline."""
        return await self._pipeline.aquery(question, chat_history=chat_history, **kwargs)

    def _retry_with_feedback(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None,
        flagged_claims: list[str],
        original_result: dict,
        kwargs: dict,
    ) -> dict[str, object]:
        feedback = (
            "以下断言可能与参考资料不一致，请重新核实：\n"
            + "\n".join(f"- {c}" for c in flagged_claims)
            + "\n请仅基于参考资料重新回答。"
        )

        retry_history = list(chat_history or [])
        retry_history.append({"role": "assistant", "content": original_result.get("answer", "")})
        retry_history.append({"role": "user", "content": feedback})

        result = original_result
        for attempt in range(self._config.max_retries):
            logger.info("Correction retry %d/%d", attempt + 1, self._config.max_retries)
            result = self._pipeline.query(question, chat_history=retry_history, **kwargs)
            new_answer = result.get("answer", "")

            new_claims = self._nli.decompose_claims(new_answer)
            sources = cast(list[dict], result.get("sources", []))
            source_texts = [s.get("content", "") for s in sources if isinstance(s, dict)]
            context = " ".join(source_texts)
            new_verdicts = self._nli.verify(new_claims, context)
            still_unsupported = [v for v in new_verdicts if not v.supported]

            if not still_unsupported:
                logger.info("Retry %d succeeded — all claims now supported", attempt + 1)
                return {
                    **result,
                    "correction": CorrectionResult(
                        passed=True,
                        flagged_claims=flagged_claims,
                        layer_results={"retries": attempt + 1},
                        confidence=0.7,
                    ),
                }

        logger.info("Max retries exhausted, returning last result")
        return {
            **result,
            "correction": CorrectionResult(
                passed=False,
                flagged_claims=flagged_claims,
                layer_results={"retries": self._config.max_retries},
                confidence=0.3,
            ),
        }

    @staticmethod
    def _extract_retrieval_results(raw: dict) -> list:
        from src.retriever.retriever import RetrievalResult

        sources = raw.get("sources", [])
        results: list[RetrievalResult] = []
        for s in sources:
            if isinstance(s, dict):
                results.append(RetrievalResult(
                    content=s.get("content", ""),
                    score=s.get("score", 0.0),
                    metadata=s.get("metadata", {}),
                    doc_id=s.get("metadata", {}).get("doc_id", ""),
                ))
        return results
