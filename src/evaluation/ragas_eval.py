from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI

    # ChatOpenAI is imported inside _get_llm for lazy loading
    from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(
        self,
        api_key: str,
        model: str = "mimo-v2-pro",
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedding_model = embedding_model
        self._base_url = base_url
        self._embedding_api_key = embedding_api_key or api_key
        self._embedding_base_url = embedding_base_url or base_url
        self._llm: ChatOpenAI | None = None

    def _get_llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM using simple lazy import."""
        if self._llm is not None:
            return self._llm
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,  # type: ignore[arg-type]
            temperature=0,
        )
        return self._llm

    def evaluate(
        self,
        questions: list[str],
        responses: list[str],
        contexts: list[list[str]],
        references: list[str] | None = None,
    ) -> dict[str, float]:
        # Local imports to keep dependencies lazy and testable
        from datasets import Dataset
        from langchain_openai import OpenAIEmbeddings
        from ragas import evaluate as ragas_evaluate

        try:
            from ragas import RunConfig
        except ImportError:
            RunConfig = None
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
        )

        data_dict: dict[str, Any] = {
            "user_input": questions,
            "response": responses,
            "retrieved_contexts": contexts,
        }
        if references:
            data_dict["reference"] = references

        dataset: Dataset = Dataset.from_dict(data_dict)

        metrics = [faithfulness, context_precision]
        if references:
            metrics.append(context_recall)

        embeddings = OpenAIEmbeddings(
            model=self._embedding_model,
            base_url=self._embedding_base_url,
            api_key=self._embedding_api_key,  # type: ignore[arg-type]
        )

        eval_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "metrics": metrics,
            "llm": self._get_llm(),
            "embeddings": embeddings,
        }
        if RunConfig is not None:
            eval_kwargs["run_config"] = RunConfig(max_workers=2, max_retries=10, max_wait=60)

        eval_result = ragas_evaluate(**eval_kwargs)

        # ragas 0.4+: EvaluationResult._repr_dict is {metric_name: average_score}
        if hasattr(eval_result, "_repr_dict"):
            raw_scores = eval_result._repr_dict
        elif hasattr(eval_result, "to_pandas"):
            pdf = eval_result.to_pandas()
            # only keep metric columns (skip dataset columns)
            metric_cols = [
                c for c in pdf.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")
            ]
            raw_scores = {col: float(pdf[col].mean()) for col in metric_cols}
        elif isinstance(eval_result, dict):
            raw_scores = eval_result
        else:
            logger.error("无法解析 RAGAS 结果: %s", type(eval_result))
            raw_scores = {}

        scores: dict[str, float] = {}
        for key, value in raw_scores.items():
            if hasattr(value, "item"):
                scores[key] = float(value.item())
            else:
                scores[key] = float(value)

        logger.info("RAGAS 评估完成: %s", scores)
        return scores

    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        eval_samples: list[dict],
    ) -> dict[str, float]:
        # Build questions and a references list aligned to the questions.
        questions = [s["question"] for s in eval_samples]
        # Build references aligned with questions; use empty string for missing references
        references = [s.get("reference") or "" for s in eval_samples]

        responses: list[str] = []
        all_contexts: list[list[str]] = []

        for i, sample in enumerate(eval_samples):
            logger.info("评估进度: %d/%d — %s", i + 1, len(eval_samples), sample["question"][:30])
            try:
                result = pipeline.query(sample["question"])
                responses.append(result["answer"])
                all_contexts.append([src["content"] for src in result["sources"]])
            except Exception as e:
                logger.warning("Pipeline 查询失败，跳过: %s", e)
                responses.append("")
                all_contexts.append([])

        # Filter to only samples that have non-empty references for recall metric
        has_ref_indices = [idx for idx, r in enumerate(references) if r]
        if has_ref_indices:
            ref_questions = [questions[i] for i in has_ref_indices]
            ref_responses = [responses[i] for i in has_ref_indices]
            ref_contexts = [all_contexts[i] for i in has_ref_indices]
            ref_list = [references[i] for i in has_ref_indices]
            return self.evaluate(
                questions=ref_questions,
                responses=ref_responses,
                contexts=ref_contexts,
                references=ref_list,
            )
        else:
            # No references available, evaluate without recall metric
            return self.evaluate(
                questions=questions,
                responses=responses,
                contexts=all_contexts,
                references=None,
            )
