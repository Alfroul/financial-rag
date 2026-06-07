from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Generator
from string import Template
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from src.agent.base_tool import BaseTool
from src.agent.react import ReActAgent
from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.financial_search import FinancialSearchTool
from src.cache import QueryCache
from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import FactExtractor
from src.graph.graph_retriever import GraphRetriever
from src.graph.graph_store import GraphStore
from src.graph.triple import Triple
from src.metrics.collector import MetricsCollector, QueryMetrics
from src.observability.langfuse_tracer import LangfuseTracer
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:
    import tiktoken

    from src.config import GraphConfig, RAGConfig, RerankerConfig
    from src.generator.mimo_llm import MimoLLM
    from src.generator.query_rewriter import QueryRewriter
    from src.reranker.local_reranker import LocalRreranker

RetrieverType: TypeAlias = Retriever | HybridRetriever

logger = logging.getLogger(__name__)

_cached_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding | None:
    global _cached_tokenizer
    if _cached_tokenizer is None:
        try:
            import tiktoken

            _cached_tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
    return _cached_tokenizer


# system prompt 模板
_SYSTEM_PROMPT_TEMPLATE = """\
你是一位严谨的金融分析师。请严格遵守以下规则回答问题：

## 回答规则
1. **只根据参考资料**：所有事实必须来自下方参考资料，禁止编造、推测或使用外部知识
2. **先分析再回答**：先判断问题类型（定义/数值/对比/时间），再从资料中定位相关信息
3. **标注来源**：每个事实后标注 [来源 N]，N 对应参考资料的编号
4. **资料不足时**：明确说明"根据现有参考资料无法完整回答"，指出缺少什么信息
5. **简洁直接**：只回答被问到的问题，不展开无关内容

## 参考资料
$retrieved_documents

## 问题
$user_question"""


class RAGPipelineError(Exception):
    pass


class RAGPipeline:
    """RAG 主流程编排：检索 → 构造 Prompt → LLM 生成。"""

    def __init__(
        self,
        retriever: Retriever | HybridRetriever,
        llm: MimoLLM,
        config: RAGConfig,
        reranker: LocalRreranker | None = None,
        reranker_config: RerankerConfig | None = None,
        query_rewriter: QueryRewriter | None = None,
        cache: QueryCache | None = None,
        fact_cache: FactCacheStore | None = None,
        fact_extractor: FactExtractor | None = None,
        fact_cache_threshold: float = 0.7,
        graph_store: GraphStore | None = None,
        graph_config: GraphConfig | None = None,
        graph_retriever: GraphRetriever | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_tokens = config.max_context_tokens
        self._reranker = reranker
        self._reranker_config = reranker_config
        self._query_rewriter = query_rewriter
        self._cache = cache
        self._fact_cache = fact_cache
        self._fact_extractor = fact_extractor
        self._fact_cache_threshold = fact_cache_threshold
        self._graph_store = graph_store
        self._graph_config = graph_config
        self._graph_retriever = graph_retriever
        self._tracer = tracer

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> dict[str, Any]:
        """同步查询：检索 → 构造 Prompt → 生成回答。"""
        if not question or not question.strip():
            return {"answer": "请输入有效的问题。", "sources": []}
        if len(question) > 4096:
            return {"answer": "问题过长，请精简后重试。", "sources": []}

        collector = MetricsCollector()
        from_cache = False
        tracer = self._tracer
        trace_id: str | None = None

        if tracer is not None and tracer.enabled:
            trace_id = tracer.start_trace(question)

        if self._cache is not None:
            cached = self._run_async(self._cache.get(question))
            if cached is not None:
                logger.info("Cache hit for sync query: %s", question)
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        total_ms=0.0,
                        cache_hit=True,
                    )
                )
                if tracer is not None and trace_id is not None:
                    tracer.end_trace(trace_id, output="cache_hit")
                return cast(dict[str, Any], cached)

        if chat_history and self._query_rewriter:
            question = self._query_rewriter.rewrite(question, chat_history)

        # 知识缓存层集成点 — 在检索前查询FactCache，命中则跳过RAG检索
        fact_cache_result = self._try_fact_cache(question, chat_history, collector)
        if fact_cache_result is not None:
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output="fact_cache_hit")
            return fact_cache_result

        graph_triples = self._try_graph_query(question)

        # --- retrieval span ---
        retrieval_span: str | None = None
        if tracer is not None and trace_id is not None:
            retrieval_span = tracer.start_span(trace_id, "retrieval", {"query": question})
        t_retrieve = perf_counter()
        results = self._retrieve_or_fallback(question, **retrieve_kwargs)
        retrieve_ms = (perf_counter() - t_retrieve) * 1000
        if retrieval_span is not None and tracer is not None:
            tracer.end_span(
                retrieval_span,
                output_data={
                    "num_results": len(results) if results else 0,
                    "ms": round(retrieve_ms, 2),
                },
            )

        if graph_triples:
            if results is None:
                results = []
            rag_context = ""
            sources: list[dict] = []
            if results:
                if self._reranker:
                    rerank_span = (
                        tracer.start_span(trace_id, "rerank")
                        if tracer is not None and trace_id is not None
                        else None
                    )
                    t_rerank = perf_counter()
                    results = self._rerank_results(question, results)
                    rerank_ms = (perf_counter() - t_rerank) * 1000
                    if rerank_span and tracer is not None:
                        tracer.end_span(
                            rerank_span,
                            output_data={
                                "ms": round(rerank_ms, 2),
                                "num_results": len(results),
                            },
                        )
                formatted_context, sources = self._format_context(results)
                rag_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)
            trimmed_context = self._build_graph_prompt(graph_triples, rag_context)
        else:
            if results is None:
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        retrieve_ms=retrieve_ms,
                        total_ms=retrieve_ms,
                        num_sources=0,
                    )
                )
                if tracer is not None and trace_id is not None:
                    tracer.end_trace(trace_id, output="no_results")
                return {"answer": "抱歉，在知识库中未找到与您问题相关的信息。", "sources": []}

            rerank_ms = 0.0
            if self._reranker and results:
                rerank_span = (
                    tracer.start_span(trace_id, "rerank")
                    if tracer is not None and trace_id is not None
                    else None
                )
                t_rerank = perf_counter()
                results = self._rerank_results(question, results)
                rerank_ms = (perf_counter() - t_rerank) * 1000
                if rerank_span and tracer is not None:
                    tracer.end_span(
                        rerank_span,
                        output_data={
                            "ms": round(rerank_ms, 2),
                            "num_results": len(results),
                        },
                    )

            formatted_context, sources = self._format_context(results)
            trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )

        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        # --- generation span ---
        gen_span: str | None = None
        if tracer is not None and trace_id is not None:
            gen_span = tracer.start_span(trace_id, "generation")
        t_generate = perf_counter()
        try:
            answer = self._llm.chat(system_prompt, messages)
        except Exception as e:
            if gen_span is not None and tracer is not None:
                tracer.end_span(gen_span, output_data={"error": str(e)})
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output="generation_error")
            raise RAGPipelineError(f"LLM 生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000
        if gen_span is not None and tracer is not None:
            tracer.end_span(gen_span, output_data={"ms": round(generate_ms, 2), "answer_len": len(answer)})

        total_ms = retrieve_ms + rerank_ms + generate_ms
        collector.record(
            QueryMetrics(
                question=question[:50],
                retrieve_ms=round(retrieve_ms, 2),
                rerank_ms=round(rerank_ms, 2),
                generate_ms=round(generate_ms, 2),
                total_ms=round(total_ms, 2),
                cache_hit=from_cache,
                strategy=self._get_strategy_name(),
                reranker_enabled=self._reranker is not None,
                num_sources=len(sources),
            )
        )

        result = {"answer": answer, "sources": sources}

        self._extract_and_cache_facts(sources, question)

        if self._cache is not None:
            self._run_async(self._cache.put(question, result))

        if tracer is not None and trace_id is not None:
            tracer.end_trace(trace_id, output={"answer_len": len(answer), "num_sources": len(sources)})
        return result

    def stream_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> Generator[dict, None, None]:
        """流式查询：检索 → 构造 Prompt → 逐 token 生成回答。

        Yields:
            首先产出 {"type": "sources", "sources": list[dict]}，
            之后逐 token 产出 {"type": "answer", "content": str}。
        """
        if not question or not question.strip():
            yield {"type": "answer", "content": "请输入有效的问题。"}
            return
        if len(question) > 4096:
            yield {"type": "answer", "content": "问题过长，请精简后重试。"}
            return

        if self._cache is not None:
            cached = self._run_async(self._cache.get(question))
            if cached is not None:
                logger.info("Cache hit for stream_query: %s", question)
                yield {"type": "sources", "sources": cached.get("sources", [])}
                yield {"type": "answer", "content": cached.get("answer", "")}
                return

        if chat_history and self._query_rewriter:
            question = self._query_rewriter.rewrite(question, chat_history)

        graph_triples = self._try_graph_query(question)

        results = self._retrieve_or_fallback(question, **retrieve_kwargs)

        if graph_triples:
            if results is None:
                results = []
            rag_context = ""
            sources: list[dict] = []
            if results:
                if self._reranker:
                    results = self._rerank_results(question, results)
                formatted_context, sources = self._format_context(results)
                rag_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)
            trimmed_context = self._build_graph_prompt(graph_triples, rag_context)
        else:
            if results is None:
                yield {"type": "answer", "content": "抱歉，在知识库中未找到与您问题相关的信息。"}
                return
            if self._reranker and results:
                results = self._rerank_results(question, results)
            formatted_context, sources = self._format_context(results)
            trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        yield {"type": "sources", "sources": sources}

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )
        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        full_answer_parts: list[str] = []
        try:
            for chunk in self._llm.stream_chat(system_prompt, messages):
                full_answer_parts.append(chunk)
                yield {"type": "answer", "content": chunk}
        except Exception as e:
            raise RAGPipelineError(f"LLM 流式生成失败: {e}") from e

        if self._cache is not None:
            self._run_async(self._cache.put(question, {"answer": "".join(full_answer_parts), "sources": sources}))

    async def aquery(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> dict:
        """异步查询：检索 → 构造 Prompt → 生成回答。"""
        if not question or not question.strip():
            return {"answer": "请输入有效的问题。", "sources": []}
        if len(question) > 4096:
            return {"answer": "问题过长，请精简后重试。", "sources": []}

        collector = MetricsCollector()
        from_cache = False
        tracer = self._tracer
        trace_id: str | None = None

        if tracer is not None and tracer.enabled:
            trace_id = tracer.start_trace(question)

        # 1) 通过语义缓存快速命中（阶段 2，若启用）
        if self._cache is not None:
            cached = await self._cache.get(question)
            if cached is not None:
                logger.info("Cache hit for async query: %s", question)
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        total_ms=0.0,
                        cache_hit=True,
                    )
                )
                if tracer is not None and trace_id is not None:
                    tracer.end_trace(trace_id, output="cache_hit")
                return cast(dict[str, Any], cached)

        if chat_history and self._query_rewriter:
            question = await self._query_rewriter.arewrite(question, chat_history)

        # 知识缓存层集成点 — 在检索前查询FactCache，命中则跳过RAG检索
        fact_cache_result = await self._atry_fact_cache(question, chat_history, collector)
        if fact_cache_result is not None:
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output="fact_cache_hit")
            return fact_cache_result

        graph_triples = self._try_graph_query(question)

        # --- retrieval span ---
        retrieval_span: str | None = None
        if tracer is not None and trace_id is not None:
            retrieval_span = tracer.start_span(trace_id, "retrieval", {"query": question})
        t_retrieve = perf_counter()
        results = await self._aretrieve_or_fallback(question, **retrieve_kwargs)
        retrieve_ms = (perf_counter() - t_retrieve) * 1000
        if retrieval_span is not None and tracer is not None:
            tracer.end_span(
                retrieval_span,
                output_data={
                    "num_results": len(results) if results else 0,
                    "ms": round(retrieve_ms, 2),
                },
            )

        if graph_triples:
            if results is None:
                results = []
            rag_context = ""
            sources: list[dict] = []
            if results:
                if self._reranker:
                    rerank_span = (
                        tracer.start_span(trace_id, "rerank")
                        if tracer is not None and trace_id is not None
                        else None
                    )
                    t_rerank = perf_counter()
                    results = await self._arerank_results(question, results)
                    rerank_ms = (perf_counter() - t_rerank) * 1000
                    if rerank_span and tracer is not None:
                        tracer.end_span(
                            rerank_span,
                            output_data={
                                "ms": round(rerank_ms, 2),
                                "num_results": len(results),
                            },
                        )
                formatted_context, sources = self._format_context(results)
                rag_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)
            trimmed_context = self._build_graph_prompt(graph_triples, rag_context)
        else:
            if results is None:
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        retrieve_ms=round(retrieve_ms, 2),
                        total_ms=round(retrieve_ms, 2),
                        num_sources=0,
                    )
                )
                if tracer is not None and trace_id is not None:
                    tracer.end_trace(trace_id, output="no_results")
                return {"answer": "抱歉，在知识库中未找到与您问题相关的信息。", "sources": []}

            rerank_ms = 0.0
            if self._reranker and results:
                rerank_span = (
                    tracer.start_span(trace_id, "rerank")
                    if tracer is not None and trace_id is not None
                    else None
                )
                t_rerank = perf_counter()
                results = await self._arerank_results(question, results)
                rerank_ms = (perf_counter() - t_rerank) * 1000
                if rerank_span and tracer is not None:
                    tracer.end_span(
                        rerank_span,
                        output_data={
                            "ms": round(rerank_ms, 2),
                            "num_results": len(results),
                        },
                    )

            formatted_context, sources = self._format_context(results)
            trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )

        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        # --- generation span ---
        gen_span: str | None = None
        if tracer is not None and trace_id is not None:
            gen_span = tracer.start_span(trace_id, "generation")
        t_generate = perf_counter()
        try:
            answer = await self._llm.achat(system_prompt, messages)
        except Exception as e:
            if gen_span is not None and tracer is not None:
                tracer.end_span(gen_span, output_data={"error": str(e)})
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output="generation_error")
            raise RAGPipelineError(f"LLM 异步生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000
        if gen_span is not None and tracer is not None:
            tracer.end_span(gen_span, output_data={"ms": round(generate_ms, 2), "answer_len": len(answer)})

        total_ms = retrieve_ms + rerank_ms + generate_ms
        collector.record(
            QueryMetrics(
                question=question[:50],
                retrieve_ms=round(retrieve_ms, 2),
                rerank_ms=round(rerank_ms, 2),
                generate_ms=round(generate_ms, 2),
                total_ms=round(total_ms, 2),
                cache_hit=from_cache,
                strategy=self._get_strategy_name(),
                reranker_enabled=self._reranker is not None,
                num_sources=len(sources),
            )
        )

        # 2) 缓存写入
        if self._cache is not None:
            await self._cache.put(question, {"answer": answer, "sources": sources})

        await asyncio.to_thread(self._extract_and_cache_facts, sources, question)

        if tracer is not None and trace_id is not None:
            tracer.end_trace(trace_id, output={"answer_len": len(answer), "num_sources": len(sources)})
        return {"answer": answer, "sources": sources}

    async def astream_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> AsyncGenerator[dict, None]:
        """异步流式查询：检索 → 构造 Prompt → 逐 token 生成回答。"""
        if not question or not question.strip():
            yield {"type": "answer", "content": "请输入有效的问题。"}
            return
        if len(question) > 4096:
            yield {"type": "answer", "content": "问题过长，请精简后重试。"}
            return

        if chat_history and self._query_rewriter:
            question = await self._query_rewriter.arewrite(question, chat_history)

        graph_triples = self._try_graph_query(question)

        results = await self._aretrieve_or_fallback(question, **retrieve_kwargs)

        if graph_triples:
            if results is None:
                results = []
            rag_context = ""
            sources: list[dict] = []
            if results:
                if self._reranker:
                    results = await self._arerank_results(question, results)
                formatted_context, sources = self._format_context(results)
                rag_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)
            trimmed_context = self._build_graph_prompt(graph_triples, rag_context)
        else:
            if results is None:
                yield {"type": "answer", "content": "抱歉，在知识库中未找到与您问题相关的信息。"}
                return
            if self._reranker and results:
                results = await self._arerank_results(question, results)
            formatted_context, sources = self._format_context(results)
            trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        yield {"type": "sources", "sources": sources}

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )
        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        try:
            async for chunk in self._llm.astream_chat(system_prompt, messages):
                yield {"type": "answer", "content": chunk}
        except Exception as e:
            raise RAGPipelineError(f"LLM 异步流式生成失败: {e}") from e

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _try_fact_cache(
        self, question: str, chat_history: list[dict[str, str]] | None, collector: MetricsCollector
    ) -> dict[str, Any] | None:
        """同步：尝试从FactCache命中回答。命中返回结果dict，未命中返回None。"""
        if self._fact_cache is None:
            return None
        fact_results = self._fact_cache.search_by_text(question, threshold=self._fact_cache_threshold)
        if not fact_results:
            return None
        logger.info("FactCache hit: %d facts for query: %s", len(fact_results), question[:50])
        system_prompt, messages, sources = self._build_fact_prompt(question, chat_history, fact_results)
        t_generate = perf_counter()
        try:
            answer = self._llm.chat(system_prompt, messages)
        except Exception as e:
            raise RAGPipelineError(f"LLM 生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000
        collector.record(
            QueryMetrics(
                question=question[:50],
                generate_ms=round(generate_ms, 2),
                total_ms=round(generate_ms, 2),
                cache_hit=True,
                strategy="fact_cache",
                num_sources=len(sources),
            )
        )
        return {"answer": answer, "sources": sources}

    async def _atry_fact_cache(
        self, question: str, chat_history: list[dict[str, str]] | None, collector: MetricsCollector
    ) -> dict[str, Any] | None:
        """异步：尝试从FactCache命中回答。命中返回结果dict，未命中返回None。"""
        if self._fact_cache is None:
            return None
        fact_results = await self._fact_cache.asearch_by_text(question, threshold=self._fact_cache_threshold)
        if not fact_results:
            return None
        logger.info("FactCache hit (async): %d facts for query: %s", len(fact_results), question[:50])
        system_prompt, messages, sources = self._build_fact_prompt(question, chat_history, fact_results)
        t_generate = perf_counter()
        try:
            answer = await self._llm.achat(system_prompt, messages)
        except Exception as e:
            raise RAGPipelineError(f"LLM 异步生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000
        collector.record(
            QueryMetrics(
                question=question[:50],
                generate_ms=round(generate_ms, 2),
                total_ms=round(generate_ms, 2),
                cache_hit=True,
                strategy="fact_cache",
                num_sources=len(sources),
            )
        )
        return {"answer": answer, "sources": sources}

    def _build_fact_prompt(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None,
        fact_results: list,
    ) -> tuple[str, list[dict[str, str]], list[dict]]:
        """从fact列表构造system_prompt、messages和sources。"""
        fact_context = "\n\n".join(
            f"[来源 {i+1}] {f.topic}: {f.fact}" for i, f in enumerate(fact_results)
        )
        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=fact_context,
            user_question=question,
        )
        messages = self._build_messages(
            chat_history, question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )
        sources = [
            {"content": f.fact, "score": 1.0, "metadata": {"topic": f.topic, "source": f.source}}
            for f in fact_results
        ]
        return system_prompt, messages, sources

    def _extract_and_cache_facts(self, sources: list[dict], question: str) -> None:
        """从RAG结果中提取fact并存入缓存（同步调用，仅作fire-and-forget）。"""
        if self._fact_cache is None or self._fact_extractor is None:
            return
        try:
            context_text = "\n\n".join(s.get("content", "") for s in sources)
            facts, triples = self._fact_extractor.extract(context_text, source=f"rag:{question[:30]}")
            if facts:
                self._fact_cache.add_facts(facts)
                logger.info("Extracted and cached %d facts from RAG result", len(facts))
            if triples and self._graph_store is not None:
                graph_enabled = self._graph_config is not None and self._graph_config.enabled
                if graph_enabled:
                    try:
                        count = self._graph_store.add_triples(triples)
                        logger.info("Stored %d triples to graph store", count)
                    except Exception as te:
                        logger.warning("Triple storage failed: %s", te)
        except Exception as e:
            if not hasattr(self, "_fact_extract_failures"):
                self._fact_extract_failures = 0
            self._fact_extract_failures += 1
            if self._fact_extract_failures <= 3 or self._fact_extract_failures % 50 == 0:
                logger.warning("Fact extraction/cache storage failed (count=%d): %s", self._fact_extract_failures, e)

    # ------------------------------------------------------------------
    # 图路由
    # ------------------------------------------------------------------

    _COMPARISON_WORDS = re.compile(r"(对比|vs|比较|和|与|区别|差异)")
    _CAUSAL_WORDS = re.compile(r"(为什么|原因|影响|导致|为什么|缘故|因素)")

    def _route_query(self, query: str) -> str:
        if self._graph_store is None:
            return "rag"
        known = self._graph_store.get_entities()
        matched = [e for e in known if e in query]
        if len(matched) >= 2 and self._COMPARISON_WORDS.search(query):
            return "graph_comparison"
        if self._CAUSAL_WORDS.search(query) and matched:
            return "graph_causal"
        return "rag"

    def _try_graph_query(self, query: str) -> list[Triple] | None:
        if self._graph_retriever is None or self._graph_config is None:
            return None
        if not self._graph_config.enabled:
            return None
        route = self._route_query(query)
        if route == "graph_comparison":
            triples = self._graph_retriever.retrieve(query, mode="comparison")
        elif route == "graph_causal":
            triples = self._graph_retriever.retrieve(query, mode="neighbors")
        else:
            return None
        return triples if triples else None

    @staticmethod
    def _build_graph_prompt(triples: list[Triple], rag_context: str) -> str:
        graph_section = "\n".join(
            f"知识点: {t.head} {t.relation} {t.tail}" for t in triples
        )
        parts = [f"[图谱知识]\n{graph_section}"]
        if rag_context:
            parts.append(f"[文档来源]\n{rag_context}")
        return "\n\n".join(parts)

    def _get_strategy_name(self) -> str:
        if isinstance(self._retriever, HybridRetriever):
            return self._retriever._config.strategy  # noqa: SLF001
        return "vector"

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    async def _aretrieve_or_fallback(self, question: str, **retrieve_kwargs) -> list[RetrievalResult] | None:
        try:
            results = await self._retriever.aretrieve(question, **retrieve_kwargs)
        except Exception as e:
            logger.warning("异步检索异常，降级为无上下文回答: %s", e)
            return None

        if not results:
            logger.info("异步检索无结果，降级为无上下文回答")
            return None
        return results

    async def _arerank_results(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Position-Aware Blending (async)."""
        cfg = self._reranker_config
        if cfg is None:
            return results

        contents = [r.content for r in results]

        orig_scores = [r.score for r in results]
        max_s, min_s = max(orig_scores), min(orig_scores)
        norm_orig = [(s - min_s) / (max_s - min_s) for s in orig_scores] if max_s > min_s else [1.0] * len(orig_scores)

        try:
            reranker = self._reranker
            if reranker is None:
                return results
            rerank_results = await reranker.arerank(
                question,
                contents,
                top_n=cfg.top_n,
            )
        except Exception as e:
            logger.warning("异步 Rerank 失败，降级为原始排序: %s", e)
            return results

        blended: list[RetrievalResult] = []
        for rr in rerank_results:
            if rr.index < 0 or rr.index >= len(results):
                logger.warning("Reranker returned out-of-range index %d, skipping", rr.index)
                continue

            if rr.index < 3:
                w_orig, w_rerank = 0.75, 0.25
            elif rr.index < 10:
                w_orig, w_rerank = 0.60, 0.40
            else:
                w_orig, w_rerank = 0.40, 0.60

            blended_score = w_orig * norm_orig[rr.index] + w_rerank * rr.relevance_score

            if blended_score < cfg.min_score:
                continue

            original = results[rr.index]
            blended.append(
                RetrievalResult(
                    content=original.content,
                    score=blended_score,
                    metadata=original.metadata,
                    doc_id=original.doc_id,
                )
            )

        blended.sort(key=lambda x: x.score, reverse=True)
        logger.info("异步 Position-Aware Rerank 完成：输入 %d 条，输出 %d 条", len(results), len(blended))
        return blended

    def _retrieve_or_fallback(self, question: str, **retrieve_kwargs) -> list[RetrievalResult] | None:
        try:
            results = self._retriever.retrieve(question, **retrieve_kwargs)
        except Exception as e:
            logger.warning("检索异常，降级为无上下文回答: %s", e)
            return None

        if not results:
            logger.info("检索无结果，降级为无上下文回答")
            return None
        return results

    def _rerank_results(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Position-Aware Blending: 按原始排名位置加权融合 RRF 分数与 Reranker 分数。

        参考 QMD 设计：顶部文档信任原始排序，靠后文档更信任 reranker。
        - 原始排名 1-3:  75% 原始分 + 25% reranker 分
        - 原始排名 4-10: 60% 原始分 + 40% reranker 分
        - 原始排名 11+:  40% 原始分 + 60% reranker 分
        """
        cfg = self._reranker_config
        if cfg is None:
            return results

        contents = [r.content for r in results]

        orig_scores = [r.score for r in results]
        max_s, min_s = max(orig_scores), min(orig_scores)
        norm_orig = [(s - min_s) / (max_s - min_s) for s in orig_scores] if max_s > min_s else [1.0] * len(orig_scores)

        try:
            reranker = self._reranker
            if reranker is None:
                return results
            rerank_results = reranker.rerank(
                question,
                contents,
                top_n=cfg.top_n,
            )
        except Exception as e:
            logger.warning("Rerank 失败，降级为原始排序: %s", e)
            return results

        blended: list[RetrievalResult] = []
        for rr in rerank_results:
            if rr.index < 0 or rr.index >= len(results):
                logger.warning("Reranker returned out-of-range index %d, skipping", rr.index)
                continue

            if rr.index < 3:
                w_orig, w_rerank = 0.75, 0.25
            elif rr.index < 10:
                w_orig, w_rerank = 0.60, 0.40
            else:
                w_orig, w_rerank = 0.40, 0.60

            blended_score = w_orig * norm_orig[rr.index] + w_rerank * rr.relevance_score

            if blended_score < cfg.min_score:
                continue

            original = results[rr.index]
            blended.append(
                RetrievalResult(
                    content=original.content,
                    score=blended_score,
                    metadata=original.metadata,
                    doc_id=original.doc_id,
                )
            )

        blended.sort(key=lambda x: x.score, reverse=True)
        logger.info("Position-Aware Rerank 完成：输入 %d 条，输出 %d 条", len(results), len(blended))
        return blended

    # ------------------------------------------------------------------
    # Agent 模式
    # ------------------------------------------------------------------

    def agent_query(
        self,
        task: str,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        """Agent 模式：支持多步推理。"""
        if not task or not task.strip():
            return {"answer": "请输入有效的任务。", "steps": []}
        agent = self._create_agent(max_steps=max_steps)
        answer = agent.run(task)
        return {"answer": answer, "steps": agent.get_steps()}

    async def aagent_query(
        self,
        task: str,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        """异步 Agent 模式。"""
        if not task or not task.strip():
            return {"answer": "请输入有效的任务。", "steps": []}
        agent = self._create_agent(max_steps=max_steps)
        answer = await agent.arun(task)
        return {"answer": answer, "steps": agent.get_steps()}

    def _create_agent(self, max_steps: int = 6) -> ReActAgent:
        """从当前组件组装 ReActAgent。"""
        tools: list[BaseTool] = [
            FinancialSearchTool(self),
            CalculatorTool(),
        ]
        return ReActAgent(llm=self._llm, tools=tools, max_steps=max_steps, tracer=self._tracer)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        chat_history: list[dict[str, str]] | None,
        question: str,
        max_history_tokens: int = 2400,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if chat_history:
            trimmed = list(chat_history)
            try:
                encoder = _get_tokenizer()
                if encoder is not None:
                    history_text = "\n".join(m.get("content", "") for m in trimmed)
                    while len(trimmed) > 2 and len(encoder.encode(history_text)) > max_history_tokens:
                        trimmed = trimmed[2:]
                        history_text = "\n".join(m.get("content", "") for m in trimmed)
                else:
                    trimmed = chat_history[-10:]
            except Exception as e:
                logger.warning("Token trimming failed, using last 10 messages: %s", e)
                trimmed = chat_history[-10:]
            messages.extend(trimmed)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _format_context(
        results: list[RetrievalResult],
    ) -> tuple[str, list[dict]]:
        """将检索结果格式化为带编号的参考文本。

        Returns:
            (formatted_context, sources_list)
        """
        parts: list[str] = []
        sources: list[dict] = []

        for idx, result in enumerate(results, start=1):
            title = result.metadata.get("title", result.metadata.get("source", "未知来源"))
            parts.append(f"[来源 {idx}] 来源: {title}\n{result.content}")
            sources.append(
                {
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                }
            )

        formatted = "\n\n".join(parts)
        return formatted, sources

    def _count_tokens(self, text: str) -> int:
        encoder = _get_tokenizer()
        if encoder is not None:
            return len(encoder.encode(text))

        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_count = len(text) - chinese_count
        return int(chinese_count * 1.5 + other_count * 0.25)

    def _trim_context(
        self, formatted_docs: str, max_tokens: int, sources: list[dict] | None = None
    ) -> tuple[str, list[dict]]:
        """Trim context to fit within token limit, always returning a tuple."""
        if not formatted_docs:
            return "", sources or []

        if self._count_tokens(formatted_docs) <= max_tokens:
            return formatted_docs, sources or []

        blocks = re.split(r"(?=^\[来源 \d+\])", formatted_docs, flags=re.MULTILINE)

        blocks = [b.strip() for b in blocks if b.strip()]

        if sources:
            while len(blocks) > 1:
                total = "\n\n".join(blocks)
                if self._count_tokens(total) <= max_tokens:
                    return total, sources[: len(blocks)]
                blocks.pop()
            return (blocks[0] if blocks else formatted_docs), sources[:1]
        else:
            while len(blocks) > 1:
                total = "\n\n".join(blocks)
                if self._count_tokens(total) <= max_tokens:
                    return total, []
                blocks.pop()
            return (blocks[0] if blocks else formatted_docs), []
