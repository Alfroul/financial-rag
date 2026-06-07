from __future__ import annotations

from dataclasses import replace

import streamlit as st

from src.cache import QueryCache
from src.config import Config, RetrieverConfig
from src.correction.pipeline import SelfCorrectingPipeline
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.fact_cache.store import FactCacheStore
from src.fact_extractor.extractor import FactExtractor
from src.generator.mimo_llm import MimoLLM
from src.generator.query_rewriter import QueryRewriter
from src.graph.entity_matcher import EntityMatcher
from src.graph.graph_retriever import GraphRetriever
from src.graph.graph_store import create_graph_store
from src.observability.langfuse_tracer import LangfuseTracer
from src.rag_pipeline import RAGPipeline
from src.reranker.local_reranker import LocalRreranker
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

config = Config()


@st.cache_resource
def _init_vectorstore() -> ChromaStore:
    return ChromaStore(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
    )


@st.cache_resource
def _init_bm25_retriever() -> BM25Retriever:
    store = _init_vectorstore()
    return BM25Retriever(store)


@st.cache_resource
def _init_embedder(_api_key: str) -> SiliconFlowEmbedder:
    return SiliconFlowEmbedder(
        api_key=_api_key,
        model=config.embedding.model,
    )


@st.cache_resource
def _init_llm(_api_key: str, _model: str, _temperature: float, _max_tokens: int) -> MimoLLM:
    return MimoLLM(
        api_key=_api_key,
        model=_model,
        base_url=config.llm.base_url,
        temperature=_temperature,
        max_tokens=_max_tokens,
    )


def _build_rag_pipeline(api_key: str) -> RAGPipeline | SelfCorrectingPipeline:
    embedder = _init_embedder(api_key)
    store = _init_vectorstore()

    strategy = st.session_state.retrieve_strategy
    retriever: Retriever | HybridRetriever
    if config.hybrid.enabled and strategy in ("hybrid", "bm25"):
        base_retriever = Retriever(
            embedder,
            store,
            RetrieverConfig(
                top_k=config.hybrid.vector_fetch_k,
                score_threshold=0.0,
            ),
        )
        bm25_retriever = _init_bm25_retriever()
        effective_config = replace(config.hybrid, strategy=strategy)
        retriever = HybridRetriever(
            retriever=base_retriever,
            bm25_retriever=bm25_retriever,
            config=effective_config,
            score_threshold=st.session_state.score_threshold,
        )
    else:
        retriever = Retriever(  # type: ignore[assignment]
            embedder,
            store,
            RetrieverConfig(
                top_k=st.session_state.top_k,
                score_threshold=st.session_state.score_threshold,
            ),
        )

    llm = _init_llm(
        api_key,
        st.session_state.model,
        st.session_state.temperature,
        st.session_state.max_tokens,
    )

    reranker = None
    if st.session_state.reranker_enabled:
        reranker = LocalRreranker()

    query_rewriter = None
    if st.session_state.query_rewrite:
        query_rewriter = QueryRewriter(llm=llm)

    cache = None
    if getattr(st.session_state, "cache_enabled", False):
        cache = QueryCache(
            embedder=embedder,
            similarity_threshold=config.cache.similarity_threshold,
            max_size=config.cache.max_size,
        )

    fact_cache = None
    fact_extractor = None
    if config.fact_cache.enabled:
        fact_cache = FactCacheStore(
            embedder=embedder,
            collection_name=config.fact_cache.collection_name,
        )
        fact_extractor = FactExtractor(api_key=api_key)

    graph_store = None
    graph_retriever = None
    if config.graph.enabled:
        graph_store = create_graph_store(config.graph)
        entity_matcher = EntityMatcher(embedder)
        entity_matcher.build_index(graph_store.get_entities())
        graph_retriever = GraphRetriever(
            graph_store=graph_store,
            entity_matcher=entity_matcher,
            max_neighbors=config.graph.max_neighbors,
            max_depth=config.graph.max_depth,
        )

    reranker_config = None
    if st.session_state.reranker_enabled and config.reranker is not None:
        reranker_config = replace(config.reranker, top_n=st.session_state.reranker_top_n)

    tracer = None
    if config.langfuse.enabled:
        tracer = LangfuseTracer(
            enabled=config.langfuse.enabled,
            public_key=config.langfuse.public_key,
            secret_key=config.langfuse.secret_key,
            host=config.langfuse.host,
        )

    pipeline = RAGPipeline(
        retriever,
        llm,
        config.rag,
        reranker=reranker,
        reranker_config=reranker_config,
        query_rewriter=query_rewriter,
        cache=cache,
        fact_cache=fact_cache,
        fact_extractor=fact_extractor,
        fact_cache_threshold=config.fact_cache.similarity_threshold,
        graph_store=graph_store,
        graph_config=config.graph,
        graph_retriever=graph_retriever,
        tracer=tracer,
    )

    if getattr(st.session_state, "self_correction_enabled", False):
        return _wrap_self_correction(pipeline, api_key)

    return pipeline


def _wrap_self_correction(pipeline: RAGPipeline, api_key: str) -> SelfCorrectingPipeline:
    return SelfCorrectingPipeline(
        pipeline=pipeline,
        config=config.self_correction,
        api_key=config.api_key,
        base_url=config.self_correction.verifier_base_url,
        model=config.self_correction.verifier_model,
    )
