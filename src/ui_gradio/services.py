"""Service layer for Gradio UI — builds RAG pipeline without Streamlit dependencies."""

from __future__ import annotations

from dataclasses import replace

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

_vectorstore: ChromaStore | None = None
_bm25: BM25Retriever | None = None
_reranker: LocalRreranker | None = None


def get_vectorstore() -> ChromaStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = ChromaStore(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name,
        )
    return _vectorstore


def get_bm25() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever(get_vectorstore())
    return _bm25


def get_reranker() -> LocalRreranker:
    global _reranker
    if _reranker is None:
        _reranker = LocalRreranker()
    return _reranker


def clear_bm25() -> None:
    global _bm25
    _bm25 = None


def build_pipeline(
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    strategy: str,
    top_k: int,
    score_threshold: float,
    reranker_enabled: bool,
    reranker_top_n: int,
    query_rewrite: bool,
    cache_enabled: bool,
    self_correction: bool,
) -> RAGPipeline | SelfCorrectingPipeline:
    embedder = SiliconFlowEmbedder(api_key=api_key, model=config.embedding.model)
    store = get_vectorstore()

    if config.hybrid.enabled and strategy in ("hybrid", "bm25"):
        base = Retriever(
            embedder, store,
            RetrieverConfig(top_k=config.hybrid.vector_fetch_k, score_threshold=0.0),
        )
        retriever: Retriever | HybridRetriever = HybridRetriever(
            retriever=base,
            bm25_retriever=get_bm25(),
            config=replace(config.hybrid, strategy=strategy),
            score_threshold=score_threshold,
        )
    else:
        retriever = Retriever(
            embedder, store,
            RetrieverConfig(top_k=top_k, score_threshold=score_threshold),
        )

    llm = MimoLLM(
        api_key=api_key,
        model=model,
        base_url=config.llm.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    rk = get_reranker() if reranker_enabled else None
    rk_cfg = (
        replace(config.reranker, top_n=reranker_top_n)
        if (reranker_enabled and config.reranker is not None)
        else None
    )

    qr = QueryRewriter(llm=llm) if query_rewrite else None

    cache = None
    if cache_enabled:
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
        em = EntityMatcher(embedder)
        em.build_index(graph_store.get_entities())
        graph_retriever = GraphRetriever(
            graph_store=graph_store,
            entity_matcher=em,
            max_neighbors=config.graph.max_neighbors,
            max_depth=config.graph.max_depth,
        )

    tracer = None
    if config.langfuse.enabled:
        tracer = LangfuseTracer(
            enabled=True,
            public_key=config.langfuse.public_key,
            secret_key=config.langfuse.secret_key,
            host=config.langfuse.host,
        )

    pipeline = RAGPipeline(
        retriever, llm, config.rag,
        reranker=rk,
        reranker_config=rk_cfg,
        query_rewriter=qr,
        cache=cache,
        fact_cache=fact_cache,
        fact_extractor=fact_extractor,
        fact_cache_threshold=config.fact_cache.similarity_threshold,
        graph_store=graph_store,
        graph_config=config.graph,
        graph_retriever=graph_retriever,
        tracer=tracer,
    )

    if self_correction:
        return SelfCorrectingPipeline(
            pipeline=pipeline,
            config=config.self_correction,
            api_key=config.api_key,
            base_url=config.self_correction.verifier_base_url,
            model=config.self_correction.verifier_model,
        )

    return pipeline
