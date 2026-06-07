"""FastAPI 依赖注入 — Pipeline 和 Store 实例的构建。"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import Config, RetrieverConfig
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

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_config() -> Config:
    return Config()


@lru_cache(maxsize=1)
def _get_store() -> ChromaStore:
    config = _get_config()
    return ChromaStore(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
    )


@lru_cache(maxsize=1)
def _get_embedder() -> SiliconFlowEmbedder:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("API Key 未配置，请设置环境变量或 .env 文件")
    return SiliconFlowEmbedder(
        api_key=api_key,
        model=config.embedding.model,
    )


@lru_cache(maxsize=1)
def _get_llm() -> MimoLLM:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("MIMO_API_KEY 未配置，请设置环境变量或 .env 文件")
    return MimoLLM(
        api_key=api_key,
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def get_store() -> ChromaStore:
    return _get_store()


def get_pipeline() -> RAGPipeline:
    config = _get_config()
    embedder = _get_embedder()
    store = _get_store()
    llm = _get_llm()

    base_retriever = Retriever(
        embedder,
        store,
        RetrieverConfig(
            top_k=config.hybrid.vector_fetch_k,
            score_threshold=0.0,
        ),
    )
    bm25_retriever = BM25Retriever(store)
    retriever = HybridRetriever(
        retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        config=config.hybrid,
        score_threshold=config.retriever.score_threshold,
    )

    reranker = LocalRreranker() if config.reranker.enabled else None
    reranker_config = config.reranker if config.reranker.enabled else None

    query_rewriter = QueryRewriter(llm=llm) if config.rag.query_rewrite else None

    fact_cache = None
    fact_extractor = None
    if config.fact_cache.enabled:
        fact_cache = FactCacheStore(
            embedder=embedder,
            collection_name=config.fact_cache.collection_name,
        )
        fact_extractor = FactExtractor(api_key=config.api_key)

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

    tracer = None
    if config.langfuse.enabled:
        tracer = LangfuseTracer(
            enabled=config.langfuse.enabled,
            public_key=config.langfuse.public_key,
            secret_key=config.langfuse.secret_key,
            host=config.langfuse.host,
        )

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        config=config.rag,
        reranker=reranker if reranker else None,
        reranker_config=reranker_config,
        query_rewriter=query_rewriter,
        fact_cache=fact_cache,
        fact_extractor=fact_extractor,
        fact_cache_threshold=config.fact_cache.similarity_threshold,
        graph_store=graph_store,
        graph_config=config.graph,
        graph_retriever=graph_retriever,
        tracer=tracer,
    )
