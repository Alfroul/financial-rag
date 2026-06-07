"""MCP Tool 实现：三个工具的内部逻辑。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.graph_retriever import GraphRetriever
    from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None
_graph_retriever: GraphRetriever | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        from src.api.deps import get_pipeline

        _pipeline = get_pipeline()
    return _pipeline


def _get_graph_retriever() -> GraphRetriever | None:
    global _graph_retriever
    if _graph_retriever is not None:
        return _graph_retriever

    from src.config import Config
    from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
    from src.graph.entity_matcher import EntityMatcher
    from src.graph.graph_retriever import GraphRetriever as GR
    from src.graph.graph_store import create_graph_store

    config = Config()
    if not config.graph.enabled:
        return None

    embedder = SiliconFlowEmbedder(
        api_key=config.api_key or "",
        model=config.embedding.model,
    )
    graph_store = create_graph_store(config.graph)
    entity_matcher = EntityMatcher(embedder)
    entity_matcher.build_index(graph_store.get_entities())
    _graph_retriever = GR(
        graph_store=graph_store,
        entity_matcher=entity_matcher,
        max_neighbors=config.graph.max_neighbors,
        max_depth=config.graph.max_depth,
    )
    return _graph_retriever


def financial_search(query: str, top_k: int = 5) -> str:
    """检索金融文档知识库，返回相关文档片段和来源信息。"""
    if not query or not query.strip():
        return "错误：查询不能为空"
    try:
        pipeline = _get_pipeline()
        results = pipeline._retriever.retrieve(query)
        if not results:
            return "未检索到相关文档。"
        results = results[:top_k]
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "未知来源")
            lines.append(f"[{i}] (分数: {r.score:.3f}, 来源: {source})\n{r.content}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("financial_search 执行失败")
        return f"检索失败：{e}"


def knowledge_graph_query(entity: str, relation: str = "") -> str:
    """查询金融实体关系图谱，返回匹配的三元组和邻居实体。"""
    if not entity or not entity.strip():
        return "错误：实体名称不能为空"
    try:
        retriever = _get_graph_retriever()
        if retriever is None:
            return "知识图谱未启用。请在 config.yaml 中设置 graph.enabled: true"
        triples = retriever.retrieve(entity)
        if not triples:
            return f"未找到与「{entity}」相关的知识图谱条目。"
        if relation:
            triples = [t for t in triples if t.relation == relation]
            if not triples:
                return f"未找到「{entity}」的关系类型为「{relation}」的条目。"
        lines = [f"- {t.head} —[{t.relation}]→ {t.tail}" for t in triples]
        header = f"实体「{entity}」的关联知识（共 {len(triples)} 条）："
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        logger.exception("knowledge_graph_query 执行失败")
        return f"图谱查询失败：{e}"


def financial_analysis(question: str) -> str:
    """完整的金融分析：检索相关文档并结合 LLM 生成分析报告。"""
    if not question or not question.strip():
        return "错误：问题不能为空"
    try:
        pipeline = _get_pipeline()
        result = pipeline.query(question)
        answer = result.get("answer", "未能生成回答。")
        sources = result.get("sources", [])
        if sources:
            src_lines = [f"  - {s}" for s in sources]
            answer += "\n\n参考来源：\n" + "\n".join(src_lines)
        return answer
    except Exception as e:
        logger.exception("financial_analysis 执行失败")
        return f"分析失败：{e}"
