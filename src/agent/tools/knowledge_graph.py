from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.agent.base_tool import BaseTool, ToolResult

if TYPE_CHECKING:
    from src.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class KnowledgeGraphTool(BaseTool):
    name = "knowledge_graph"
    description = (
        "查询金融实体关系图谱。当你需要对比两家公司、追踪指标变化、"
        "查找因果关联时使用。输入：entity（实体名称）。"
    )

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        self.graph_store = graph_store

    def run(self, entity: str = "", **kwargs: Any) -> ToolResult:
        if self.graph_store is None:
            return ToolResult(success=False, output="知识图谱未启用")
        if not entity or not entity.strip():
            return ToolResult(success=False, output="实体名称不能为空")
        try:
            triples = self.graph_store.query_neighbors(entity)
            if not triples:
                return ToolResult(success=True, output=f"未找到 {entity} 的关联知识")
            lines = [f"- {t.head} {t.relation} {t.tail}" for t in triples]
            return ToolResult(success=True, output="\n".join(lines))
        except Exception:
            logger.exception("KnowledgeGraphTool 图谱查询失败")
            return ToolResult(success=False, output="图谱查询失败，请稍后重试")
