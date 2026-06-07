from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.agent.base_tool import BaseTool, ToolResult

if TYPE_CHECKING:
    from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class FinancialSearchTool(BaseTool):
    name = "financial_search"
    description = (
        "检索金融文档知识库。当你需要查找公司财报数据、行业分析、"
        "经济指标时使用。输入：query（自然语言查询）。"
    )

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, output="查询不能为空")
        try:
            result = self.pipeline.query(query)
            return ToolResult(success=True, output=result["answer"])
        except Exception:
            logger.exception("FinancialSearchTool 检索失败")
            return ToolResult(success=False, output="检索失败，请稍后重试")
