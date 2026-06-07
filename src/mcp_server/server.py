"""MCP Server — 将 RAG 系统能力暴露为 MCP Tools。

启动方式：
    python -m src.mcp_server.server          # 默认 stdio
    python -m src.mcp_server.server --sse    # SSE 模式
"""

from __future__ import annotations

import argparse
import logging

from mcp.server.fastmcp import FastMCP

from src.mcp_server import tools

logger = logging.getLogger(__name__)

mcp = FastMCP("financial-rag")


@mcp.tool()
def financial_search(query: str, top_k: int = 5) -> str:
    """检索金融文档知识库。当你需要查找公司财报数据、行业分析报告、
    经济指标等文档片段时使用。返回相关文档内容及其来源。

    Args:
        query: 查询文本，如"贵州茅台2024年营收"
        top_k: 返回数量，默认 5
    """
    return tools.financial_search(query, top_k)


@mcp.tool()
def knowledge_graph_query(entity: str, relation: str = "") -> str:
    """查询金融实体关系知识图谱。当你需要查找实体之间的关联关系、
    对比指标、因果链路时使用。

    Args:
        entity: 实体名称，如"贵州茅台"
        relation: 关系类型过滤（可选），如"同比增长"
    """
    return tools.knowledge_graph_query(entity, relation)


@mcp.tool()
def financial_analysis(question: str) -> str:
    """完整的金融分析问答。检索相关文档后由 LLM 生成结构化分析报告。
    适合需要综合推理、对比分析、趋势解读的复杂问题。

    Args:
        question: 分析问题，如"对比茅台和五粮液2024年的盈利能力"
    """
    return tools.financial_analysis(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial RAG MCP Server")
    parser.add_argument("--sse", action="store_true", help="使用 SSE transport（默认 stdio）")
    parser.add_argument("--host", default=None, help="SSE 模式的监听地址")
    parser.add_argument("--port", type=int, default=None, help="SSE 模式的监听端口")
    args = parser.parse_args()

    try:
        from src.config import Config

        config = Config()
        host = args.host or config.mcp_server.host
        port = args.port or config.mcp_server.port
    except Exception:
        host = args.host or "localhost"
        port = args.port or 8080

    transport = "sse" if args.sse else "stdio"
    logger.info("启动 MCP Server，transport=%s, host=%s, port=%d", transport, host, port)

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
