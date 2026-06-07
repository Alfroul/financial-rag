"""Agent 集成测试 — RAGPipeline.agent_query() 端到端。"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline

# ── helpers ───────────────────────────────────────────────────────────


def _make_pipeline() -> RAGPipeline:
    """构建一个 mock RAGPipeline。"""
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = MagicMock()
    config = MagicMock()
    config.max_context_tokens = 4000
    return RAGPipeline(retriever=retriever, llm=llm, config=config)


# ── test_agent_query_basic ────────────────────────────────────────────


class TestAgentQueryBasic:
    def test_agent_query_returns_answer_and_steps(self):
        """agent_query() 端到端调用（mock LLM）。"""
        pipeline = _make_pipeline()
        # Mock the LLM to return a direct answer (no Action)
        pipeline._llm.chat.return_value = "Thought: 直接回答。\n茅台营收1680亿。"

        result = pipeline.agent_query(task="茅台营收多少？")

        assert "answer" in result
        assert "steps" in result
        assert "1680亿" in result["answer"]
        assert isinstance(result["steps"], list)

    def test_agent_query_empty_task(self):
        """空任务返回提示。"""
        pipeline = _make_pipeline()
        result = pipeline.agent_query(task="")
        assert "请输入" in result["answer"]
        assert result["steps"] == []


# ── test_agent_disabled ───────────────────────────────────────────────


class TestAgentDisabled:
    def test_agent_query_always_callable(self):
        """agent.enabled=false 时 agent_query() 仍可手动调用。"""
        pipeline = _make_pipeline()
        pipeline._llm.chat.return_value = "直接回答。"

        # agent_query 不依赖 agent.enabled 配置
        result = pipeline.agent_query(task="test")
        assert "answer" in result


# ── test_agent_with_graph ─────────────────────────────────────────────


class TestAgentWithGraph:
    def test_create_agent_includes_calculator_and_search(self):
        """_create_agent 组装的工具包含 financial_search 和 calculator。"""
        pipeline = _make_pipeline()
        agent = pipeline._create_agent()

        assert "financial_search" in agent.tools
        assert "calculator" in agent.tools

    def test_agent_multi_step_with_tools(self):
        """Agent 多步推理：先搜索再计算。"""
        pipeline = _make_pipeline()

        # Mock financial_search tool 返回
        responses = [
            'Thought: 需要查营收\nAction: financial_search(query="茅台营收")',
            'Thought: 需要算增长率\nAction: calculator(expression="1680 - 1453")',
            "Thought: 计算完成。\n茅台营收1680亿，增长227亿。",
        ]
        pipeline._llm.chat.side_effect = responses

        result = pipeline.agent_query(task="茅台营收和增长")

        assert "answer" in result
        assert len(result["steps"]) >= 2
