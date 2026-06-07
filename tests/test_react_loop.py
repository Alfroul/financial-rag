"""ReAct 循环单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.base_tool import ToolResult
from src.agent.react import ReActAgent

# ── helpers ───────────────────────────────────────────────────────────


def _make_tool(name: str = "test_tool", output: str = "ok") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = f"Test tool: {name}"
    tool.run.return_value = ToolResult(success=True, output=output)
    return tool


def _make_llm(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.chat.side_effect = responses
    return llm


# ── test_react_single_step ────────────────────────────────────────────


class TestReActSingleStep:
    def test_no_action_returns_answer(self):
        """一步就出答案（无 Action 行）。"""
        llm = _make_llm(["Thought: 我知道答案。\n茅台2024年营收1680亿。"])
        agent = ReActAgent(llm=llm, tools=[], max_steps=6)

        answer = agent.run("茅台营收多少？")

        assert "1680亿" in answer
        assert llm.chat.call_count == 1


# ── test_react_multi_step ─────────────────────────────────────────────


class TestReActMultiStep:
    def test_multi_step_flow(self):
        """多步推理完整流程。"""
        tool = _make_tool("financial_search", "茅台营收1680亿")
        llm = _make_llm([
            'Thought: 需要检索\nAction: financial_search(query="茅台营收")',
            "Thought: 已有数据，直接回答。\n茅台营收1680亿。",
        ])
        agent = ReActAgent(llm=llm, tools=[tool], max_steps=6)

        answer = agent.run("茅台营收多少？")

        assert "1680亿" in answer
        assert tool.run.call_count == 1
        assert llm.chat.call_count == 2
        assert len(agent.get_steps()) == 2


# ── test_react_unknown_tool ───────────────────────────────────────────


class TestReActUnknownTool:
    def test_unknown_tool_returns_error_hint(self):
        """LLM 选了不存在的工具，返回错误提示。"""
        tool = _make_tool("financial_search", "data")
        llm = _make_llm([
            'Thought: 查一下\nAction: unknown_tool(query="test")',
            "Thought: 工具不存在，直接回答。\n无法查询。",
        ])
        agent = ReActAgent(llm=llm, tools=[tool], max_steps=6)

        agent.run("test")

        assert llm.chat.call_count == 2
        steps = agent.get_steps()
        assert "未知工具" in steps[0]["observation"]


# ── test_react_max_steps ──────────────────────────────────────────────


class TestReActMaxSteps:
    def test_max_steps_forces_summary(self):
        """超步数强制总结。"""
        tool = _make_tool("financial_search", "data")
        # 每步都返回 Action，永远不会自然结束
        responses = ['Thought: 查\nAction: financial_search(query="test")'] * 3
        llm = _make_llm(responses + ["最终总结回答。"])
        agent = ReActAgent(llm=llm, tools=[tool], max_steps=3)

        answer = agent.run("test")

        assert "最终总结" in answer
        # max_steps=3, 每步一次 LLM 调用 + 最后总结一次
        assert llm.chat.call_count == 4


# ── test_react_repeated_action ────────────────────────────────────────


class TestReActRepeatedAction:
    def test_repeated_action_triggers_warning(self):
        """重复动作检测触发。"""
        tool = _make_tool("financial_search", "data")
        llm = _make_llm([
            'Thought: 查\nAction: financial_search(query="test")',
            'Thought: 再查一次\nAction: financial_search(query="test")',
            "Thought: 收到提示，直接回答。\n结果是 data。",
        ])
        agent = ReActAgent(llm=llm, tools=[tool], max_steps=6)

        answer = agent.run("test")

        steps = agent.get_steps()
        # 第二次应该触发重复提示
        assert "重复" in steps[1]["observation"]
        assert "data" in answer


# ── test_parse_action_success ─────────────────────────────────────────


class TestParseAction:
    def test_parse_action_success(self):
        """正确解析 Action: tool(query=\"xxx\")。"""
        result = ReActAgent._parse_action(
            'Thought: 查一下\nAction: financial_search(query="贵州茅台营收")'
        )
        assert result is not None
        name, kwargs = result
        assert name == "financial_search"
        assert kwargs["query"] == "贵州茅台营收"

    def test_parse_action_no_action(self):
        """无 Action 行返回 None。"""
        result = ReActAgent._parse_action("Thought: 我知道答案。\n直接回答。")
        assert result is None

    def test_parse_action_malformed(self):
        """格式错误时兜底（整个字符串作为 query）。"""
        result = ReActAgent._parse_action(
            'Thought: 试试\nAction: financial_search(贵州茅台)'
        )
        assert result is not None
        name, kwargs = result
        assert name == "financial_search"
        assert "query" in kwargs


# ── test_parse_action_multiple_kwargs ─────────────────────────────────


class TestParseActionMultipleKwargs:
    def test_parse_multiple_kwargs(self):
        """解析多个关键字参数。"""
        result = ReActAgent._parse_action(
            'Action: tool(entity="茅台", query="营收")'
        )
        assert result is not None
        _, kwargs = result
        assert kwargs["entity"] == "茅台"
        assert kwargs["query"] == "营收"


# ── test_extract_answer ───────────────────────────────────────────────


class TestExtractAnswer:
    def test_strips_thought_prefix(self):
        """去掉 Thought: 前缀。"""
        answer = ReActAgent._extract_answer(
            "Thought: 分析完毕。\n茅台营收1680亿。"
        )
        assert "Thought:" not in answer
        assert "1680亿" in answer

    def test_plain_text(self):
        """纯文本直接返回。"""
        answer = ReActAgent._extract_answer("茅台营收1680亿。")
        assert "1680亿" in answer


# ── test_get_steps ────────────────────────────────────────────────────


class TestGetSteps:
    def test_steps_recorded(self):
        """步骤被正确记录。"""
        tool = _make_tool("financial_search", "data")
        llm = _make_llm([
            'Thought: 查\nAction: financial_search(query="test")',
            "Thought: 回答。\n结果是 data。",
        ])
        agent = ReActAgent(llm=llm, tools=[tool], max_steps=6)
        agent.run("test")

        steps = agent.get_steps()
        assert len(steps) == 2
        assert steps[0]["action"] != ""
        assert steps[0]["observation"] != ""
        assert steps[1]["action"] == ""
