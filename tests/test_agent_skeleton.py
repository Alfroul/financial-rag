import pytest
import yaml

from src.agent import BaseTool, ReActAgent, ToolResult
from src.config import AgentConfig


def test_tool_result_creation():
    """ToolResult dataclass 能正常创建"""
    result = ToolResult(success=True, output="test output")
    assert result.success is True
    assert result.output == "test output"
    assert result.metadata == {}


def test_tool_result_with_metadata():
    """ToolResult 支持自定义 metadata"""
    result = ToolResult(success=False, output="error", metadata={"code": 404})
    assert result.success is False
    assert result.metadata["code"] == 404


def test_base_tool_interface():
    """BaseTool 是抽象类，不能直接实例化"""
    with pytest.raises(TypeError):
        BaseTool()


def test_custom_tool():
    """继承 BaseTool 的子类能正常实例化并调用 run()"""

    class DummyTool(BaseTool):
        name = "dummy"
        description = "A dummy tool for testing"

        def run(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="dummy result")

    tool = DummyTool()
    assert tool.name == "dummy"
    assert tool.description == "A dummy tool for testing"
    result = tool.run()
    assert result.success is True
    assert result.output == "dummy result"


def test_react_agent_init():
    """ReActAgent 能用 llm + tools 初始化"""

    class DummyTool(BaseTool):
        name = "dummy"
        description = "test"

        def run(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="ok")

    agent = ReActAgent(llm=None, tools=[DummyTool()], max_steps=3)
    assert agent.max_steps == 3
    assert "dummy" in agent.tools


def test_react_agent_run_returns_answer():
    """run() 使用 mock LLM 返回直接答案（无 Action）"""

    class _MockLLM:
        def chat(self, system_prompt, messages):
            return "Thought: 这是一个测试任务，我可以直接回答。\n测试回答内容。"

    agent = ReActAgent(llm=_MockLLM(), tools=[], max_steps=6)
    answer = agent.run("test task")
    assert "测试回答" in answer
    assert len(agent.get_steps()) == 1


def test_agent_config():
    """AgentConfig 从 yaml 正确加载"""
    config = AgentConfig(enabled=True, max_steps=10, model="gpt-4")
    assert config.enabled is True
    assert config.max_steps == 10
    assert config.model == "gpt-4"


def test_agent_config_defaults():
    """AgentConfig 默认值正确"""
    config = AgentConfig()
    assert config.enabled is False
    assert config.max_steps == 6
    assert config.model == ""


def test_config_yaml_agent():
    """config.yaml 包含 agent 配置块"""
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert "agent" in raw
    assert "enabled" in raw["agent"]
    assert "max_steps" in raw["agent"]
