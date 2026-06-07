### 阶段 1：架构 Spike — Agent 骨架

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/agent/stage-1.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/agent/stage-1.md`、测试命令。

**目标**：铺设 Agent 模块的最小骨架——BaseTool 协议、ToolResult 数据模型、ReActAgent 空循环、配置入口、hello world 验证。不写任何业务逻辑。

**任务清单**：

- [x] 1.1 创建 `src/agent/` 模块目录
  - 创建 `src/agent/__init__.py`（导出 BaseTool, ToolResult, ReActAgent）
  - 创建 `src/agent/tools/__init__.py`

- [x] 1.2 创建 `src/agent/base_tool.py` — 工具协议
  ```python
  from abc import ABC, abstractmethod
  from dataclasses import dataclass, field


  @dataclass
  class ToolResult:
      success: bool
      output: str
      metadata: dict = field(default_factory=dict)


  class BaseTool(ABC):
      name: str
      description: str  # 给 LLM 看的工具说明

      @abstractmethod
      def run(self, **kwargs) -> ToolResult:
          ...
  ```

- [x] 1.3 创建 `src/agent/react.py` — ReActAgent 空壳
  ```python
  class ReActAgent:
      def __init__(self, llm, tools: list[BaseTool], max_steps: int = 6):
          self.llm = llm
          self.tools = {t.name: t for t in tools}
          self.max_steps = max_steps

      def run(self, task: str) -> str:
          """ReAct 主循环 — 阶段3实现"""
          raise NotImplementedError("阶段3实现")

      async def arun(self, task: str) -> str:
          """异步 ReAct 主循环 — 阶段3实现"""
          raise NotImplementedError("阶段3实现")
  ```

- [x] 1.4 创建 `src/agent/prompts.py` — ReAct Prompt 模板
  - `REACT_SYSTEM_PROMPT` — system prompt 模板（含占位符 `{tool_descriptions}`）
  - `REACT_FEW_SHOT` — 1-2 条 few-shot 示例（阶段3调优时细化）
  - `STOP_PROMPT` — 步数超限时的强制总结 prompt

- [x] 1.5 更新 `src/config.py` — 新增 AgentConfig
  ```python
  @dataclass(frozen=True)
  class AgentConfig:
      enabled: bool = False
      max_steps: int = 6
      model: str = ""  # 空=使用默认 LLM
  ```
  在 `load_config()` 中解析 config.yaml 的 `agent:` 块

- [x] 1.6 更新 `config.yaml` — 新增 agent 配置块
  ```yaml
  # Agent 配置（方案C）
  agent:
    enabled: false
    max_steps: 6
  ```

- [x] 1.7 更新 `CONTEXT.md` — 新增 Agent 相关术语（ReAct, BaseTool, ToolResult, ReActAgent, Action Parser, Step Limiter）

- [x] 1.8 创建 `docs/adr/0003-react-agent-handwritten.md`
  - 决策：手写 ReAct 而不用 LangChain/LlamaIndex
  - 原因：完全可控、面试能讲清每一行、工具数量有限不需要框架级发现
  - 代价：需要自己调 Prompt 格式稳定性、没有内置的 Memory 管理

- [x] 1.9 编写测试 `tests/test_agent_skeleton.py`
  - `test_tool_result_creation` — ToolResult dataclass 能正常创建
  - `test_base_tool_interface` — BaseTool 是抽象类，不能直接实例化
  - `test_custom_tool` — 继承 BaseTool 的子类能正常实例化并调用 run()
  - `test_react_agent_init` — ReActAgent 能用 llm + tools 初始化
  - `test_react_agent_run_not_implemented` — run() 在骨架阶段抛 NotImplementedError
  - `test_agent_config` — AgentConfig 从 yaml 正确加载
  - `test_config_yaml_agent` — config.yaml 包含 agent 配置块

- [x] 1.10 验证
  - `ruff check src/agent/ src/config.py` 无错误
  - `pytest tests/test_agent_skeleton.py -v` 全部通过
  - `streamlit run app.py` 能正常启动（agent.enabled=false 不影响现有功能）

**验收标准**：
- `src/agent/` 模块存在，BaseTool 和 ToolResult 定义完整
- ReActAgent 骨架存在，run() 签名已定义
- config.yaml 包含 agent 配置块
- 所有骨架测试通过
- 现有功能不受影响

**完成确认**：

- [x] 阶段 1 全部任务完成，已通过验收标准
