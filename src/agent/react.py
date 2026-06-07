"""ReAct Agent — Thought-Action-Observation 循环实现。"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.agent.base_tool import BaseTool

if TYPE_CHECKING:
    from src.observability.langfuse_tracer import LangfuseTracer

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.+?)\)", re.DOTALL | re.IGNORECASE)
_KWARG_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# 边界情况模式
_CASUAL_CHAT_RE = re.compile(
    r"^(你好|hello|hi|嗨|哈喽|hey|在吗|你是谁|干嘛|ok|好的|嗯|谢谢|thanks|bye|再见|拜拜)[!！。.~～]*$",
    re.IGNORECASE,
)
_KNOWLEDGE_KEYWORDS = {
    "是什么", "什么是", "定义", "含义", "意思", "解释",
    "什么叫", "指的是", "全称", "缩写",
}

_SYSTEM_PROMPT = """\
你是一个金融分析助手，可以使用以下工具回答问题：

{tool_descriptions}

## 输出格式（必须严格遵守）
每一步输出必须包含 Thought，可选 Action：
Thought: <你的思考过程>
Action: <工具名>(<参数名>="<参数值>")

## 工具选择指南
- 需要检索公司财报、行业数据、经济指标 → financial_search
- 需要查询实体关系、对比分析、因果链 → knowledge_graph
- 需要计算比率、增长率、复合增长率 → calculator
- 可以组合使用多个工具，每步只调用一个工具

## 直接回答条件
当已有足够信息时，不要输出 Action，直接给出最终答案。答案应：
- 包含具体数字和数据来源
- 有明确的分析结论
- 结构清晰，分点说明"""

# 多个 few-shot 示例，覆盖不同工具组合
_FEW_SHOT_EXAMPLES = [
    # 示例1: financial_search + calculator
    {
        "task": "贵州茅台2024年的净利润率是多少？",
        "turns": [
            {
                "assistant": (
                    "Thought: 用户想知道茅台的净利润率，我需要先检索营收和净利润数据。"
                    '\nAction: financial_search(query="贵州茅台 2024年 营收 净利润")'
                ),
                "observation": "贵州茅台2024年营业收入1680亿元，净利润862亿元。",
            },
            {
                "assistant": (
                    "Thought: 已获得数据，现在计算净利润率。"
                    '\nAction: calculator(expression="862 / 1680 * 100")'
                ),
                "observation": "51.31",
            },
            {
                "assistant": (
                    "Thought: 计算完成，可以给出最终答案。"
                    "贵州茅台2024年净利润率为51.31%，计算方式为净利润862亿元除以营收1680亿元。"
                    "这一比率在白酒行业中处于领先水平，体现了茅台强大的盈利能力和品牌溢价。"
                ),
            },
        ],
    },
    # 示例2: financial_search + knowledge_graph（对比分析）
    {
        "task": "对比茅台和五粮液的营收增速，哪家增长更快？",
        "turns": [
            {
                "assistant": (
                    "Thought: 这是一个对比分析任务，需要检索两家公司的营收数据。先查茅台。"
                    '\nAction: financial_search(query="贵州茅台 2024年 营收 增速")'
                ),
                "observation": "贵州茅台2024年营业收入1680亿元，同比增长15.66%。",
            },
            {
                "assistant": (
                    "Thought: 已获得茅台数据，现在查五粮液。"
                    '\nAction: financial_search(query="五粮液 2024年 营收 增速")'
                ),
                "observation": "五粮液2024年营业收入832亿元，同比增长12.3%。",
            },
            {
                "assistant": (
                    "Thought: 两家数据都有了，可以进行对比。茅台增速15.66%高于五粮液的12.3%。"
                    "对比结果：贵州茅台2024年营收增速15.66%高于五粮液的12.3%，增速领先3.36个百分点。"
                    "从绝对规模看，茅台营收1680亿元也远超五粮液的832亿元。"
                ),
            },
        ],
    },
    # 示例3: 直接回答（无需工具）
    {
        "task": "GDP是什么？",
        "turns": [
            {
                "assistant": (
                    "Thought: 这是一个概念解释类问题，不需要使用工具，可以直接回答。"
                    "GDP（Gross Domestic Product）即国内生产总值，"
                    "是指一个国家或地区在一定时期内生产的"
                    "所有最终商品和服务的市场价值总和。"
                    "它是衡量一个国家经济规模和经济健康状况的核心指标。"
                    "GDP的计算方式包括生产法、收入法和支出法三种。"
                ),
            },
        ],
    },
]


def _build_few_shot_messages() -> list[dict[str, str]]:
    """构建多组 few-shot 对话消息。"""
    messages: list[dict[str, str]] = []
    for example in _FEW_SHOT_EXAMPLES:
        task: str = example["task"]  # type: ignore[assignment]
        messages.append({"role": "user", "content": task})
        for turn in example["turns"]:
            assistant: str = turn["assistant"]  # type: ignore[assignment, index]
            messages.append({"role": "assistant", "content": assistant})
            if "observation" in turn:
                obs: str = turn["observation"]  # type: ignore[assignment, index]
                messages.append({"role": "user", "content": f"Observation: {obs}"})
    return messages


class ReActAgent:
    """手写 ReAct 循环，核心约 60 行。"""

    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        max_steps: int = 6,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self.max_steps = max_steps
        self._steps: list[dict[str, str]] = []
        self._tracer = tracer

    @staticmethod
    def _classify_task(task: str) -> str:
        """预分类任务类型，处理边界情况。

        Returns:
            "empty" | "too_long" | "casual" | "knowledge" | "normal"
        """
        stripped = task.strip()
        if not stripped:
            return "empty"
        if len(stripped) > 500:
            return "too_long"
        if _CASUAL_CHAT_RE.match(stripped):
            return "casual"
        # 纯知识问答检测：短问题 + 包含知识关键词 + 无明确金融实体
        if len(stripped) < 30:
            for kw in _KNOWLEDGE_KEYWORDS:
                if kw in stripped:
                    return "knowledge"
        return "normal"

    def _handle_edge_case(self, task: str, task_type: str) -> str | None:
        """同步处理边界情况，返回直接回答或 None（需要进入 ReAct 循环）。"""
        if task_type == "empty":
            return "请输入您的问题，我可以帮您分析金融数据、查询公司财报、进行指标计算等。"
        if task_type == "too_long":
            logger.warning("任务过长（%d字），已截断至500字", len(task))
            return None
        if task_type == "casual":
            return "你好！我是金融分析助手，可以帮您查询公司财报、分析行业趋势、计算财务指标等。请问有什么可以帮您的？"
        if task_type == "knowledge":
            try:
                return self._call_llm([
                    {"role": "system", "content": "你是一个金融知识问答助手，请简洁准确地回答以下问题。"},
                    {"role": "user", "content": task},
                ])
            except Exception:
                return None
        return None

    async def _ahandle_edge_case(self, task: str, task_type: str) -> str | None:
        """异步处理边界情况。"""
        if task_type == "empty":
            return "请输入您的问题，我可以帮您分析金融数据、查询公司财报、进行指标计算等。"
        if task_type == "too_long":
            logger.warning("任务过长（%d字），已截断至500字", len(task))
            return None
        if task_type == "casual":
            return "你好！我是金融分析助手，可以帮您查询公司财报、分析行业趋势、计算财务指标等。请问有什么可以帮您的？"
        if task_type == "knowledge":
            try:
                return await self._acall_llm([
                    {"role": "system", "content": "你是一个金融知识问答助手，请简洁准确地回答以下问题。"},
                    {"role": "user", "content": task},
                ])
            except Exception:
                return None
        return None

    def _run_loop(self, task: str, call_llm: Any, trace_id: str | None = None) -> str:
        """ReAct 主循环（同步/异步共用）。

        Args:
            task: 用户任务（已截断）。
            call_llm: 同步或异步的 LLM 调用函数。
            trace_id: Langfuse trace ID（可选）。
        """
        messages = self._build_initial_messages(task)
        recent_actions: list[tuple[str, str]] = []
        tracer = self._tracer

        for step_idx in range(self.max_steps):
            response = call_llm(messages)
            action = self._parse_action(response)

            if action is None:
                answer = self._extract_answer(response)
                self._steps.append({"thought": response, "action": "", "observation": ""})
                return answer

            tool_name, tool_input = action
            self._steps.append({
                "thought": response,
                "action": f"{tool_name}({tool_input})",
                "observation": "",
            })

            # 重复动作检测
            recent_actions.append((tool_name, str(tool_input)))
            if len(recent_actions) > 3:
                recent_actions.pop(0)
            if len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
                observation = "你已经在重复相同操作，请直接给出回答。"
            elif tool_name not in self.tools:
                observation = f"未知工具 '{tool_name}'，可用：{list(self.tools.keys())}"
            else:
                # trace tool call as span
                tool_span = None
                if tracer is not None and tracer.enabled and trace_id is not None:
                    tool_span = tracer.start_span(trace_id, f"tool:{tool_name}", input_data=tool_input)
                result = self.tools[tool_name].run(**tool_input)
                observation = result.output
                if tool_span is not None and tracer is not None:
                    tracer.end_span(tool_span, output_data=observation, metadata={"step": step_idx})

            self._steps[-1]["observation"] = observation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # 超步数强制总结
        messages.append({"role": "user", "content": "已达到最大推理步数，请基于已有信息给出最终回答。"})
        return call_llm(messages)

    def run(self, task: str) -> str:
        """同步 ReAct 主循环。"""
        self._steps.clear()
        task_type = self._classify_task(task)

        tracer = self._tracer
        trace_id: str | None = None
        if tracer is not None and tracer.enabled:
            trace_id = tracer.start_trace(task)

        edge_result = self._handle_edge_case(task, task_type)
        if edge_result is not None:
            self._steps.append({"thought": f"[边界处理: {task_type}]", "action": "", "observation": ""})
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output=edge_result)
            return edge_result
        if task_type == "too_long":
            task = task[:500]
        answer = self._run_loop(task, self._call_llm, trace_id=trace_id)
        if tracer is not None and trace_id is not None:
            tracer.end_trace(trace_id, output=answer)
        return answer

    async def arun(self, task: str) -> str:
        """异步 ReAct 主循环。"""
        self._steps.clear()
        task_type = self._classify_task(task)

        tracer = self._tracer
        trace_id: str | None = None
        if tracer is not None and tracer.enabled:
            trace_id = tracer.start_trace(task)

        edge_result = await self._ahandle_edge_case(task, task_type)
        if edge_result is not None:
            self._steps.append({"thought": f"[边界处理: {task_type}]", "action": "", "observation": ""})
            if tracer is not None and trace_id is not None:
                tracer.end_trace(trace_id, output=edge_result)
            return edge_result
        if task_type == "too_long":
            task = task[:500]
        answer = self._run_loop(task, self._acall_llm, trace_id=trace_id)
        if tracer is not None and trace_id is not None:
            tracer.end_trace(trace_id, output=answer)
        return answer

    def get_steps(self) -> list[dict[str, str]]:
        """返回已记录的推理步骤（供 API/UI 使用）。"""
        return list(self._steps)

    def _build_initial_messages(self, task: str) -> list[dict[str, str]]:
        """组装 system prompt + few-shot + 用户任务。"""
        tool_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )
        system = _SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        # 添加多组 few-shot 示例
        messages.extend(_build_few_shot_messages())
        # 添加用户任务
        messages.append({"role": "user", "content": task})
        return messages

    @staticmethod
    def _parse_action(response: str) -> tuple[str, dict[str, str]] | None:
        """从 LLM 输出中解析 Action 行。

        Returns:
            (tool_name, {param: value}) 或 None（无 Action 行）。
        """
        match = _ACTION_RE.search(response)
        if match is None:
            return None

        tool_name = match.group(1)
        raw_args = match.group(2).strip()

        kwargs: dict[str, str] = {}
        for m in _KWARG_RE.finditer(raw_args):
            kwargs[m.group(1)] = m.group(2)

        if not kwargs:
            kwargs = {"query": raw_args}

        return tool_name, kwargs

    @staticmethod
    def _extract_answer(response: str) -> str:
        """从 LLM 输出中提取最终答案（去掉 Thought: 前缀）。"""
        lines = response.strip().splitlines()
        answer_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Thought:"):
                continue
            answer_lines.append(stripped)
        return "\n".join(answer_lines).strip() or response.strip()

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """同步调用 LLM。"""
        return self.llm.chat("", messages)

    async def _acall_llm(self, messages: list[dict[str, str]]) -> str:
        """异步调用 LLM。"""
        return await self.llm.achat("", messages)
