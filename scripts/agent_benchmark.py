#!/usr/bin/env python
"""Agent Benchmark — 量化评测 ReAct Agent 在多步推理任务上的表现。

用法:
    python scripts/agent_benchmark.py
    python scripts/agent_benchmark.py --api-key YOUR_KEY
    python scripts/agent_benchmark.py --max-tasks 5          # 只跑前5条
    python scripts/agent_benchmark.py --output report.md     # 保存报告

输出:
    Markdown 格式评测报告，包含工具使用分布、步数统计、答案质量评分。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.base_tool import BaseTool, ToolResult
from src.agent.react import ReActAgent
from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.financial_search import FinancialSearchTool
from src.agent.tools.knowledge_graph import KnowledgeGraphTool

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_benchmark")

EVAL_PATH = _PROJECT_ROOT / "data" / "eval" / "agent_eval.jsonl"
DEFAULT_PERSIST_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")
DEFAULT_COLLECTION = "financial_docs"


# ---------------------------------------------------------------------------
# MiMo LLM（OpenAI 兼容，小米大模型平台）
# ---------------------------------------------------------------------------


class MimoLLM:
    """通过 OpenAI 兼容 API 调用 MiMo 模型。"""

    def __init__(
        self,
        api_key: str,
        model: str = "mimo-v2-pro",
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        all_msgs: list[dict[str, str]] = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        all_msgs.extend(messages)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": all_msgs,
                        "temperature": temperature if temperature is not None else self._temperature,
                        "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("API 调用失败 (%d/3): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"API 连续 3 次调用失败: {self._model}")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """单条评测任务的结果。"""

    task: str
    category: str
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    step_count: int = 0
    elapsed_ms: float = 0.0
    quality_score: int = 0
    quality_reason: str = ""
    error: str = ""
    steps_detail: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """完整评测报告。"""

    results: list[TaskResult] = field(default_factory=list)
    total_tasks: int = 0
    success_count: int = 0
    error_count: int = 0


# ---------------------------------------------------------------------------
# 评测数据集加载
# ---------------------------------------------------------------------------


def _load_eval_dataset(path: Path) -> list[dict]:
    """加载 JSONL 格式评测数据集。"""
    if not path.exists():
        logger.error("评测数据集不存在: %s", path)
        sys.exit(1)
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    logger.info("加载了 %d 条评测任务: %s", len(tasks), path)
    return tasks


# ---------------------------------------------------------------------------
# Agent 构建
# ---------------------------------------------------------------------------


def _build_agent(llm: MimoLLM, max_steps: int = 6) -> ReActAgent:
    """构建评测用的 ReActAgent（不依赖 RAGPipeline）。"""

    class MockSearchTool(BaseTool):
        """模拟 financial_search 工具，返回预设回答。"""

        name = "financial_search"
        description = "检索金融文档知识库，查询公司财报、行业分析、经济指标等信息。"

        def run(self, **kwargs) -> ToolResult:
            query = kwargs.get("query", "")
            if not query:
                return ToolResult(success=False, output="请提供查询关键词。")
            # 使用 LLM 生成模拟检索结果
            try:
                resp = llm.chat(
                    "",
                    [
                        {
                            "role": "user",
                            "content": f"请基于你的知识，模拟金融文档检索系统，对以下查询提供尽可能详细的回答（包含具体数字和数据）：\n{query}",
                        }
                    ],
                )
                return ToolResult(success=True, output=resp)
            except Exception as e:
                return ToolResult(success=False, output=f"检索失败: {e}")

    class MockKGTool(BaseTool):
        """模拟 knowledge_graph 工具。"""

        name = "knowledge_graph"
        description = "查询金融实体关系图谱，获取公司间的对比、关联、因果关系等信息。"

        def run(self, **kwargs) -> ToolResult:
            entity = kwargs.get("entity", "") or kwargs.get("query", "")
            if not entity:
                return ToolResult(success=False, output="请提供实体名称。")
            try:
                resp = llm.chat(
                    "",
                    [
                        {
                            "role": "user",
                            "content": (
                                "请基于你的知识，模拟知识图谱查询，"
                                f"以三元组格式列出与「{entity}」相关的金融实体关系"
                                "（格式：- 主体 关系 客体）："
                            ),
                        }
                    ],
                )
                return ToolResult(success=True, output=resp)
            except Exception as e:
                return ToolResult(success=False, output=f"图谱查询失败: {e}")

    tools: list[BaseTool] = [
        MockSearchTool(),
        MockKGTool(),
        CalculatorTool(),
    ]
    return ReActAgent(llm=llm, tools=tools, max_steps=max_steps)


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


_JUDGE_PROMPT = """\
你是一个金融分析评测专家。请对以下 Agent 回答进行评分。

## 评分任务
任务描述：{task}
评测标准：{criteria}

## Agent 回答
{answer}

## Agent 使用的工具
{tools}

## Agent 推理步数
{steps}

## 评分规则
请从 1-5 分进行评分：
- 5分：完全满足所有评测标准，回答准确、全面、有数据支撑
- 4分：满足大部分评测标准，回答基本准确
- 3分：满足部分评测标准，回答有一定参考价值
- 2分：仅满足少数评测标准，回答不够准确或完整
- 1分：未能满足评测标准，回答错误或无意义

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{"score": <1-5>, "reason": "<评分理由，50字以内>"}}"""


def _judge_answer(
    llm: MimoLLM,
    task: str,
    criteria: list[str],
    answer: str,
    tools: list[str],
    steps: int,
) -> tuple[int, str]:
    """使用 LLM 对答案质量进行评分。"""
    prompt = _JUDGE_PROMPT.format(
        task=task,
        criteria="、".join(criteria),
        answer=answer[:1500],
        tools=", ".join(tools) if tools else "无",
        steps=steps,
    )
    try:
        resp = llm.chat("", [{"role": "user", "content": prompt}])
        # 解析 JSON
        resp = resp.strip()
        # 处理可能的 markdown 代码块
        if resp.startswith("```"):
            resp = resp.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(resp)
        return int(data.get("score", 0)), str(data.get("reason", ""))
    except Exception as e:
        logger.warning("评分解析失败: %s — 原始返回: %s", e, resp[:200] if 'resp' in dir() else "N/A")
        return 0, f"评分解析失败: {e}"


# ---------------------------------------------------------------------------
# 单条任务评测
# ---------------------------------------------------------------------------


def _run_single_task(
    agent: ReActAgent,
    llm: MimoLLM,
    eval_item: dict,
    category: str,
) -> TaskResult:
    """运行单条评测任务。"""
    task = eval_item["task"]
    expected_tools = eval_item.get("expected_tools", [])
    expected_steps = eval_item.get("expected_steps", 3)
    criteria = eval_item.get("evaluation_criteria", [])

    result = TaskResult(task=task, category=category)

    t0 = time.perf_counter()
    try:
        answer = agent.run(task)
        result.answer = answer
    except Exception as e:
        result.error = str(e)
        logger.error("任务执行失败: %s — %s", task[:40], e)
        return result
    t1 = time.perf_counter()
    result.elapsed_ms = (t1 - t0) * 1000

    # 提取步骤信息
    steps = agent.get_steps()
    result.steps_detail = steps
    result.step_count = len(steps)

    # 提取使用的工具
    for step in steps:
        action = step.get("action", "")
        if action:
            tool_name = action.split("(")[0].strip()
            if tool_name and tool_name not in result.tools_used:
                result.tools_used.append(tool_name)

    # LLM-as-Judge 评分
    score, reason = _judge_answer(
        llm, task, criteria, result.answer, result.tools_used, result.step_count
    )
    result.quality_score = score
    result.quality_reason = reason

    return result


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def _render_report(report: BenchmarkReport) -> str:
    """将评测结果渲染为 Markdown 报告。"""
    lines: list[str] = []
    lines.append("## Agent 评测报告")
    lines.append("")
    lines.append("> 由 `scripts/agent_benchmark.py` 自动生成。")
    lines.append("")

    # --- 总览 ---
    lines.append("### 总览")
    lines.append("")
    lines.append(f"- 评测任务总数：{report.total_tasks}")
    lines.append(f"- 成功执行：{report.success_count}")
    lines.append(f"- 执行失败：{report.error_count}")
    if report.results:
        valid_scores = [r.quality_score for r in report.results if r.quality_score > 0]
        if valid_scores:
            lines.append(f"- 平均质量评分：{statistics.mean(valid_scores):.2f} / 5")
            lines.append(f"- 评分 ≥ 4 的任务：{sum(1 for s in valid_scores if s >= 4)} / {len(valid_scores)}")
    lines.append("")

    # --- 按类别统计 ---
    lines.append("### 按类别统计")
    lines.append("")
    lines.append("| 类别 | 任务数 | 平均评分 | 平均步数 | 平均耗时(s) |")
    lines.append("|------|--------|---------|---------|------------|")

    categories = ["对比分析", "因果推理", "指标计算", "综合分析"]
    for cat in categories:
        cat_results = [r for r in report.results if r.category == cat]
        if not cat_results:
            continue
        valid = [r for r in cat_results if r.quality_score > 0]
        avg_score = statistics.mean([r.quality_score for r in valid]) if valid else 0
        avg_steps = statistics.mean([r.step_count for r in cat_results]) if cat_results else 0
        avg_time = statistics.mean([r.elapsed_ms / 1000 for r in cat_results]) if cat_results else 0
        lines.append(f"| {cat} | {len(cat_results)} | {avg_score:.2f} | {avg_steps:.1f} | {avg_time:.1f} |")

    lines.append("")

    # --- 工具使用分布 ---
    lines.append("### 工具使用分布")
    lines.append("")
    all_tools: list[str] = []
    for r in report.results:
        all_tools.extend(r.tools_used)
    tool_counter = Counter(all_tools)
    if tool_counter:
        lines.append("| 工具 | 使用次数 | 占比 |")
        lines.append("|------|---------|------|")
        total_uses = sum(tool_counter.values())
        for tool, count in tool_counter.most_common():
            pct = count / total_uses * 100
            lines.append(f"| {tool} | {count} | {pct:.1f}% |")
    else:
        lines.append("无工具使用记录。")
    lines.append("")

    # --- 步数统计 ---
    lines.append("### 步数统计")
    lines.append("")
    if report.results:
        step_counts = [r.step_count for r in report.results]
        over_step = sum(1 for r in report.results if r.step_count > r.steps_detail.__len__() > 0)
        lines.append(f"- 平均步数：{statistics.mean(step_counts):.2f}")
        lines.append(f"- 最大步数：{max(step_counts)}")
        lines.append(f"- 最小步数：{min(step_counts)}")
        # 超步数比例（步数 > expected_steps 的任务）
        over_expected = 0
        for r in report.results:
            # 从原始评测数据中获取 expected_steps（通过 task 匹配）
            over_expected += 1 if r.step_count > 6 else 0
        lines.append(f"- 超步数任务（>6步）：{over_expected} / {len(report.results)}")
    lines.append("")

    # --- Action 解析成功率 ---
    lines.append("### Action 解析成功率")
    lines.append("")
    total_actions = 0
    parse_failures = 0
    for r in report.results:
        for step in r.steps_detail:
            action = step.get("action", "")
            if action:
                total_actions += 1
            thought = step.get("thought", "")
            if "Action:" in thought and not action:
                parse_failures += 1
    if total_actions > 0:
        success_rate = (total_actions - parse_failures) / total_actions * 100
        lines.append(f"- 总 Action 次数：{total_actions}")
        lines.append(f"- 解析成功次数：{total_actions - parse_failures}")
        lines.append(f"- 解析成功率：{success_rate:.1f}%")
    else:
        lines.append("无 Action 记录（所有任务均直接回答）。")
    lines.append("")

    # --- 典型成功案例 ---
    lines.append("### 典型成功案例（评分 ≥ 4）")
    lines.append("")
    successes = sorted(
        [r for r in report.results if r.quality_score >= 4],
        key=lambda r: r.quality_score,
        reverse=True,
    )
    for r in successes[:5]:
        lines.append(f"**任务**：{r.task[:60]}")
        lines.append(f"- 评分：{r.quality_score}/5 — {r.quality_reason}")
        lines.append(f"- 工具：{', '.join(r.tools_used) or '无'} | 步数：{r.step_count} | 耗时：{r.elapsed_ms / 1000:.1f}s")
        lines.append(f"- 回答摘要：{r.answer[:150]}...")
        lines.append("")

    # --- 典型失败案例 ---
    lines.append("### 典型失败案例（评分 ≤ 2）")
    lines.append("")
    failures = sorted(
        [r for r in report.results if r.quality_score <= 2 and r.quality_score > 0],
        key=lambda r: r.quality_score,
    )
    for r in failures[:5]:
        lines.append(f"**任务**：{r.task[:60]}")
        lines.append(f"- 评分：{r.quality_score}/5 — {r.quality_reason}")
        lines.append(f"- 工具：{', '.join(r.tools_used) or '无'} | 步数：{r.step_count}")
        lines.append(f"- 回答摘要：{r.answer[:150]}...")
        lines.append("")

    # --- 错误任务 ---
    errors = [r for r in report.results if r.error]
    if errors:
        lines.append("### 执行错误")
        lines.append("")
        for r in errors:
            lines.append(f"- {r.task[:50]}... → {r.error}")
        lines.append("")

    # --- 详细结果表 ---
    lines.append("### 详细结果")
    lines.append("")
    lines.append("| # | 类别 | 任务摘要 | 评分 | 步数 | 工具 | 耗时(s) |")
    lines.append("|---|------|---------|------|------|------|--------|")
    for i, r in enumerate(report.results, 1):
        task_short = r.task[:30].replace("|", "\\|")
        tools_str = ",".join(r.tools_used) if r.tools_used else "-"
        score_str = str(r.quality_score) if r.quality_score > 0 else "ERR"
        time_str = f"{r.elapsed_ms / 1000:.1f}" if r.elapsed_ms > 0 else "-"
        lines.append(f"| {i} | {r.category} | {task_short} | {score_str} | {r.step_count} | {tools_str} | {time_str} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Benchmark")
    parser.add_argument("--api-key", type=str, default=None, help="API Key（也可通过环境变量配置）")
    parser.add_argument("--eval-path", type=str, default=None, help="评测数据集路径")
    parser.add_argument("--max-tasks", type=int, default=None, help="最多评测的任务数量")
    parser.add_argument("--max-steps", type=int, default=6, help="Agent 最大步数")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 文件路径")
    parser.add_argument("--model", type=str, default="mimo-v2-pro", help="LLM 模型名")
    args = parser.parse_args()

    # 1. 加载 API Key
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
    except ImportError:
        pass

    api_key = args.api_key or os.environ.get("MIMO_API_KEY", "") or os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        logger.error("未提供 API Key。请通过 --api-key 或 .env 配置 MIMO_API_KEY。")
        sys.exit(1)

    # 2. 加载评测数据集
    eval_path = Path(args.eval_path) if args.eval_path else EVAL_PATH
    eval_dataset = _load_eval_dataset(eval_path)
    if args.max_tasks:
        eval_dataset = eval_dataset[: args.max_tasks]
    logger.info("本次评测 %d 条任务", len(eval_dataset))

    # 3. 构建 LLM 和 Agent
    base_url = "https://token-plan-cn.xiaomimimo.com/v1"
    llm = MimoLLM(api_key=api_key, model=args.model, base_url=base_url)
    agent = _build_agent(llm, max_steps=args.max_steps)

    # 4. 分配类别
    categories = ["对比分析"] * 5 + ["因果推理"] * 5 + ["指标计算"] * 5 + ["综合分析"] * 5

    # 5. 逐条运行评测
    report = BenchmarkReport(total_tasks=len(eval_dataset))
    for i, item in enumerate(eval_dataset):
        cat = categories[i] if i < len(categories) else "综合分析"
        logger.info("[%d/%d] %s — %s", i + 1, len(eval_dataset), cat, item["task"][:40])
        task_result = _run_single_task(agent, llm, item, cat)
        report.results.append(task_result)
        if task_result.error:
            report.error_count += 1
        else:
            report.success_count += 1
        # 重置 agent 步骤记录
        agent._steps.clear()

    # 6. 生成报告
    md = _render_report(report)
    print("\n")
    print(md)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(md, encoding="utf-8")
        logger.info("报告已保存到: %s", output_path)

    # 7. 保存原始结果为 JSON
    json_path = eval_path.parent / "agent_eval_results.json"
    raw_results = []
    for r in report.results:
        raw_results.append({
            "task": r.task,
            "category": r.category,
            "answer": r.answer,
            "tools_used": r.tools_used,
            "step_count": r.step_count,
            "elapsed_ms": r.elapsed_ms,
            "quality_score": r.quality_score,
            "quality_reason": r.quality_reason,
            "error": r.error,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    logger.info("原始结果已保存到: %s", json_path)


if __name__ == "__main__":
    main()
