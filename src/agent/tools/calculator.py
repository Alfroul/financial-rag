from __future__ import annotations

from typing import Any

from src.agent.base_tool import BaseTool, ToolResult

_MAX_EXPRESSION_LENGTH = 500


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "执行金融指标计算。当你需要计算比率、增长率、对比数值时使用。"
        "输入：expression（Python 数学表达式）。"
    )

    ALLOWED_NAMES: dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sorted": sorted,
        "float": float,
        "int": int,
    }

    def run(self, expression: str = "", **kwargs: Any) -> ToolResult:
        if not expression or not expression.strip():
            return ToolResult(success=False, output="表达式不能为空")
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            return ToolResult(success=False, output="表达式过长，已拒绝执行")
        try:
            result = eval(expression, {"__builtins__": {}}, self.ALLOWED_NAMES)  # noqa: S307
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output=f"计算错误: {e}")
