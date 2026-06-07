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
