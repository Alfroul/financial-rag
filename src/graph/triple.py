from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    head: str       # 主体实体，如 "贵州茅台"
    relation: str   # 关系，如 "营收"
    tail: str       # 客体实体/值，如 "1680亿"
    source: str     # 来源文件路径

    def to_text(self) -> str:
        """返回 "head relation tail" 用于 embedding"""
        return f"{self.head} {self.relation} {self.tail}"
