from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from src.fact_extractor.prompt import FACT_EXTRACTION_PROMPT
from src.graph.triple import Triple

logger = logging.getLogger(__name__)

ALLOWED_RELATIONS = frozenset({
    "属于",
    "包含",
    "同比增长",
    "环比增长",
    "发布",
    "对比",
    "影响",
    "高于",
    "低于",
    "等于",
})


@dataclass
class Fact:
    topic: str
    fact: str
    category: list[str] = field(default_factory=list)
    source: str = ""


class FactExtractor:
    """从RAG检索结果中提取结构化Fact条目。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mimo-v2-pro",
        llm=None,
    ) -> None:
        if llm is not None:
            self._llm = llm
        else:
            if not api_key:
                raise ValueError("必须提供 api_key 或 llm 实例")
            from src.generator.mimo_llm import MimoLLM

            self._llm = MimoLLM(
                api_key=api_key,
                model=model,
                temperature=0.1,
            )

    def extract(self, context: str, source: str) -> tuple[list[Fact], list[Triple]]:
        if not context or not context.strip():
            return [], []

        prompt = FACT_EXTRACTION_PROMPT.replace("{context}", context)

        try:
            raw = self._llm.chat(
                system_prompt="你是一个金融知识提取专家。请严格按照要求输出JSON数组。",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("LLM调用失败: %s", e)
            return [], []

        data = self._extract_json(raw)
        if data is None:
            return [], []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return [], []

        facts = self._parse_facts(data)
        for f in facts:
            f.source = source

        triples = self._parse_triples(data, source)

        return facts, triples

    @staticmethod
    def _parse_facts(data: list) -> list[Fact]:
        """从解析后的JSON数组中提取Fact列表（原有逻辑不变）。"""
        facts: list[Fact] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic", "")
            fact_text = item.get("fact", "")
            if not topic or not fact_text:
                continue
            category = item.get("category", [])
            if isinstance(category, str):
                category = [category]
            facts.append(Fact(topic=topic, fact=fact_text, category=category))
        return facts

    @staticmethod
    def _parse_triples(data: list, source: str) -> list[Triple]:
        """从解析后的JSON数组中提取Triple列表。

        遍历每个fact条目的triples字段，验证后收集。
        无效Triple（空head/tail、head超长、relation不在允许列表）静默跳过。
        """
        triples: list[Triple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            raw_triples = item.get("triples", [])
            if not isinstance(raw_triples, list):
                continue
            for raw in raw_triples:
                if not isinstance(raw, dict):
                    continue
                head = str(raw.get("head", "")).strip()
                relation = str(raw.get("relation", "")).strip()
                tail = str(raw.get("tail", "")).strip()
                if not head or not tail:
                    continue
                if len(head) > 20:
                    continue
                if relation not in ALLOWED_RELATIONS:
                    continue
                triples.append(Triple(head=head, relation=relation, tail=tail, source=source))
        return triples

    @staticmethod
    def _extract_json(raw: str) -> list | dict | None:
        # 直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 提取 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 提取第一个 [ ... ] 或 { ... }
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        return None
