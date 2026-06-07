from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.fact_extractor.extractor import Fact, FactExtractor
from src.fact_extractor.prompt import FACT_EXTRACTION_PROMPT


def _make_extractor(llm_response: str) -> FactExtractor:
    """创建一个使用mock LLM的FactExtractor。"""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = llm_response
    return FactExtractor(llm=mock_llm)


SAMPLE_FACTS = [
    {
        "topic": "国内生产总值",
        "fact": (
            "GDP即国内生产总值，是衡量一个国家经济总量的核心指标。"
            "2024年中国GDP为134.9万亿元，同比增长5.0%，"
            "其中第三产业增加值占比最高。"
        ),
        "category": ["宏观经济"],
    },
    {
        "topic": "居民消费价格指数",
        "fact": (
            "CPI即居民消费价格指数，反映消费品和服务价格变动。"
            "2024年中国CPI同比上涨0.2%，处于较低水平，"
            "表明内需仍需进一步提振。"
        ),
        "category": ["宏观经济", "市场数据"],
    },
    {
        "topic": "社会消费品零售总额",
        "fact": "社会消费品零售总额衡量居民消费支出。2024年全年为48.8万亿元，同比增长3.5%，增速较上年有所放缓。",
        "category": ["宏观经济"],
    },
]


class TestExtractReturnsFacts:
    """test_extract_returns_facts — mock LLM返回标准JSON，验证Fact结构"""

    def test_extract_returns_facts(self):
        response = json.dumps(SAMPLE_FACTS, ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, triples = extractor.extract("GDP即国内生产总值...2024年中国GDP为134.9万亿元...", "qa.json")

        assert len(facts) == 3
        assert all(isinstance(f, Fact) for f in facts)
        assert facts[0].topic == "国内生产总值"
        assert "134.9万亿元" in facts[0].fact
        assert facts[0].category == ["宏观经济"]


class TestExtractHandlesMarkdownJson:
    """test_extract_handles_markdown_json — LLM返回```json包裹的JSON"""

    def test_extract_handles_markdown_json(self):
        response = f"```json\n{json.dumps(SAMPLE_FACTS, ensure_ascii=False)}\n```"
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("some context", "src.json")

        assert len(facts) == 3
        assert facts[1].topic == "居民消费价格指数"


class TestExtractHandlesInvalidJson:
    """test_extract_handles_invalid_json — LLM返回非法JSON，返回空列表"""

    def test_extract_handles_invalid_json(self):
        extractor = _make_extractor("这不是JSON，是LLM的废话回答")

        facts, _ = extractor.extract("some context", "src.json")

        assert facts == []

    def test_extract_handles_partial_garbage(self):
        extractor = _make_extractor("好的，以下是提取结果：\n[broken json")

        facts, _ = extractor.extract("some context", "src.json")

        assert facts == []


class TestExtractHandlesEmptyContext:
    """test_extract_handles_empty_context — 空context输入"""

    def test_extract_handles_empty_context(self):
        extractor = _make_extractor("[]")

        assert extractor.extract("", "src.json") == ([], [])
        assert extractor.extract("   ", "src.json") == ([], [])
        assert extractor.extract(None, "src.json") == ([], [])


class TestExtractPreservesSource:
    """test_extract_preserves_source — source字段正确传递到每个Fact"""

    def test_extract_preserves_source(self):
        response = json.dumps(SAMPLE_FACTS, ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("some context", "data/qa.json")

        assert all(f.source == "data/qa.json" for f in facts)

    def test_extract_different_source(self):
        response = json.dumps(SAMPLE_FACTS[:1], ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("some context", "年报/2024.pdf")

        assert facts[0].source == "年报/2024.pdf"


class TestExtractMultipleFacts:
    """test_extract_multiple_facts — 一段context提取出多个Fact"""

    def test_extract_multiple_facts(self):
        response = json.dumps(SAMPLE_FACTS, ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("macro context", "macro.json")

        assert len(facts) == 3
        topics = [f.topic for f in facts]
        assert "国内生产总值" in topics
        assert "居民消费价格指数" in topics
        assert "社会消费品零售总额" in topics

    def test_extract_single_fact_as_object(self):
        """LLM返回单个对象（非数组）也能处理"""
        response = json.dumps(SAMPLE_FACTS[0], ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("context", "src.json")

        assert len(facts) == 1
        assert facts[0].topic == "国内生产总值"


class TestFactPromptFormat:
    """test_fact_prompt_format — prompt模板正确替换{context}"""

    def test_prompt_contains_context(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "[]"
        extractor = FactExtractor(llm=mock_llm)

        test_context = "2024年中国GDP为134.9万亿元"
        extractor.extract(test_context, "src.json")

        call_args = mock_llm.chat.call_args
        user_msg = (
            call_args[1]["messages"][0]["content"]
            if "messages" in call_args[1]
            else call_args[0][1][0]["content"]
        )
        assert test_context in user_msg
        assert "{context}" not in user_msg

    def test_prompt_template_has_placeholder(self):
        assert "{context}" in FACT_EXTRACTION_PROMPT


class TestJsonParsing:
    """边界情况：额外文字前后缀、空数组等"""

    def test_json_with_prefix_text(self):
        response = "以下是提取的知识条目：\n" + json.dumps(SAMPLE_FACTS[:2], ensure_ascii=False)
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("ctx", "src.json")
        assert len(facts) == 2

    def test_empty_array(self):
        extractor = _make_extractor("[]")
        facts, _ = extractor.extract("some context", "src.json")
        assert facts == []

    def test_json_with_suffix_text(self):
        response = json.dumps(SAMPLE_FACTS[:1], ensure_ascii=False) + "\n以上是提取结果。"
        extractor = _make_extractor(response)

        facts, _ = extractor.extract("ctx", "src.json")
        assert len(facts) == 1

    def test_category_as_string(self):
        data = [{"topic": "CPI", "fact": "CPI是消费者物价指数", "category": "宏观经济"}]
        extractor = _make_extractor(json.dumps(data, ensure_ascii=False))

        facts, _ = extractor.extract("ctx", "src.json")
        assert facts[0].category == ["宏观经济"]
