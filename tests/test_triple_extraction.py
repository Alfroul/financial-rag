"""Triple extraction 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.fact_extractor.extractor import ALLOWED_RELATIONS, FactExtractor
from src.graph.triple import Triple


@pytest.fixture
def mock_llm():
    """创建 mock LLM。"""
    return MagicMock()


@pytest.fixture
def extractor(mock_llm):
    """创建使用 mock LLM 的 FactExtractor。"""
    return FactExtractor(llm=mock_llm)


class TestParseTriples:
    """_parse_triples 静态方法测试。"""

    def test_triple_creation_from_dict(self):
        """从 JSON dict 正确创建 Triple。"""
        data = [
            {
                "topic": "贵州茅台营收",
                "fact": "贵州茅台2024年营收1680亿元。",
                "category": ["个股分析"],
                "triples": [
                    {"head": "贵州茅台营收", "relation": "同比增长", "tail": "15.6%"},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 2
        assert triples[0] == Triple(head="贵州茅台营收", relation="同比增长", tail="15.6%", source="test.txt")
        assert triples[1] == Triple(head="贵州茅台", relation="属于", tail="白酒行业", source="test.txt")

    def test_triple_validation_empty_head(self):
        """空 head 被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": [
                    {"head": "", "relation": "属于", "tail": "某行业"},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 1
        assert triples[0].head == "贵州茅台"

    def test_triple_validation_empty_tail(self):
        """空 tail 被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": [
                    {"head": "贵州茅台", "relation": "属于", "tail": ""},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 1
        assert triples[0].tail == "白酒行业"

    def test_triple_validation_invalid_relation(self):
        """不在允许列表中的 relation 被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": [
                    {"head": "贵州茅台", "relation": "营收", "tail": "1680亿"},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 1
        assert triples[0].relation == "属于"

    def test_triple_validation_head_too_long(self):
        """head 长度超过 20 字被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": [
                    {"head": "这是一个超过二十个字的非常非常长的实体名称测试", "relation": "属于", "tail": "某行业"},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 1
        assert triples[0].head == "贵州茅台"

    def test_triple_invalid_dict_skipped(self):
        """非 dict 元素被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": [
                    "not a dict",
                    123,
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"},
                ],
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 1

    def test_triple_non_list_triples_field(self):
        """triples 字段非 list 时被跳过。"""
        data = [
            {
                "topic": "test",
                "fact": "test fact",
                "triples": "not a list",
            }
        ]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == 0

    def test_all_allowed_relations(self):
        """所有允许的关系类型都能正确创建 Triple。"""
        triples_raw = [
            {"head": "A", "relation": rel, "tail": "B"}
            for rel in ALLOWED_RELATIONS
        ]
        data = [{"topic": "t", "fact": "f", "triples": triples_raw}]
        triples = FactExtractor._parse_triples(data, source="test.txt")
        assert len(triples) == len(ALLOWED_RELATIONS)
        assert {t.relation for t in triples} == ALLOWED_RELATIONS


class TestExtractIntegration:
    """extract() 方法集成测试。"""

    def test_triple_extraction_from_llm_output(self, extractor, mock_llm):
        """从完整 LLM JSON 输出中提取 Fact + Triple。"""
        llm_output = """[
            {
                "topic": "贵州茅台营收",
                "fact": "贵州茅台2024年实现营收1680亿元，同比增长15.6%，继续保持白酒行业龙头地位。",
                "category": ["个股分析", "财务指标"],
                "triples": [
                    {"head": "贵州茅台营收", "relation": "同比增长", "tail": "15.6%"},
                    {"head": "贵州茅台", "relation": "属于", "tail": "白酒行业"}
                ]
            }
        ]"""
        mock_llm.chat.return_value = llm_output

        facts, triples = extractor.extract("贵州茅台2024年营收1680亿元", source="test.txt")

        assert len(facts) == 1
        assert facts[0].topic == "贵州茅台营收"
        assert facts[0].source == "test.txt"

        assert len(triples) == 2
        assert triples[0].head == "贵州茅台营收"
        assert triples[0].relation == "同比增长"
        assert triples[0].tail == "15.6%"
        assert triples[0].source == "test.txt"
        assert triples[1].head == "贵州茅台"
        assert triples[1].relation == "属于"

    def test_triple_extraction_robust_parsing(self, extractor, mock_llm):
        """markdown 代码块包裹的 JSON 也能解析。"""
        llm_output = """```json
[
    {
        "topic": "GDP增长",
        "fact": "2024年中国GDP同比增长5.0%。",
        "category": ["宏观经济"],
        "triples": [
            {"head": "中国GDP", "relation": "同比增长", "tail": "5.0%"}
        ]
    }
]
```"""
        mock_llm.chat.return_value = llm_output

        facts, triples = extractor.extract("2024年中国GDP数据", source="macro.txt")

        assert len(facts) == 1
        assert len(triples) == 1
        assert triples[0].head == "中国GDP"

    def test_triple_extraction_no_triples_field(self, extractor, mock_llm):
        """无 triples 字段时返回空 Triple 列表（向后兼容）。"""
        llm_output = """[
            {
                "topic": "测试主题",
                "fact": "这是一条没有三元组的测试事实。",
                "category": ["宏观经济"]
            }
        ]"""
        mock_llm.chat.return_value = llm_output

        facts, triples = extractor.extract("测试文本", source="test.txt")

        assert len(facts) == 1
        assert facts[0].topic == "测试主题"
        assert len(triples) == 0

    def test_fact_extraction_unchanged(self, extractor, mock_llm):
        """原有 Fact 提取逻辑不受 Triple 影响。"""
        llm_output = """[
            {
                "topic": "沪深300",
                "fact": "沪深300指数2024年涨幅14.68%，跑赢大部分主动管理基金。",
                "category": ["市场数据"]
            },
            {
                "topic": "央行降息",
                "fact": "央行2024年两次降息，LPR累计下调25个基点。",
                "category": ["政策法规", "宏观经济"]
            }
        ]"""
        mock_llm.chat.return_value = llm_output

        facts, triples = extractor.extract("市场综述", source="summary.txt")

        assert len(facts) == 2
        assert facts[0].topic == "沪深300"
        assert facts[0].category == ["市场数据"]
        assert facts[0].source == "summary.txt"
        assert facts[1].topic == "央行降息"
        assert facts[1].category == ["政策法规", "宏观经济"]
        assert len(triples) == 0

    def test_extract_empty_context(self, extractor):
        """空文本返回空列表。"""
        facts, triples = extractor.extract("", source="test.txt")
        assert facts == []
        assert triples == []

    def test_extract_returns_tuple(self, extractor, mock_llm):
        """extract() 返回 tuple[list[Fact], list[Triple]]。"""
        mock_llm.chat.return_value = "[]"
        result = extractor.extract("test", source="test.txt")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
