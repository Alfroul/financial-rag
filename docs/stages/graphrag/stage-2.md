### 阶段 2：功能 Slice — 实体关系抽取

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/graphrag/stage-2.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/graphrag/stage-2.md`、测试命令。

**目标**：改造 FactExtractor，使其在同一次 LLM 调用中同时输出 Fact + Triple。实现 Triple 的 JSON 解析和验证。

**前置依赖**：阶段 1 已完成（Triple 数据模型已定义）。

**任务清单**：

- [x] 2.1 改造 `src/fact_extractor/prompt.py`
  - 修改 Fact 提取 prompt，在输出 schema 中增加 Triple 段：
    ```
    输出格式（JSON数组）：
    [
      {
        "topic": "主题词",
        "fact": "事实描述",
        "category": ["分类"],
        "triples": [
          {"head": "主体实体", "relation": "关系", "tail": "客体实体/值"}
        ]
      }
    ]
    ```
  - 在 prompt 中增加关系类型限制：
    ```
    关系类型限于：属于、包含、同比增长、环比增长、发布、对比、影响、高于、低于、等于
    ```
  - 增加一条 few-shot 示例，展示带 Triple 的输出

- [x] 2.2 改造 `src/fact_extractor/extractor.py`
  - 修改 `extract()` 返回类型为 `tuple[list[Fact], list[Triple]]`
  - 在 JSON 解析后，从每个 fact 条目中提取 `triples` 字段
  - Triple 验证逻辑：
    - head/relation/tail 都非空
    - head 长度 ≤ 20 字
    - relation 必须在允许的关系类型列表中
    - 无效 Triple 静默跳过（不影响 Fact 提取）
  - 保持现有 Fact 提取逻辑完全不变——Triple 是附加输出，Fact 流程零改动
  - 新增方法 `_parse_triples(raw_triples: list[dict], source: str) -> list[Triple]`

- [x] 2.3 更新 `src/rag_pipeline.py` 中的 `_extract_and_cache_facts()`
  - 接收 extractor 返回的 `tuple[list[Fact], list[Triple]]`
  - Fact 存入 FactCache（现有逻辑不变）
  - Triple 存入 GraphStore（新增，graph.enabled 时才调用）
  - Triple 存储失败不影响主流程（try/except 静默处理）

- [x] 2.4 编写测试 `tests/test_triple_extraction.py`
  - `test_triple_creation_from_dict` — 从 JSON dict 正确创建 Triple
  - `test_triple_validation_empty_head` — 空 head 被跳过
  - `test_triple_validation_invalid_relation` — 不在允许列表中的 relation 被跳过
  - `test_triple_extraction_from_llm_output` — 从完整 LLM JSON 输出中提取 Fact + Triple
  - `test_triple_extraction_robust_parsing` — markdown 代码块包裹的 JSON 也能解析
  - `test_triple_extraction_no_triples_field` — 无 triples 字段时返回空列表（向后兼容）
  - `test_fact_extraction_unchanged` — 原有 Fact 提取逻辑不受影响

- [x] 2.5 验证
  - `ruff check src/fact_extractor/` 无错误
  - `pytest tests/test_triple_extraction.py -v` 全部通过
  - `pytest tests/test_fact_cache*.py -v` 原有测试仍通过

**验收标准**：
- FactExtractor 同时输出 Fact + Triple
- 原有 Fact 提取流程零改动
- Triple JSON 解析有兜底（无 triples 字段、字段缺失、无效值均静默跳过）
- 所有测试通过

**完成确认**：

- [x] 阶段 2 全部任务完成，已通过验收标准
