### 阶段 4：功能 Slice — 图增强自纠 + 增量同步

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/graphrag/stage-4.md，完成所有任务后确认完成`

**目标**：让 Self-Correction 的 RuleChecker 能用图做实体验证，让数据源变更时 Triple 能增量重建。

**前置依赖**：阶段 3 已完成（GraphStore 和路由已集成）。

**任务清单**：

- [x] 4.1 改造 `src/correction/rule_checker.py` — 图增强实体验证
  - 在 `__init__` 中新增可选参数 `graph_store: GraphStore | None = None`
  - 在实体检查环节（现有 `_check_entities()` 方法）增加图验证：
    ```
    现有逻辑：检查回答中的金融术语是否出现在 source 文档中
    新增逻辑：如果 graph_store 存在，额外检查实体是否在图中有对应节点
    ```
  - 图验证结果作为补充信号，不覆盖原有 source 验证结果
  - graph_store=None 时完全跳过，零影响

- [x] 4.2 改造 `src/fact_cache/sync.py` — Triple 增量同步
  - 在 `CacheSynchronizer` 中新增 `graph_store: GraphStore | None` 参数
  - 文件变更时同步重建 Triple：
    - `added` 文件 → 提取 Triple → `graph_store.add_triples()`
    - `modified` 文件 → 删除旧 Triple（`delete_by_source`）→ 重新提取
    - `deleted` 文件 → `delete_by_source()`
  - Triple 重建失败不影响 FactCache 同步（独立 try/except）
  - 同步完成后 `graph_store.save()` 持久化

- [x] 4.3 编写测试 `tests/test_graph_correction.py`
  - `test_entity_check_with_graph` — 图中有实体时验证通过
  - `test_entity_check_without_graph` — graph_store=None 时跳过图验证
  - `test_graph_sync_added_file` — 新文件触发 Triple 提取
  - `test_graph_sync_modified_file` — 修改文件触发删除+重建
  - `test_graph_sync_deleted_file` — 删除文件触发 delete_by_source
  - `test_graph_sync_failure_isolated` — Triple 同步失败不影响 FactCache

- [x] 4.4 Benchmark — 新增一轮 RAGAS 对比
  - 更新 `scripts/benchmark.py`，增加配置 `hybrid + graph`
  - 对比维度：
    - `hybrid`（现有最优）
    - `hybrid + graph`（新增图路由）
  - 关注指标变化：对比类/因果类问题的 Faithfulness 和 Context Recall
  - 生成对比报告 `docs/benchmark_results_graphrag.md`

- [x] 4.5 验证
  - `pytest tests/test_graph*.py -v` 全部通过
  - `pytest tests/test_self_correction.py -v` 原有自纠测试仍通过
  - Benchmark 对比报告已生成

**验收标准**：
- RuleChecker 可选接入 GraphStore，无 Graph 时行为不变
- CacheSynchronizer 可选同步 Triple，失败不影响 FactCache
- Benchmark 报告显示图增强对特定查询类型的改进
- 所有测试通过

**完成确认**：

- [x] 阶段 4 全部任务完成，已通过验收标准
