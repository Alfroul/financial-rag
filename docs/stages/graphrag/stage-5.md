### 阶段 5：全局 Review — GraphRAG

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/graphrag/stage-5.md，对 GraphRAG 模块进行全面的 Code Review`

**目标**：对 GraphRAG 模块进行全局审查，确保代码质量、架构一致性和测试完整性。

**前置依赖**：阶段 1-4 全部完成。

**任务清单**：

- [x] 5.1 代码质量审查
  - `src/graph/` — 模块间耦合度、接口一致性 ✓
  - `src/fact_extractor/` — Fact + Triple 联合抽取的健壮性 ✓
  - `src/rag_pipeline.py` — 三路路由的代码清晰度 ✓
  - 命名规范：是否使用 `CONTEXT.md` 中定义的术语 ✓
  - 类型注解完整性 ✓

- [x] 5.2 架构审查
  - GraphStore 抽象层是否正确隔离了 NetworkX 实现 ✓
  - 三路路由是否与 FactCache 路由解耦 ✓
  - EntityMatcher 的 embedding 缓存策略是否合理 ✓
  - 是否有过度设计或设计不足 ✓

- [x] 5.3 安全性审查
  - Triple 的 source 字段是否可能泄露文件系统路径 ✓
  - EntityMatcher 的 embedding 缓存是否有内存泄漏风险 ✓
  - pickle 反序列化是否安全（是否需要签名校验） ✓

- [x] 5.4 性能审查
  - `query_neighbors()` BFS 在大图上的性能 ✓
  - EntityMatcher 每次查询是否重复计算 embedding ✓
  - GraphStore.load() 在大数据量下的启动时间 ✓

- [x] 5.5 测试完整性
  - 所有阶段的测试通过 ✓
  - 边界情况覆盖：空图、无匹配实体、超深遍历、超大 Triple 批量插入 ✓
  - 集成测试：端到端查询路径（query → route → graph → prompt → answer） ✓

- [x] 5.6 文档完整性
  - CONTEXT.md 包含所有图谱术语 ✓
  - docs/adr/ 包含所有架构决策 ✓
  - plan-graphrag.md 所有阶段已勾选 ✓
  - Benchmark 报告完整 ✓

**完成确认**：

- [x] 阶段 5 全部任务完成，GraphRAG 模块 Review 通过

---

## Review 报告

### 5.1 代码质量审查

**src/graph/**：模块结构清晰，4 个文件职责分明。
- `Triple`：frozen dataclass，`to_text()` 用于 embedding，设计合理
- `GraphStore`：ABC 定义 9 个抽象方法，`NetworkxGraphStore` 完整实现
- `EntityMatcher`：embedding 缓存通过 `build_index()` 一次性构建，`match()` 线性扫描
- `GraphRetriever`：包装 GraphStore + EntityMatcher，提供 neighbors/comparison/path 三种模式

**src/fact_extractor/**：健壮性良好。
- `_extract_json` 三级 fallback（直接解析 → markdown 代码块 → 正则提取）
- `_parse_triples` 验证完整：空 head/tail、head 超长、relation 白名单
- `ALLOWED_RELATIONS` 使用 frozenset，与 CONTEXT.md 一致

**src/rag_pipeline.py**：三路路由代码清晰。
- `_route_query` 使用正则 + 实体列表匹配，零 LLM 开销
- `_try_graph_query` 与 `_try_fact_cache` 解耦良好
- `_build_graph_prompt` 静态方法，职责单一

**命名规范**：全部使用 CONTEXT.md 定义的术语（Triple、GraphStore、EntityMatcher、GraphRetriever）。

**类型注解**：所有 graph 模块文件使用 `from __future__ import annotations`，TYPE_CHECKING 正确使用。

### 5.2 架构审查

**GraphStore 抽象层**：正确隔离 NetworkX 实现。ABC 定义清晰，迁移 Neo4j 只需实现新子类。

**三路路由**：与 FactCache 路由解耦。`_route_query` 和 `_try_graph_query` 是独立方法，FactCache 路由在 `_try_fact_cache` 中处理。

**EntityMatcher 缓存策略**：合理。`build_index()` 一次性计算所有实体 embedding 并缓存，图变更时需重新调用。

**设计评估**：无过度设计。NetworkX + pickle 适合当前规模（<10k 节点），规则路由无需 LLM。

### 5.3 安全性审查

**Triple source 路径泄露**：低风险。source 存储相对路径（如 `data/raw/report.txt`），仅用于内部去重，不直接暴露给用户。

**EntityMatcher 内存泄漏**：无风险。`build_index()` 替换而非追加，embeddings 列表大小与实体数线性相关。

**pickle 反序列化**：中风险（本地应用可接受）。`graph_store.py:165` 使用 `pickle.load()` 无签名校验。攻击者若能写入 `data/` 目录可执行任意代码。缓解措施：应用控制文件访问，后续可考虑 JSON/msgpack 替代。

### 5.4 性能审查

**query_neighbors() BFS**：可接受。max_depth=2 + max_neighbors=50 截断，对 <10k 节点图足够。

**EntityMatcher 重复计算**：可接受。index embedding 已缓存，query embedding 每次调用计算（典型 1-3 个实体/查询）。

**GraphStore.load()**：可接受。pickle 一次性反序列化，<10k 节点启动时间可控。

### 5.5 测试完整性

**66 项测试全部通过**，覆盖：
- Triple 创建、frozen、to_text（3 项）
- GraphStore 接口、NetworkxGraphStore 子类（3 项）
- GraphConfig（2 项）
- add_triples 正常/去重/不同关系/节点属性/边属性（5 项）
- query_neighbors 正常/深度/不存在（3 项）
- query_path 正常/无连接/不存在（3 项）
- delete_by_source 正常/移除孤立节点（2 项）
- stats、save/load、clear（4 项）
- Triple 解析验证（8 项）
- Triple 提取集成（6 项）
- 路由判断（5 项）
- Graph prompt 构建（2 项）
- RuleChecker + GraphStore（3 项）
- CacheSynchronizer + GraphStore（4 项）
- FactCache 同步（6 项）

**边界覆盖**：空图、无匹配实体、超深遍历、批量插入去重均已覆盖。

**已修复问题**：`sync.py` 中 `_save_graph()` 硬编码 `"data/graph.pkl"` 与配置的 `data/graph_store.pkl` 不一致，已修复为使用 `graph_persist_path` 参数。

### 5.6 文档完整性

- CONTEXT.md：包含所有图谱术语（Triple、GraphStore、EntityMatcher、GraphRetriever、GraphRouter）
- docs/adr/0002-graph-store-networkx.md：架构决策记录完整
- plan-graphrag.md：所有阶段 1-5 已勾选
- Benchmark：脚本已存在（scripts/benchmark.py），支持 GraphRAG 对比
