### 阶段 3：功能 Slice — 图查询与路由集成

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/graphrag/stage-3.md，完成所有任务后确认完成`

**目标**：实现 GraphStore 的 NetworkX 完整实现、三路路由（FactCache / Graph / RAG）、Graph prompt 组装，让图谱查询端到端可用。

**前置依赖**：阶段 1（骨架）+ 阶段 2（Triple 抽取）已完成。

**任务清单**：

- [x] 3.1 实现 `src/graph/graph_store.py` — NetworkxGraphStore 完整实现
  - `add_triples(triples)` — 添加三元组到有向图
    - 节点：head 和 tail 作为节点，附带属性 `{type: "entity"}`
    - 边：head→tail，附带属性 `{relation: relation, source: source}`
    - 去重：同一 (head, relation, tail) 不重复添加
  - `query_neighbors(entity, max_depth=1)` — BFS 遍历邻居
    - 从 entity 节点出发，按 max_depth 做 BFS
    - 返回所有可达的 Triple 列表
    - max_depth=1 时只返回直接邻居
  - `query_path(entity_a, entity_b)` — 最短路径查询
    - 使用 `nx.shortest_path()` 查找两条路径
    - 返回路径上的所有 Triple
    - 无路径时返回空列表
  - `get_entities()` — 返回所有节点名称列表
  - `delete_by_source(source)` — 删除指定来源的所有 Triple
  - `clear()` — 清空图
  - `stats()` — 返回 `{nodes: int, edges: int, sources: int}`
  - `save(path)` / `load(path)` — pickle 序列化/反序列化

- [x] 3.2 创建 `src/graph/entity_matcher.py` — 实体模糊匹配
  - 目的：用户查询中的实体名可能与图中的节点名不完全一致
  - 实现：
    - `build_index(entities: list[str], embedder)` — 为所有实体名生成 embedding
    - `match(query_entity: str, threshold: float = 0.85) -> str | None` — 找最相似的图节点
  - 使用现有 SiliconFlowEmbedder 的 `embed_query()` 方法
  - embedding 缓存：匹配时先计算 query_entity 的 embedding，与预存的实体 embedding 做余弦相似度

- [x] 3.3 创建 `src/graph/graph_retriever.py` — 图检索器
  - 包装 GraphStore + EntityMatcher，提供高层检索接口
  - `retrieve(query: str, mode: str = "neighbors") -> list[Triple]`
    - 从 query 中提取候选实体（使用 jieba 分词 + 已知实体列表交叉）
    - 通过 EntityMatcher 映射到图节点
    - 根据 mode 调用 GraphStore 的不同查询方法
  - 模式：
    - `"neighbors"` — 查询实体邻居（默认）
    - `"comparison"` — 两个实体的邻居合并
    - `"path"` — 两实体间路径

- [x] 3.4 实现三路路由 — 改造 `src/rag_pipeline.py`
  - 新增路由判断方法 `_route_query(query: str) -> str`
    - 规则 1：查询中包含两个已知实体名 + 对比词（"对比"/"vs"/"和"/"与"）→ `"graph_comparison"`
    - 规则 2：查询中包含因果词（"为什么"/"原因"/"影响"/"导致"）→ `"graph_causal"`
    - 规则 3：其他 → `"rag"`（现有逻辑）
  - 在 `query()` 和 `aquery()` 中，在 FactCache 检查之后、RAG 检索之前，插入图路由：
    ```
    graph_triples = self._try_graph_query(query)  # 新增
    if graph_triples:
        # 图 + RAG 混合模式
        context = self._build_graph_prompt(graph_triples, rag_results)
    else:
        # 纯 RAG 模式（现有逻辑不变）
    ```
  - 新增 `_build_graph_prompt(triples, chunks)` — 将 Triple 和 RAG chunk 合并组装 prompt
    - Triple 格式化为 `知识点: {head} {relation} {tail}` 列表
    - 追加在 RAG context 之前，标注 "[图谱知识]" 和 "[文档来源]"

- [x] 3.5 依赖注入 — 更新 `src/ui/services.py` 和 `src/api/deps.py`
  - graph.enabled=true 时：
    - 创建 NetworkxGraphStore 实例
    - 调用 `load()` 加载持久化数据（如文件存在）
    - 创建 EntityMatcher（传入 embedder + graph.get_entities()）
    - 创建 GraphRetriever（传入 graph_store + entity_matcher）
    - 传入 RAGPipeline 构造函数

- [x] 3.6 编写测试
  - `tests/test_graph_store.py`
    - `test_add_triples` — 添加 Triple 后图有正确的节点和边
    - `test_add_triples_dedup` — 重复 Triple 不重复添加
    - `test_query_neighbors` — 查询邻居返回正确 Triple
    - `test_query_neighbors_depth` — max_depth 控制遍历深度
    - `test_query_path` — 两实体间路径查询
    - `test_query_path_no_connection` — 无连接返回空
    - `test_delete_by_source` — 按来源删除
    - `test_save_load` — pickle 持久化和加载
    - `test_stats` — 统计信息正确
  - `tests/test_graph_routing.py`
    - `test_route_comparison` — "茅台和五粮液对比" 路由到 graph_comparison
    - `test_route_causal` — "为什么茅台涨了" 路由到 graph_causal
    - `test_route_rag` — "GDP是什么" 路由到 rag
    - `test_graph_prompt_format` — Graph prompt 格式正确
    - `test_graph_disabled` — graph.enabled=false 时完全跳过图路由

- [x] 3.7 验证
  - `ruff check src/graph/ src/rag_pipeline.py` 无错误
  - `pytest tests/test_graph*.py -v` 全部通过
  - 手动验证：启动 app.py，输入对比类查询，观察是否有图谱知识出现在回答中

**验收标准**：
- GraphStore NetworkX 实现完整，所有接口可用
- 三路路由正确判断查询类型
- 图 + RAG 混合 prompt 正确组装
- graph.enabled=false 时现有功能完全不受影响
- 所有测试通过

**完成确认**：

- [x] 阶段 3 全部任务完成，已通过验收标准
