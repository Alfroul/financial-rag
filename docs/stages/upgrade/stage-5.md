### 阶段 5：Neo4j 知识图谱 — 替换 NetworkX

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-5.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-5.md`、`src/graph/`（现有图谱模块）

**目标**：用 Neo4j 社区版替换 NetworkX + pickle 的图谱存储方案，支持 Cypher 查询，提升图谱查询能力和生产可信度。

**前置依赖**：阶段 1 已完成（LLM 已切换到 MiMo）。

**任务清单**：

1. 依赖与基础设施
   - `requirements.txt` 新增 `neo4j>=5.0.0`
   - `docker-compose.yml` 新增 Neo4j 服务：
     ```yaml
     neo4j:
       image: neo4j:5-community
       ports:
         - "7474:7474"   # Browser UI
         - "7687:7687"   # Bolt protocol
       environment:
         NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
       volumes:
         - neo4j_data:/data
     ```
   - `config.yaml` 新增/修改 `graph` 配置段：
     ```yaml
     graph:
       enabled: false
       backend: "neo4j"          # "networkx" 或 "neo4j"
       neo4j_uri: "bolt://localhost:7687"
       neo4j_user: "neo4j"
       neo4j_password_env: "NEO4J_PASSWORD"
     ```
   - `src/config.py`：`GraphConfig` 新增 `backend`、`neo4j_uri`、`neo4j_user`、`neo4j_password_env` 字段

2. Neo4jGraphStore 实现
   - 修改 `src/graph/graph_store.py`：
     - 新增 `Neo4jGraphStore` 类，实现 `GraphStore` 接口
     - `add_triples(triples)`：批量插入三元组，使用 UNWIND + MERGE 保证幂等
     - `query_neighbors(entity, max_depth)`：Cypher 多跳邻居查询
     - `query_path(entity1, entity2)`：Cypher 最短路径查询
     - `query_relations(entity, relation)`：按关系类型过滤查询
     - `delete_by_source(source)`：按来源删除
     - `get_stats()`：节点数、关系数统计
   - 节点标签：`Entity`，属性：`name`、`type`
   - 关系类型：使用 `Triple.relation` 作为关系类型（如 `同比增长`、`对比`）
   - 创建索引：`CREATE INDEX FOR (e:Entity) ON (e.name)`

3. GraphStore 工厂模式
   - `src/graph/graph_store.py` 新增 `create_graph_store(config: GraphConfig) -> GraphStore`：
     - `backend == "networkx"` → 返回 `NetworkxGraphStore`（现有实现）
     - `backend == "neo4j"` → 返回 `Neo4jGraphStore`
   - 下游代码（`GraphRetriever`、`RAGPipeline`）通过工厂创建，无需关心具体实现

4. 数据迁移
   - 创建 `scripts/migrate_graph_to_neo4j.py`：
     - 读取现有 `data/graph_store.pkl`
     - 解析 NetworkX 图为 Triple 列表
     - 批量插入到 Neo4j
     - 验证迁移完整性（节点数、关系数一致）

5. 更新下游调用
   - `src/rag_pipeline.py`：图存储创建改用工厂方法
   - `src/graph/graph_retriever.py`：确认接口兼容
   - `src/agent/tools/knowledge_graph.py`：确认接口兼容

6. 测试
   - `tests/test_neo4j_graph_store.py`：
     - 使用 Neo4j test container 或 mock 测试
     - 测试 add_triples、query_neighbors、query_path
     - 测试幂等性（重复插入不报错）
     - 测试空图查询返回空结果
   - 运行 `pytest tests/ -x -q` 确认全量通过
   - 确认 NetworkX 后端的测试仍然通过（向后兼容）

**验收标准**：
- `Neo4jGraphStore` 实现完整，所有 `GraphStore` 接口方法正常
- `docker compose up neo4j` 启动后，图谱数据可正常读写
- 迁移脚本成功将现有 NetworkX 数据导入 Neo4j
- 工厂模式支持在 `networkx` 和 `neo4j` 之间切换
- 全量测试通过

**技术备注**：
- Neo4j 社区版免费，单机足够本项目使用
- `MERGE` 语句保证幂等插入，避免重复数据
- 连接池管理：`neo4j.GraphDatabase.driver` 是线程安全的，全局持有一个实例即可
- 如果 Neo4j 不可用，应 fallback 到 NetworkX（日志警告）
