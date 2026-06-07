# ADR 0002: 选择 NetworkX 作为图存储引擎

## 状态

已接受

## 上下体

GraphRAG 方案需要一个图存储引擎来存储金融实体关系三元组（Triple）。候选方案：

1. **NetworkX** — Python 原生图库，pickle 持久化
2. **Neo4j** — 专业图数据库，Cypher 查询语言
3. **iGraph** — 高性能图库，C 核心 Python 绑定

## 决策

选择 **NetworkX**，通过 GraphStore 抽象层封装实现细节。

## 原因

| 维度 | NetworkX | Neo4j | iGraph |
|------|----------|-------|--------|
| 外部依赖 | 零（pip install 即可） | 需要 JVM + Docker | C 编译依赖 |
| 数据量适用范围 | < 10万节点 | 百万+ | 百万+ |
| 部署复杂度 | 无 | 中等 | 低 |
| 查询能力 | BFS/最短路径/子图 | Cypher 全功能 | BFS/最短路径 |
| 学习曲线 | 极低 | 中等（Cypher） | 低 |
| 项目数据量预估 | < 1万节点（金融文档有限） | — | — |

金融文档数据量有限（数百份报告、数千个实体），NetworkX 完全够用。零外部依赖意味着不需要 Docker、不需要 JVM、不需要额外配置。

## 迁移条件

当以下任一条件满足时，考虑迁移到 Neo4j：

- 节点数超过 10 万
- 需要 Cypher 级别的复杂图查询（多跳聚合、图算法）
- 需要多进程并发读写图数据

迁移成本可控：GraphStore 抽象层已隔离实现细节，只需写一个 `Neo4jGraphStore(GraphStore)` 实现类。

## 后果

- 正面：零部署成本，快速集成，Python 原生调试方便
- 负面：大数据量下性能可能不足，pickle 文件不是事务安全的
- 缓解：GraphStore 抽象层 + 持久化前备份
