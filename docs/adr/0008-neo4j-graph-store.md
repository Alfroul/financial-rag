# ADR 0008: 用 Neo4j 替换 NetworkX 作为知识图谱存储

## 状态

已批准

## 上下文

现有 `NetworkxGraphStore` 使用 NetworkX + pickle 存储：
- pickle 不是可靠的持久化方案（版本兼容性问题）
- NetworkX 是纯内存图库，无法处理大规模图谱
- 面试官一看就知道不是生产方案

## 决策

使用 Neo4j 社区版替换 NetworkX，理由：
1. 业界主流图数据库，简历辨识度高
2. Cypher 查询语言表达力强，支持复杂路径查询
3. 社区版免费，Docker 一键启动
4. 支持 ACID 事务，数据一致性有保障

## 后果

- 正面：生产级图谱存储、Cypher 查询能力、面试加分
- 负面：新增 Neo4j 依赖（Docker），部署复杂度略增
- 缓解：通过工厂模式支持 `networkx` 和 `neo4j` 两种后端，开发阶段可免 Docker
