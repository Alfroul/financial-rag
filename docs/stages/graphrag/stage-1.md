### 阶段 1：架构 Spike — 图谱骨架

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/graphrag/stage-1.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：项目结构、`CONTEXT.md`、`docs/stages/graphrag/stage-1.md`、测试命令。只需要读阶段 1 的内容。

**目标**：铺设图谱模块的最小骨架——Triple 数据模型、GraphStore 抽象接口、NetworkX 空实现、配置入口、hello world 验证。不写任何业务逻辑。

**任务清单**：

- [x] 1.1 创建 `src/graph/` 模块目录
  - 创建 `src/graph/__init__.py`（导出 Triple, GraphStore）
  - 创建 `src/graph/triple.py` — Triple 数据模型（frozen dataclass）
    ```python
    @dataclass(frozen=True)
    class Triple:
        head: str       # 主体实体，如 "贵州茅台"
        relation: str   # 关系，如 "营收"
        tail: str       # 客体实体/值，如 "1680亿"
        source: str     # 来源文件路径

        def to_text(self) -> str:
            """返回 "head relation tail" 用于 embedding"""
            return f"{self.head} {self.relation} {self.tail}"
    ```

- [x] 1.2 创建 `src/graph/graph_store.py` — GraphStore 抽象基类 + NetworkX 空实现
  - `GraphStore` ABC 接口签名（空实现，只 raise NotImplementedError）：
    ```python
    class GraphStore(ABC):
        @abstractmethod
        def add_triples(self, triples: list[Triple]) -> int: ...

        @abstractmethod
        def query_neighbors(self, entity: str, max_depth: int = 1) -> list[Triple]: ...

        @abstractmethod
        def query_path(self, entity_a: str, entity_b: str) -> list[list[Triple]]: ...

        @abstractmethod
        def get_entities(self) -> list[str]: ...

        @abstractmethod
        def delete_by_source(self, source: str) -> int: ...

        @abstractmethod
        def clear(self) -> None: ...

        @abstractmethod
        def stats(self) -> dict: ...

        @abstractmethod
        def save(self, path: str) -> None: ...

        @abstractmethod
        def load(self, path: str) -> None: ...
    ```
  - `NetworkxGraphStore(GraphStore)` 空壳实现——所有方法先 `raise NotImplementedError("阶段2实现")`
  - `__init__` 初始化空的有向图 `self._graph = nx.DiGraph()`

- [x] 1.3 更新 `src/config.py` — 新增 GraphConfig
  ```python
  @dataclass(frozen=True)
  class GraphConfig:
      enabled: bool = False
      persist_path: str = "data/graph_store.pkl"
      max_neighbors: int = 50
      max_depth: int = 2
  ```
  在 `load_config()` 中解析 config.yaml 的 `graph:` 块

- [x] 1.4 更新 `config.yaml` — 新增 graph 配置块
  ```yaml
  # 知识图谱配置（方案A）
  graph:
    enabled: false
    persist_path: "data/graph_store.pkl"
    max_neighbors: 50
    max_depth: 2
  ```

- [x] 1.5 更新 `CONTEXT.md` — 新增图谱相关术语（Triple, GraphStore, GraphRetriever, EntityMatcher, GraphRouter）

- [x] 1.6 创建 `docs/adr/0002-graph-store-networkx.md`
  - 决策：选择 NetworkX 而非 Neo4j
  - 原因：零外部依赖、数据量 < 10万节点足够、GraphStore 抽象层便于后续迁移
  - 迁移条件：节点数 > 10万 或需要 Cypher 复杂查询

- [x] 1.7 编写测试 `tests/test_graph_skeleton.py`
  - `test_triple_creation` — Triple dataclass 能正常创建
  - `test_triple_frozen` — frozen=True 不可修改
  - `test_triple_to_text` — to_text() 返回正确格式
  - `test_graph_store_interface` — GraphStore 所有方法签名存在
  - `test_graph_config` — GraphConfig 从 yaml 正确加载
  - `test_config_yaml_graph` — config.yaml 包含 graph 配置块

- [x] 1.8 验证
  - `ruff check src/graph/ src/config.py` 无错误
  - `pytest tests/test_graph_skeleton.py -v` 全部通过
  - `streamlit run app.py` 能正常启动（graph.enabled=false 不影响现有功能）

**验收标准**：
- `src/graph/` 模块存在，Triple 和 GraphStore 接口签名已定义
- config.yaml 包含 graph 配置块，config.py 能正确解析
- 所有骨架测试通过
- 现有功能不受影响（graph.enabled=false）

**完成确认**：

- [x] 阶段 1 全部任务完成，已通过验收标准
