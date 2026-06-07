### 阶段 4：MCP Server — 包装 RAG 为 MCP 工具服务

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-4.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-4.md`、`src/agent/base_tool.py`（工具接口参考）

**目标**：将 RAG 系统的核心能力（检索、图谱查询、Agent 分析）包装为 MCP Server，让 Claude Desktop / Cursor / 其他 MCP Client 可以直接调用。

**前置依赖**：阶段 1 已完成（LLM 已切换到 MiMo）。

**任务清单**：

1. 依赖安装
   - `requirements.txt` 新增 `mcp>=1.0.0`（Anthropic 官方 Python MCP SDK）
   - `config.yaml` 新增 `mcp_server` 配置段：
     ```yaml
     mcp_server:
       host: "localhost"
       port: 8080
       transport: "stdio"   # stdio 或 sse
     ```

2. MCP Server 骨架
   - 创建 `src/mcp_server/__init__.py`
   - 创建 `src/mcp_server/server.py`：
     - 使用 `mcp` Python SDK 创建 `Server` 实例
     - 注册 3 个 MCP Tools（见下方）
     - 支持 `stdio` transport（Claude Desktop 调用）和 `sse` transport（远程调用）
     - 入口：`python -m src.mcp_server.server`

3. MCP Tool 定义
   - 创建 `src/mcp_server/tools.py`
   - Tool 1：`financial_search`
     - 入参：`query: str`（查询文本）、`top_k: int`（返回数量，默认 5）
     - 返回：检索到的文档片段 + 来源信息
     - 内部调用 `RAGPipeline` 的检索链路（不含 LLM 生成）
   - Tool 2：`knowledge_graph_query`
     - 入参：`entity: str`（实体名）、`relation: str`（关系类型，可选）
     - 返回：匹配的三元组列表 + 邻居实体
     - 内部调用 `GraphRetriever`
   - Tool 3：`financial_analysis`
     - 入参：`question: str`（分析问题）
     - 返回：完整的 RAG 分析结果（检索 + LLM 生成）
     - 内部调用 `RAGPipeline.query()`

4. 与 Claude Desktop 集成配置
   - 创建 `docs/mcp_integration_guide.md`：
     - Claude Desktop 配置示例（`claude_desktop_config.json`）
     - Cursor 配置示例
     - 使用示例（"帮我查一下茅台的ROE" → 调用 financial_search）
   - 配置文件中 Server 路径指向 `python -m src.mcp_server.server`

5. 测试
   - `tests/test_mcp_server.py`：
     - 测试 MCP Server 启动不报错
     - 测试 3 个 Tool 的 schema 定义正确（name、description、inputSchema）
     - Mock RAG Pipeline，测试 Tool 调用返回预期格式
   - 手动测试（可选）：在 Claude Desktop 中配置 Server，发送测试 query

**验收标准**：
- `python -m src.mcp_server.server` 启动成功，MCP 协议握手正常
- 3 个 MCP Tool 定义完整，schema 符合 MCP 规范
- Claude Desktop 配置后能识别并调用 Tools
- Mock 测试通过

**技术备注**：
- MCP Python SDK 的 `@server.tool` 装饰器自动处理 schema 生成
- `stdio` transport 适用于本地 Claude Desktop 调用，`sse` 适用于远程/网页调用
- Tool 内部需要初始化 RAG Pipeline（构建检索器、LLM 等），首次调用可能有冷启动延迟
- 错误处理：Tool 执行失败时返回 `isError: true` + 错误信息，不要 crash server
