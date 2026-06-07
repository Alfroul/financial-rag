# MCP Server 集成指南

本项目的 RAG 能力已封装为 MCP Server，可被 Claude Desktop、Cursor 等 MCP Client 直接调用。

## 启动 Server

在项目根目录下运行：

```bash
# stdio 模式（Claude Desktop 使用）
python -m src.mcp_server.server

# SSE 模式（远程调用）
python -m src.mcp_server.server --sse
python -m src.mcp_server.server --sse --host 0.0.0.0 --port 9090
```

## 可用工具

| 工具名 | 说明 | 参数 |
|--------|------|------|
| `financial_search` | 检索金融文档知识库 | `query` (str), `top_k` (int, 默认 5) |
| `knowledge_graph_query` | 查询实体关系图谱 | `entity` (str), `relation` (str, 可选) |
| `financial_analysis` | 完整 RAG 分析问答 | `question` (str) |

## Claude Desktop 配置

编辑 `claude_desktop_config.json`（路径因系统而异）：

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "financial-rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/your/deep-learning/rag"
    }
  }
}
```

将 `cwd` 替换为你的实际项目路径。

配置完成后重启 Claude Desktop，在对话中即可使用 RAG 工具。

## Cursor 配置

在 Cursor 设置中添加 MCP Server：

1. 打开 Settings → MCP
2. 点击 "Add new MCP server"
3. 填写：
   - **Name**: `financial-rag`
   - **Type**: `command`
   - **Command**: `python -m src.mcp_server.server`
   - **Working Directory**: 项目根目录路径

## 使用示例

配置完成后，在 MCP Client 中发送以下类型的消息会自动触发工具调用：

- "帮我查一下茅台的ROE" → 调用 `financial_search`
- "茅台和五粮液有什么关系" → 调用 `knowledge_graph_query`
- "对比分析茅台和五粮液2024年盈利能力" → 调用 `financial_analysis`

## 配置说明

`config.yaml` 中的 `mcp_server` 段：

```yaml
mcp_server:
  host: "localhost"   # SSE 模式监听地址
  port: 8080          # SSE 模式监听端口
  transport: "stdio"  # 默认传输方式：stdio 或 sse
```

## 注意事项

- 首次调用会有冷启动延迟（需要初始化检索器、LLM 等组件）
- `knowledge_graph_query` 需要 `graph.enabled: true` 才能使用
- 确保 `.env` 文件中配置了 `MIMO_API_KEY`
- stdio 模式下 Server 日志输出到 stderr，不会干扰 MCP 协议通信
