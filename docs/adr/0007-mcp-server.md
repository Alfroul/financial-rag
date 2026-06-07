# ADR 0007: 将 RAG 系统包装为 MCP Server

## 状态

已批准

## 上下文

2026 年 MCP (Model Context Protocol) 已成为 Agent 交互的标准协议。一个 RAG 项目如果不能被 MCP Client（Claude Desktop、Cursor 等）调用，缺少生态接入能力。

## 决策

使用 `mcp` Python SDK 将 RAG 核心能力包装为 MCP Server，暴露 3 个 Tools（financial_search、knowledge_graph_query、financial_analysis）。

## 后果

- 正面：简历上可写"MCP 协议接入"，面试可现场用 Claude Desktop 演示
- 负面：新增 `mcp` 依赖，MCP SDK 版本迭代快可能需要适配
- 缓解：MCP Server 是独立模块，不影响主流程
