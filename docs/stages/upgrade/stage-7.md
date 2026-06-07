### 阶段 7：全局 Review — 代码审查 + Benchmark + 文档更新

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-7.md，对整个升级进行全面的 Code Review`

**目标**：对阶段 1-6 的全部变更进行全局审查，确保代码质量、架构一致性、安全性，重跑完整 benchmark，更新 README。

**前置依赖**：阶段 1-6 全部完成。

**任务清单**：

1. 代码质量审查
   - 所有新增模块（`mimo_llm.py`、`observability/`、`ui_gradio/`、`mcp_server/`、`neo4j` 相关）的代码质量
   - 检查命名一致性：是否使用 CONTEXT.md 中的统一术语
   - 检查错误处理：所有外部调用（MiMo API、Neo4j、Langfuse）是否有降级策略
   - 检查类型注解完整性
   - 运行 `ruff check src/ tests/ --fix`
   - 运行 `mypy src/ --config-file mypy.ini`

2. 架构一致性
   - 所有新模块是否遵循现有目录结构和分层约定
   - 依赖方向是否正确（无循环依赖）
   - 配置管理是否统一（都通过 `config.yaml` + `Config` 类）
   - 新增 ADR 记录关键决策：
     - `docs/adr/0004-mimo-model-migration.md`
     - `docs/adr/0005-langfuse-observability.md`
     - `docs/adr/0006-gradio-frontend.md`
     - `docs/adr/0007-mcp-server-protocol.md`
     - `docs/adr/0008-neo4j-graph-store.md`

3. 安全审查
   - `.env` 和 `.gitignore`：确认无 API Key 泄露到 Git
   - MCP Server：确认无命令注入风险
   - WebSocket：确认输入校验、连接限制
   - Neo4j：确认 Cypher 参数化查询（防注入）

4. 完整 Benchmark
   - 运行 `python scripts/benchmark.py`，生成最终 benchmark 结果
   - 对比 v9（原始 SiliconFlow）和 v10+（MiMo 升级后）数据
   - 记录到 `benchmark_results_final.md`
   - 目标：Faithfulness ≥ 0.85（如果未达到，分析原因并记录）

5. README 更新
   - 更新技术栈描述（MiMo、Neo4j、Langfuse、Gradio、MCP）
   - 更新架构图
   - 更新启动命令（推荐 Gradio）
   - 新增 MCP 集成说明
   - 新增 benchmark 结果对比表
   - 更新截图（如果 Gradio UI 已完成）

6. CONTEXT.md 更新
   - 将阶段 1-6 新增的所有术语补充到词汇表
   - 更新关键文件路径表

7. 清理
   - 删除所有 Checkpoint 文件（如存在）
   - 删除 `plan-upgrade.md` 中已完成阶段的临时文件
   - 确认 `docs/stages/upgrade/` 中的阶段文件完整保留（供参考）
   - 确认 Streamlit 入口（`app.py`）是否保留（决定并执行）

**验收标准**：
- `ruff check` 零错误
- `mypy` 零错误
- `pytest tests/ -x -q` 全部通过
- Benchmark 结果有完整记录
- README 反映最新的技术栈和功能
- 所有 ADR 已创建
- 无 API Key 泄露
- CONTEXT.md 术语完整

**最终交付物**：
- 完整的升级后代码（可运行）
- Benchmark 对比报告
- 更新的 README
- ADR 文档集
- MCP 集成指南
