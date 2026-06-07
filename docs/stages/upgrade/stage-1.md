### 阶段 1：模型升级 — 接入 MiMo-V2-Pro

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-1.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-1.md`、`config.yaml`、`src/generator/siliconflow_llm.py`

**目标**：将 LLM 从 SiliconFlow Qwen3-8B 切换到小米 MiMo-V2-Pro，确保所有 LLM 调用链路正常工作，跑出新的 benchmark 数据。

**前置依赖**：无（本阶段是升级计划的起点）

**任务清单**：

1. 配置层改造
   - `config.yaml`：将 `llm.model` 改为 `"MiMo-V2-Pro"`，新增 `llm.base_url: "https://api.xiaomimimo.com"`
   - `config.yaml`：新增 `mimo` 配置段，包含 `api_key_env: "MIMO_API_KEY"`
   - `src/config.py`：`LLMConfig` 新增 `base_url` 字段（默认 `"https://api.xiaomimimo.com"`）
   - `src/config.py`：`Config` 类新增 `mimo_api_key` property，从环境变量 `MIMO_API_KEY` 读取

2. LLM 模块重构
   - 将 `src/generator/siliconflow_llm.py` 重命名为 `src/generator/mimo_llm.py`
   - 类名 `SiliconFlowLLM` 重命名为 `MimoLLM`
   - `__init__` 参数 `base_url` 默认值改为 MiMo 的 endpoint
   - 错误日志中将 "SiliconFlow" 替换为 "MiMo"
   - 在 `src/generator/__init__.py` 中导出 `MimoLLM`（同时保留 `SiliconFlowLLM` 别名向后兼容）

3. 下游调用点适配
   - `src/api/deps.py`：将 `SiliconFlowLLM` 改为 `MimoLLM`，base_url 从 config 读取
   - `src/rag_pipeline.py`：更新 LLM 实例化代码
   - `src/correction/external_verifier.py`：更新模型引用（如有）
   - `src/correction/rule_checker.py`：更新模型引用（如有）
   - `src/fact_extractor/extractor.py`：更新模型引用（如有）
   - `src/fact_cache/sync.py`：更新 judge_model 配置
   - `src/ui/services.py`：更新 LLM 构建逻辑
   - `src/agent/react.py`：更新 Agent 的 LLM 引用

4. 环境变量更新
   - `.env.example`：新增 `MIMO_API_KEY=your_key_here`
   - `.env`：新增实际 API Key（用户自行填写）
   - `.gitignore`：确认 `.env` 已被忽略

5. 测试适配
   - 全局搜索 `SiliconFlowLLM`，确保无遗漏引用
   - 运行 `pytest tests/ -x -q` 确认所有测试通过
   - 运行 `mypy src/ --config-file mypy.ini` 确认类型检查通过
   - 运行 `ruff check src/ tests/` 确认代码检查通过

6. Benchmark 重跑（如果 API Key 可用）
   - 运行 `python scripts/benchmark.py`
   - 记录结果到 `benchmark_results_v10_mimo.md`
   - 对比 v9（SiliconFlow）数据，确认 Faithfulness 目标 ≥ 0.85

**验收标准**：
- `MimoLLM` 类正常工作，`chat`/`achat`/`stream_chat`/`astream_chat` 四个方法均可用
- 所有现有测试通过（`pytest tests/ -x -q`）
- 无 mypy 错误、无 ruff 错误
- `config.yaml` 中模型配置清晰，base_url 可切换
- 如 API Key 可用，benchmark 结果有记录

**技术备注**：
- MiMo API 兼容 OpenAI 格式，现有 httpx 调用逻辑无需修改核心结构
- 如果 MiMo-V2-Pro 的某些字段名与 OpenAI 标准不同（如 tool_calls 格式），在 `mimo_llm.py` 中做适配层
- 保留 `SiliconFlowLLM` 作为别名，避免一次性改动太多下游代码
