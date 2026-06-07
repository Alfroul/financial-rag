### 阶段 3：前端升级 — Gradio 4.x 替换 Streamlit

> **独立会话指令**：`阅读 CONTEXT.md 和 docs/stages/upgrade/stage-3.md，完成所有任务后确认完成`
>
> **开工前只需阅读**：`CONTEXT.md`、`docs/stages/upgrade/stage-3.md`、`src/ui/`（现有 Streamlit UI 参考）、`app.py`

**目标**：用 Gradio 4.x 构建新前端，替换 Streamlit。保留原有三个 Tab 功能（智能问答、文档管理、系统评估），同时提升视觉效果和交互体验。

**前置依赖**：阶段 1 已完成（LLM 已切换到 MiMo）。

**任务清单**：

1. 依赖安装
   - `requirements.txt` 新增 `gradio>=4.0.0`
   - 确认与现有依赖无冲突

2. Gradio 应用骨架
   - 创建 `src/ui_gradio/__init__.py`
   - 创建 `src/ui_gradio/app.py`：Gradio 应用主入口
     - 使用 `gr.Blocks` 布局（非 `gr.Interface`，更灵活）
     - 自定义 CSS 主题（金融深蓝色调）
     - `gr.Tab` 实现三个标签页
   - 入口命令：`python -m src.ui_gradio.app` 或 `gradio src/ui_gradio/app.py`

3. 智能问答 Tab
   - 创建 `src/ui_gradio/chat.py`
   - `gr.ChatInterface` 或自定义 `gr.Chatbot` + `gr.Textbox` 组合
   - 支持流式输出（generator yield）
   - 显示来源引用（类似现有 Streamlit 的 source expander）
   - 侧边栏参数面板：top_k、score_threshold、检索策略、reranker 开关等
   - Agent 模式切换（问答/分析）

4. 文档管理 Tab
   - 创建 `src/ui_gradio/docs.py`
   - `gr.File` 支持拖拽上传
   - `gr.Dataframe` 显示已索引文件列表
   - 索引进度条（`gr.Progress`）
   - 删除文档功能

5. 系统评估 Tab
   - 创建 `src/ui_gradio/eval.py`
   - `gr.Dataframe` 显示 RAGAS 评测结果表格
   - `gr.Plot` 或 `gr.BarPlot` 显示指标对比图
   - 触发评测按钮 + 进度条

6. 样式定制
   - 自定义 CSS：深色主题，金融蓝（#1a73e8）为主色调
   - 字体：中文优先思源黑体/Noto Sans SC
   - 布局：左侧参数面板 + 右侧主内容区
   - 响应式设计，适配不同屏幕宽度

7. 清理与共存
   - 保留 `app.py`（Streamlit 入口）不删除，两套 UI 共存
   - 更新 `README.md` 中的启动命令，推荐 Gradio
   - `docker-compose.yml` 新增 Gradio 服务端口（7860）

8. 测试
   - 手动启动 `python -m src.ui_gradio.app`，验证三个 Tab 功能正常
   - 验证流式输出效果
   - 验证文件上传和索引
   - 运行 `pytest tests/ -x -q`（后端无变化，应全部通过）

**验收标准**：
- `python -m src.ui_gradio.app` 启动成功，Gradio UI 可访问
- 三个 Tab 功能完整，与 Streamlit 版本对等
- 流式输出有真实的打字机效果
- 自定义 CSS 生效，界面风格统一
- 后端测试全部通过

**技术备注**：
- Gradio 4.x 的 `gr.Blocks` 支持完全自定义布局，比 `gr.Interface` 灵活得多
- 流式输出用 generator yield + `gr.Chatbot.stream_chat` 模式
- Gradio 自带暗色主题 `gr.themes.Soft()`，在其基础上微调即可
- 不要删 Streamlit 入口，两套共存到 Review 阶段再决定
