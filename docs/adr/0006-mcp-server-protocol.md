# ADR 0006: 用 Gradio 4.x 替换 Streamlit 作为前端

## 状态

已批准

## 上下文

Streamlit 在 2026 年的前端展示中显得简陋：
- 全页面重渲染机制导致交互卡顿
- 自定义能力有限，难以实现专业的 UI 效果
- 面试官对 Streamlit 的第一印象是"demo 工具"

## 决策

使用 Gradio 4.x 替换 Streamlit，理由：
1. `gr.Blocks` 支持完全自定义布局
2. 原生支持流式输出（generator yield）
3. 自带暗色主题，微调成本低
4. 社区活跃，2026 年生态比 Streamlit 更好
5. 不用重写后端，只替换 UI 层

## 后果

- 正面：更专业的 UI、更流畅的交互、更好的流式体验
- 负面：需要重写 UI 层代码，Gradio 的一些高级组件学习成本
- 缓解：保留 Streamlit 入口，两套共存直到验证完毕
