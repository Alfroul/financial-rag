"""Financial RAG — Gradio 4.x UI entry point.

Usage:
    python -m src.ui_gradio.app
    gradio src/ui_gradio/app.py
"""

from __future__ import annotations

import logging

import gradio as gr

from src.config import Config
from src.metrics.collector import MetricsCollector
from src.ui_gradio.chat import create_chat_tab
from src.ui_gradio.docs import create_docs_tab
from src.ui_gradio.eval import create_eval_tab
from src.ui_gradio.graph import create_graph_tab
from src.ui_gradio.services import get_vectorstore
from src.ui_gradio.styles import CUSTOM_CSS

logger = logging.getLogger(__name__)
config = Config()

# Available MiMo models
_MODEL_CHOICES = ["mimo-v2-pro", "mimo-v2-omni"]


def _get_kb_stats() -> str:
    try:
        store = get_vectorstore()
        stats = store.get_stats()
        return f"Indexed Chunks: {stats['document_count']}"
    except Exception:
        return "Connection Failed"


def _get_metrics_summary() -> str:
    try:
        collector = MetricsCollector()
        summary = collector.summary()
        if summary["total_queries"] == 0:
            return "No queries yet"
        total_tokens = summary["total_input_tokens"] + summary["total_output_tokens"]
        return (
            f"Queries: {summary['total_queries']} | "
            f"Avg: {summary['avg_latency_ms']:.0f}ms | "
            f"P95: {summary['p95_latency_ms']:.0f}ms | "
            f"Cache: {summary['cache_hit_rate'] * 100:.1f}% | "
            f"Tokens: {total_tokens}"
        )
    except Exception:
        return "Metrics unavailable"


def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.Color(
            c50="#eff6ff", c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd",
            c400="#60a5fa", c500="#3b82f6", c600="#1a73e8",
            c700="#1d4ed8", c800="#1e40af", c900="#1e3a8a", c950="#172554",
        ),
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Noto Sans SC"),
    ).set(
        body_background_fill="#0A0E27",
        body_text_color="#E8E6E3",
        body_text_color_subdued="#9CA3AF",
        background_fill_primary="#0D1421",
        background_fill_secondary="#111832",
        border_color_primary="#1E293B",
        border_color_accent="#1a73e8",
        input_background_fill="#111832",
        input_border_color="#1E293B",
        button_primary_background_fill="#1a73e8",
        button_primary_background_fill_hover="#1565c0",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="transparent",
        button_secondary_border_color="#C9A84C",
        button_secondary_text_color="#C9A84C",
        block_title_text_color="#C9A84C",
        block_label_text_color="#9CA3AF",
    )

    with gr.Blocks(  # noqa: SIM117
        theme=theme,
        css=CUSTOM_CSS,
        title="Financial RAG - Gradio",
    ) as demo:

        # ===== Layout: sidebar + main content =====
        with gr.Row(equal_height=False):
            # ---------- Sidebar ----------
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar-col"]):

                # Branding
                gr.HTML(
                    '<div style="margin-bottom:0.6rem;">'
                    '<div class="sidebar-title">Financial RAG</div>'
                    '<div class="sidebar-subtitle">Intelligent Query System</div>'
                    '</div>'
                    '<hr style="border-color:rgba(201,168,76,0.2);margin:0.5rem 0;">'
                )

                # API Key
                gr.HTML('<div class="section-header">API CONFIGURATION</div>')
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="输入 API Key...",
                )
                api_status = gr.HTML(
                    '<span class="status-waiting">● AWAITING KEY</span>'
                )

                def update_api_status(key):
                    if key:
                        return '<span class="status-connected">● CONNECTED</span>'
                    return '<span class="status-waiting">● AWAITING KEY</span>'

                api_key.change(update_api_status, [api_key], [api_status])

                # Model Parameters
                gr.HTML('<div class="section-header">MODEL PARAMETERS</div>')
                model = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=config.llm.model if config.llm.model in _MODEL_CHOICES else _MODEL_CHOICES[0],
                    label="Model",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=config.llm.temperature, step=0.1,
                    label="Temperature",
                )
                max_tokens = gr.Slider(
                    minimum=256, maximum=4096, value=config.llm.max_tokens, step=256,
                    label="Max Tokens",
                )

                # Query Mode
                gr.HTML('<div class="section-header">QUERY MODE</div>')
                query_mode = gr.Radio(
                    choices=["qa", "agent"],
                    value="qa",
                    label="Query Mode",
                    info="问答模式 vs 分析模式(Agent 多步推理)",
                )

                # Retrieval
                gr.HTML('<div class="section-header">RETRIEVAL</div>')
                strategy = gr.Radio(
                    choices=["hybrid", "vector", "bm25"],
                    value=config.hybrid.strategy,
                    label="Retrieval Mode",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=20, value=config.retriever.top_k, step=1,
                    label="Top-K",
                )
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=config.retriever.score_threshold, step=0.05,
                    label="Score Threshold",
                )

                # Reranker
                gr.HTML('<div class="section-header">RERANKER</div>')
                reranker_enabled = gr.Checkbox(
                    value=config.reranker.enabled,
                    label="Enable Reranker",
                )
                reranker_top_n = gr.Slider(
                    minimum=1, maximum=20, value=config.reranker.top_n, step=1,
                    label="Rerank Top-N",
                )

                query_rewrite = gr.Checkbox(
                    value=config.rag.query_rewrite,
                    label="Enable Query Rewriting",
                    info="多轮对话指代消解",
                )

                cache_enabled = gr.Checkbox(
                    value=config.cache.enabled,
                    label="Enable Query Cache",
                    info="语义缓存，减少 API 调用",
                )

                self_correction = gr.Checkbox(
                    value=config.self_correction.enabled,
                    label="自我修正",
                    info="四层幻觉检测与修正",
                )

                # Chunking
                gr.HTML('<div class="section-header">CHUNKING</div>')
                chunk_size = gr.Slider(
                    minimum=128, maximum=2048, value=config.chunker.chunk_size, step=64,
                    label="Chunk Size",
                )
                chunk_overlap = gr.Slider(
                    minimum=0, maximum=512, value=config.chunker.chunk_overlap, step=32,
                    label="Chunk Overlap",
                )

                # Knowledge Base Stats
                gr.HTML('<div class="section-header">KNOWLEDGE BASE</div>')
                gr.HTML(value=f'<div class="metric-card"><div class="metric-value">{_get_kb_stats()}</div></div>')

                # System Metrics
                gr.HTML('<div class="section-header">SYSTEM METRICS</div>')
                gr.HTML(value=f'<div style="font-size:0.75rem;color:var(--text-muted);">{_get_metrics_summary()}</div>')

                # Branding Footer
                gr.HTML(
                    '<div class="brand-footer">'
                    '<div class="brand-name">Financial RAG</div>'
                    '<div class="brand-version">v2.0 // Powered by MiMo + Gradio</div>'
                    '</div>'
                )

            # ---------- Main Content ----------
            with gr.Column(scale=4):
                gr.HTML(
                    '<div style="text-align:center;padding:0.3rem 0 0.2rem 0;'
                    'border-bottom:1px solid rgba(26,115,232,0.2);">'
                    '<div style="font-size:0.95rem;font-weight:600;color:#1a73e8;'
                    'text-transform:uppercase;letter-spacing:3px;">'
                    'Financial RAG — Query Terminal</div>'
                    '<div style="font-size:0.7rem;color:#4B5563;'
                    'letter-spacing:1px;text-transform:uppercase;">'
                    'Retrieval-Augmented Generation for Financial Intelligence</div>'
                    '</div>'
                )

                with gr.Tabs():
                    with gr.Tab("智能问答"):
                        # Build sidebar component dict for chat tab
                        sidebar_components = {
                            "api_key": api_key,
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "strategy": strategy,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            "reranker_enabled": reranker_enabled,
                            "reranker_top_n": reranker_top_n,
                            "query_rewrite": query_rewrite,
                            "cache_enabled": cache_enabled,
                            "self_correction": self_correction,
                            "query_mode": query_mode,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                        }
                        create_chat_tab(sidebar_components)

                    with gr.Tab("文档管理"):
                        create_docs_tab(sidebar_components)

                    with gr.Tab("知识图谱"):
                        create_graph_tab()

                    with gr.Tab("系统评估"):
                        create_eval_tab(sidebar_components)

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
