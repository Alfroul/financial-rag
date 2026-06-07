"""Evaluation tab — RAGAS evaluation and real-time metrics display."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import Config
from src.metrics.collector import MetricsCollector
from src.ui_gradio.services import build_pipeline

logger = logging.getLogger(__name__)
config = Config()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _format_metrics_html(summary: dict) -> str:
    if summary["total_queries"] == 0:
        return "<div style='color:var(--text-muted);'>暂无查询数据</div>"

    total_tokens = summary["total_input_tokens"] + summary["total_output_tokens"]
    return f"""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:8px 0;">
        <div class="metric-card">
            <div class="metric-value">{summary['total_queries']}</div>
            <div class="metric-label">Total Queries</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary['avg_latency_ms']:.0f}ms</div>
            <div class="metric-label">Avg Latency</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary['cache_hit_rate'] * 100:.1f}%</div>
            <div class="metric-label">Cache Hit</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary['p50_latency_ms']:.0f}ms</div>
            <div class="metric-label">P50 Latency</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary['p95_latency_ms']:.0f}ms</div>
            <div class="metric-label">P95 Latency</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_tokens}</div>
            <div class="metric-label">Total Tokens</div>
        </div>
    </div>
    """


def _avg_sources(summary: dict) -> float:
    recent: list = summary.get("recent_queries", [])
    if not recent:
        return 0.0
    return float(sum(r.get("num_sources", 0) for r in recent) / len(recent))


def _load_metrics() -> tuple:
    try:
        collector = MetricsCollector()
        summary = collector.summary()
        metrics = _format_metrics_html(summary)
        recent = summary.get("recent_queries", [])
        if recent:
            rows = [
                [r["question"][:30], r["total_ms"], r["retrieve_ms"], r["generate_ms"]]
                for r in recent
            ]
        else:
            rows = []
        return metrics, rows
    except Exception:
        return "<div style='color:var(--text-muted);'>Metrics unavailable</div>", []


# Sidebar input order for eval (subset needed for pipeline building)
_EVAL_SIDEBAR_KEYS = [
    "api_key", "model", "temperature", "max_tokens",
    "strategy", "top_k", "score_threshold",
    "reranker_enabled", "reranker_top_n",
    "query_rewrite", "cache_enabled", "self_correction",
]


def create_eval_tab(sidebar_components: dict) -> None:
    """Create evaluation tab with RAGAS evaluation and metrics."""
    sidebar_components["api_key"]
    sidebar_refs = [sidebar_components[k] for k in _EVAL_SIDEBAR_KEYS]

    with gr.Tabs():
        with gr.Tab("RAGAS Evaluation"):
            gr.Markdown("### RAGAS Evaluation")
            gr.Markdown("Automated quality assessment using RAGAS framework")

            eval_source = gr.Radio(
                ["内置数据集", "上传 JSON 文件"],
                value="内置数据集",
                label="评估数据来源",
            )

            eval_file = gr.File(
                label="上传 JSON 评估文件",
                file_types=[".json"],
                visible=False,
            )

            dataset_info = gr.Markdown(value="")

            with gr.Accordion("预览数据集", open=False):
                preview_df = gr.Dataframe(
                    headers=["Question", "Reference"],
                    interactive=False,
                )

            eval_btn = gr.Button("开始评估", variant="primary")
            eval_progress = gr.Markdown("")

            eval_result_df = gr.Dataframe(
                headers=["Metric", "Score"],
                datatype=["str", "number"],
                label="评估结果",
                interactive=False,
            )

            avg_score = gr.Markdown("")
            csv_download = gr.File(label="Export Results (CSV)", visible=False)

            def toggle_upload(source):
                return gr.update(visible=(source == "上传 JSON 文件"))

            eval_source.change(toggle_upload, [eval_source], [eval_file])

            def load_dataset(source, uploaded_file):
                eval_path = _PROJECT_ROOT / "data" / "eval" / "financial_qa_eval.json"
                samples = None

                if source == "内置数据集":
                    if eval_path.exists():
                        with open(eval_path, encoding="utf-8") as f:
                            samples = json.load(f)
                        info = f"已加载内置数据集: **{len(samples)}** 条问答对"
                    else:
                        return "内置评估数据集不存在", [], None
                else:
                    if not uploaded_file:
                        return "请上传 JSON 文件", [], None
                    try:
                        path = Path(uploaded_file.name) if hasattr(uploaded_file, "name") else Path(uploaded_file)
                        content = path.read_text(encoding="utf-8")
                        samples = json.loads(content)
                        info = f"已加载上传数据集: **{len(samples)}** 条问答对"
                    except Exception as e:
                        return f"解析 JSON 失败: {e}", [], None

                preview = [
                    [s.get("question", "")[:50], (s.get("reference", "") or "")[:50]]
                    for s in (samples or [])[:10]
                ]
                return info, preview, samples

            eval_source.change(
                load_dataset, [eval_source, eval_file],
                [dataset_info, preview_df, gr.State(None)],
            )
            eval_file.change(
                load_dataset, [eval_source, eval_file],
                [dataset_info, preview_df, gr.State(None)],
            )

            def run_eval(source, uploaded_file, *sidebar_values):
                vals = dict(zip(_EVAL_SIDEBAR_KEYS, sidebar_values))
                api_key = vals["api_key"]

                if not api_key:
                    yield "请先输入 API Key", gr.update(), "", gr.update(visible=False)
                    return

                # Load dataset
                eval_path = _PROJECT_ROOT / "data" / "eval" / "financial_qa_eval.json"
                samples = None
                if source == "内置数据集":
                    if eval_path.exists():
                        with open(eval_path, encoding="utf-8") as f:
                            samples = json.load(f)
                else:
                    if uploaded_file:
                        try:
                            path = Path(uploaded_file.name) if hasattr(uploaded_file, "name") else Path(uploaded_file)
                            samples = json.loads(path.read_text(encoding="utf-8"))
                        except Exception:
                            pass

                if not samples:
                    yield "无法加载评估数据集", gr.update(), "", gr.update(visible=False)
                    return

                try:
                    from src.evaluation.ragas_eval import RAGEvaluator
                    evaluator = RAGEvaluator(api_key=api_key)
                except ImportError as e:
                    yield f"缺少评估依赖: {e}", gr.update(), "", gr.update(visible=False)
                    return

                yield "Building pipeline...", gr.update(), "", gr.update(visible=False)

                try:
                    pipeline = build_pipeline(**vals)
                except Exception as e:
                    yield f"Pipeline 构建失败: {e}", gr.update(), "", gr.update(visible=False)
                    return

                questions = [s["question"] for s in samples]
                responses = []
                all_contexts = []

                for i, sample in enumerate(samples):
                    progress = f"Querying {i + 1}/{len(samples)}..."
                    yield progress, gr.update(), "", gr.update(visible=False)
                    try:
                        result = pipeline.query(sample["question"])
                        answer = result.get("answer", "")
                        responses.append(str(answer) if answer else "")
                        raw_sources = result.get("sources", [])
                        all_contexts.append([str(src["content"]) for src in raw_sources])
                    except Exception as e:
                        logger.warning("Q%d failed: %s", i + 1, e)
                        responses.append("")
                        all_contexts.append([])

                yield "Running RAGAS evaluation...", gr.update(), "", gr.update(visible=False)

                ref_list = [s.get("reference") or "" for s in samples]
                scores = evaluator.evaluate(
                    questions=questions,
                    responses=responses,
                    contexts=all_contexts,
                    references=ref_list,
                )

                metrics_rows = [
                    [k.replace("_", " ").title(), round(v, 4)]
                    for k, v in scores.items()
                ]

                avg = sum(scores.values()) / len(scores) if scores else 0

                csv_buffer = io.StringIO()
                rows = [
                    {"question": q, "response": r, **{k: "" for k in scores}}
                    for q, r in zip(questions, responses)
                ]
                pd.DataFrame(rows).to_csv(csv_buffer, index=False)
                csv_path = str(_PROJECT_ROOT / "data" / "eval" / "rag_eval_results.csv")
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(csv_buffer.getvalue())

                yield "Evaluation complete!", metrics_rows, f"**Average Score: {avg:.4f}**", csv_path

            eval_btn.click(
                run_eval,
                [eval_source, eval_file] + sidebar_refs,
                [eval_progress, eval_result_df, avg_score, csv_download],
            )

        with gr.Tab("Real-time Metrics"):
            gr.Markdown("### System Metrics")
            gr.Markdown("Real-time query performance monitoring")

            metrics_html = gr.HTML(value=_load_metrics()[0])

            with gr.Row():
                refresh_metrics_btn = gr.Button("刷新")
                clear_metrics_btn = gr.Button("Clear Metrics")

            latency_df = gr.Dataframe(
                headers=["Query", "Total (ms)", "Retrieve (ms)", "Generate (ms)"],
                label="Recent Query Latency",
                interactive=False,
                value=_load_metrics()[1],
            )

            def refresh_metrics():
                return _load_metrics()

            def clear_metrics():
                MetricsCollector().clear()
                return _load_metrics()

            refresh_metrics_btn.click(refresh_metrics, None, [metrics_html, latency_df])
            clear_metrics_btn.click(clear_metrics, None, [metrics_html, latency_df])
