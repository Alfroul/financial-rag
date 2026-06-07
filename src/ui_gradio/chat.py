"""Chat tab — streaming RAG Q&A with source display, correction, and agent mode."""

from __future__ import annotations

import logging

import gradio as gr

from src.config import Config
from src.generator.mimo_llm import (
    LLMAuthError,
    LLMError,
    LLMQuotaError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.rag_pipeline import RAGPipelineError
from src.ui_gradio.services import build_pipeline

logger = logging.getLogger(__name__)
config = Config()

# Sidebar input order (must match app.py)
_SIDEBAR_KEYS = [
    "api_key", "model", "temperature", "max_tokens",
    "strategy", "top_k", "score_threshold",
    "reranker_enabled", "reranker_top_n",
    "query_rewrite", "cache_enabled", "self_correction", "query_mode",
]


def _format_sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    parts = []
    for i, src in enumerate(sources):
        meta = src.get("metadata", {})
        title = meta.get("title", meta.get("source", f"Source {i + 1}"))
        score = src.get("score", 0)
        content = src.get("content", "")
        snippet = content[:300] + ("..." if len(content) > 300 else "")
        # Score bar color: green > 0.7, yellow > 0.4, red otherwise
        bar_pct = min(max(score * 100, 0), 100)
        if score >= 0.7:
            bar_color = "var(--success)"
        elif score >= 0.4:
            bar_color = "var(--warning)"
        else:
            bar_color = "var(--error)"
        parts.append(
            f'<div class="source-card">'
            f'<div class="source-title">[{i + 1}] {title}</div>'
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="flex:1;background:var(--border);border-radius:3px;height:6px;overflow:hidden;">'
            f'<div style="width:{bar_pct:.0f}%;height:100%;background:{bar_color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-size:0.7rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;'
            f'min-width:52px;text-align:right;">{score:.4f}</span>'
            f'</div>'
            f'<div class="source-content">{snippet}</div>'
            f"</div>"
        )
    return "".join(parts)


def _format_correction_html(correction) -> str:
    if not correction:
        return ""
    passed = correction.passed
    confidence = correction.confidence
    status_cls = "correction-pass" if passed else "correction-flag"
    status_text = "通过" if passed else "有疑点"
    icon = "&#10003;" if passed else "&#9888;"

    # Confidence bar
    conf_pct = min(max(confidence * 100, 0), 100)
    conf_color = "var(--success)" if confidence >= 0.7 else ("var(--warning)" if confidence >= 0.4 else "var(--error)")

    html = (
        f'<div style="margin-bottom:8px;">'
        f'<span class="{status_cls}">{icon} 修正结果: {status_text}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
        f'<span style="font-size:0.75rem;color:var(--text-muted);">置信度</span>'
        f'<div style="flex:1;background:var(--border);border-radius:3px;height:6px;overflow:hidden;">'
        f'<div style="width:{conf_pct:.0f}%;height:100%;background:{conf_color};border-radius:3px;"></div>'
        f'</div>'
        f'<span style="font-size:0.75rem;color:var(--text-muted);'
        f'font-family:JetBrains Mono,monospace;">{confidence:.0%}</span>'
        f'</div>'
    )

    if correction.flagged_claims:
        html += "<div style='margin:4px 0;'><b>标记的断言:</b></div>"
        for claim in correction.flagged_claims:
            html += f'<div style="color:var(--error);padding:2px 8px;">- {claim}</div>'

    # Layer pipeline visualization
    layer_results = correction.layer_results or {}
    if layer_results:
        html += '<div style="margin-top:10px;"><b>修正流程:</b></div>'
        html += '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">'

        layers = [
            ("retrieval_quality", "检索门控"),
            ("rule_issues", "规则预检"),
            ("nli", "Claim NLI"),
            ("external", "外部验证"),
        ]
        for key, label in layers:
            if key in layer_results:
                if key == "retrieval_quality":
                    rq = layer_results[key]
                    layer_ok = rq.level in ("good", "medium")
                elif key == "rule_issues":
                    issues = layer_results[key]
                    layer_ok = not issues
                elif key == "nli":
                    verdicts = layer_results[key]
                    layer_ok = all(v.supported for v in verdicts)
                elif key == "external":
                    ext = layer_results[key]
                    layer_ok = all(v.supported for v in ext)
                else:
                    layer_ok = True

                dot_color = "var(--success)" if layer_ok else "var(--error)"
                html += (
                    f'<div style="background:var(--bg-input);border:1px solid var(--border);'
                    f'border-radius:4px;padding:4px 10px;font-size:0.75rem;">'
                    f'<span style="color:{dot_color};">&#9679;</span> {label}</div>'
                )

        html += '</div>'

        if "retrieval_quality" in layer_results:
            rq = layer_results["retrieval_quality"]
            html += (
                f'<div style="margin-top:6px;font-size:0.8rem;">'
                f"<b>检索质量:</b> {rq.level} "
                f"(top: {rq.top_score:.2f}, avg: {rq.avg_score:.2f}, "
                f"sources: {rq.num_sources})"
                f"</div>"
            )

        if "retries" in layer_results:
            html += f'<div style="font-size:0.8rem;"><b>重试次数:</b> {layer_results["retries"]}</div>'

        if "rule_issues" in layer_results:
            issues = layer_results["rule_issues"]
            if issues:
                html += f'<div style="font-size:0.8rem;"><b>规则检查问题:</b> {len(issues)} 项</div>'
                for issue in issues[:5]:
                    sev = issue.get("severity", "MEDIUM")
                    msg = issue.get("message", "")
                    html += (
                        f'<div style="padding:1px 8px;font-size:0.75rem;">'
                        f'  - [{sev}] {msg}</div>'
                    )

        if "nli" in layer_results:
            verdicts = layer_results["nli"]
            unsupported = sum(1 for v in verdicts if not v.supported)
            html += (
                f'<div style="font-size:0.8rem;">'
                f'<b>NLI 验证:</b> {len(verdicts)} 条断言, '
                f'{unsupported} 条未通过</div>'
            )

        if "external" in layer_results:
            ext = layer_results["external"]
            ext_unsupported = sum(1 for v in ext if not v.supported)
            html += f'<div style="font-size:0.8rem;"><b>外部验证:</b> {len(ext)} 条, {ext_unsupported} 条未通过</div>'

    return html


def _format_agent_steps_html(steps: list[dict]) -> str:
    if not steps:
        return ""
    parts = ['<div class="agent-timeline">']
    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        connector = "" if is_last else '<div class="timeline-connector"></div>'
        parts.append(
            f'<div class="timeline-step">'
            f'<div class="timeline-dot"></div>'
            f'<div class="timeline-content">'
            f'<div class="step-label">Step {i + 1}</div>'
        )
        thought = step.get("thought", "")
        if thought:
            parts.append(f'<div class="step-content">Thought: {thought}</div>')
        action = step.get("action", "")
        if action:
            parts.append(f'<div style="color:var(--accent-blue);">Action: <code>{action}</code></div>')
        observation = step.get("observation", "")
        if observation:
            obs_snippet = observation[:500] + ("..." if len(observation) > 500 else "")
            parts.append(f'<div style="color:var(--text-muted);">Observation: {obs_snippet}</div>')
        parts.append(f'</div></div>{connector}')
    parts.append("</div>")
    return "".join(parts)


def _gradio_history_to_chat_history(history: list) -> list[dict]:
    """Convert Gradio [(user, assistant), ...] to pipeline [{"role":..., "content":...}, ...]."""
    result: list[dict] = []
    for user_msg, assistant_msg in history:
        result.append({"role": "user", "content": user_msg})
        if assistant_msg:
            result.append({"role": "assistant", "content": assistant_msg})
    return result


def create_chat_tab(sidebar_components: dict) -> None:
    """Create chat tab UI and wire up events.

    sidebar_components: dict mapping key names to Gradio component references.
    """
    chatbot = gr.Chatbot(height=520, label="对话", bubble_full_width=False)

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="请输入您的问题...",
            show_label=False,
            scale=5,
            autofocus=True,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空", scale=1)

    with gr.Accordion("References", open=False):
        sources_html = gr.HTML(value="")

    with gr.Accordion("自我修正报告", open=False):
        correction_html = gr.HTML(value="")

    with gr.Accordion("Agent 思考过程", open=False):
        agent_html = gr.HTML(value="")

    # Collect sidebar component references in known order
    sidebar_refs = [sidebar_components[k] for k in _SIDEBAR_KEYS]

    def user_submit(message, history):
        return "", history + [[message, None]]

    def bot_response(history, *sidebar_values):
        vals = dict(zip(_SIDEBAR_KEYS, sidebar_values))
        api_key = vals["api_key"]

        if not history or not history[-1][0]:
            yield history, "", "", ""
            return

        message = history[-1][0]

        if not api_key:
            history[-1] = [message, "请先在左侧面板输入 API Key"]
            yield history, "", "", ""
            return

        chat_history = _gradio_history_to_chat_history(history[:-1])

        try:
            pipeline = build_pipeline(
                api_key=api_key,
                model=vals["model"],
                temperature=vals["temperature"],
                max_tokens=vals["max_tokens"],
                strategy=vals["strategy"],
                top_k=vals["top_k"],
                score_threshold=vals["score_threshold"],
                reranker_enabled=vals["reranker_enabled"],
                reranker_top_n=vals["reranker_top_n"],
                query_rewrite=vals["query_rewrite"],
                cache_enabled=vals["cache_enabled"],
                self_correction=vals["self_correction"],
            )

            is_agent = vals["query_mode"] == "agent"
            use_correction = vals["self_correction"]

            if is_agent:
                result = pipeline.agent_query(task=message)
                answer = str(result.get("answer", ""))
                steps = result.get("steps", [])
                history[-1] = [message, answer]
                yield history, "", "", _format_agent_steps_html(steps)

            elif use_correction:
                # Stream the answer first for responsiveness, then run correction
                from src.correction.pipeline import SelfCorrectingPipeline

                inner = pipeline.inner_pipeline if isinstance(
                    pipeline, SelfCorrectingPipeline
                ) else pipeline

                sources: list[dict] = []
                partial = ""
                for chunk in inner.stream_query(
                    question=message, chat_history=chat_history,
                ):
                    if chunk["type"] == "sources":
                        sources = chunk["sources"]
                    elif chunk["type"] == "answer":
                        partial += chunk["content"]
                        history[-1] = [message, partial]
                        yield history, "", "", ""

                # Now run correction on the complete answer
                corrected = pipeline.query(
                    question=message, chat_history=chat_history,
                )
                corrected_answer = str(corrected.get("answer", partial))
                correction = corrected.get("correction")

                # Update with corrected answer if it changed
                if corrected_answer != partial:
                    history[-1] = [message, corrected_answer]
                yield (
                    history,
                    _format_sources_html(sources),
                    _format_correction_html(correction),
                    "",
                )

            else:
                sources: list[dict] = []
                partial = ""
                for chunk in pipeline.stream_query(
                    question=message, chat_history=chat_history,
                ):
                    if chunk["type"] == "sources":
                        sources = chunk["sources"]
                    elif chunk["type"] == "answer":
                        partial += chunk["content"]
                        history[-1] = [message, partial]
                        yield history, "", "", ""
                # Final yield with sources
                yield history, _format_sources_html(sources), "", ""

        except LLMAuthError:
            history[-1] = [message, "API Key 无效，请重新输入"]
            yield history, "", "", ""
        except LLMQuotaError:
            history[-1] = [message, "API 额度已用完，请充值或更换 Key"]
            yield history, "", "", ""
        except LLMTimeoutError:
            history[-1] = [message, "请求超时，请检查网络连接后重试"]
            yield history, "", "", ""
        except LLMRateLimitError:
            history[-1] = [message, "请求过于频繁，请稍后再试"]
            yield history, "", "", ""
        except (LLMError, RAGPipelineError) as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
            history[-1] = [message, f"错误: {e}"]
            yield history, "", "", ""
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            history[-1] = [message, f"未知错误: {e}"]
            yield history, "", "", ""

    # Wire up events
    msg_input.submit(
        user_submit,
        [msg_input, chatbot],
        [msg_input, chatbot],
        queue=False,
    ).then(
        bot_response,
        [chatbot] + sidebar_refs,
        [chatbot, sources_html, correction_html, agent_html],
    )

    submit_btn.click(
        user_submit,
        [msg_input, chatbot],
        [msg_input, chatbot],
        queue=False,
    ).then(
        bot_response,
        [chatbot] + sidebar_refs,
        [chatbot, sources_html, correction_html, agent_html],
    )

    clear_btn.click(
        lambda: ([], "", "", ""),
        None,
        [chatbot, sources_html, correction_html, agent_html],
    )
