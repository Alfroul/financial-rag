from __future__ import annotations

import logging

import streamlit as st

from src.correction.types import ClaimVerdict, CorrectionResult, RetrievalQuality
from src.generator.mimo_llm import (
    LLMAuthError,
    LLMError,
    LLMQuotaError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.rag_pipeline import RAGPipelineError
from src.ui.constants import _extract_keywords, _highlight_keywords
from src.ui.services import _build_rag_pipeline

logger = logging.getLogger(__name__)


def _render_correction_info(correction: CorrectionResult) -> None:
    passed = correction.passed
    confidence = correction.confidence
    flagged = correction.flagged_claims
    layer_results = correction.layer_results or {}

    status_icon = "✅" if passed else "⚠️"
    status_text = "通过" if passed else "有疑点"
    status_color = "#22C55E" if passed else "#EAB308"

    with st.expander(f"{status_icon} 自我修正报告（置信度: {confidence:.0%}）", expanded=not passed):
        st.markdown(
            f'<div style="color:{status_color};font-size:0.85rem;font-weight:600;">'
            f"修正结果: {status_text}"
            "</div>",
            unsafe_allow_html=True,
        )

        if flagged:
            st.markdown("**标记的断言:**")
            for claim in flagged:
                st.markdown(f"- 🚫 {claim}")

        if "retrieval_quality" in layer_results:
            rq: RetrievalQuality = layer_results["retrieval_quality"]  # type: ignore[assignment]
            st.markdown(
                f"**检索质量:** {rq.level} (top score: {rq.top_score:.2f}, "
                f"avg: {rq.avg_score:.2f}, sources: {rq.num_sources})"
            )

        if "retries" in layer_results:
            st.markdown(f"**重试次数:** {layer_results['retries']}")

        if "rule_issues" in layer_results:
            issues: list[dict] = layer_results["rule_issues"]  # type: ignore[assignment]
            if issues:
                st.markdown(f"**规则检查问题:** {len(issues)} 项")
                for issue in issues[:5]:
                    severity = issue.get("severity", "MEDIUM")
                    st.markdown(f"  - [{severity}] {issue.get('message', '')}")

        if "nli" in layer_results:
            verdicts: list[ClaimVerdict] = layer_results["nli"]  # type: ignore[assignment]
            unsupported_count = sum(1 for v in verdicts if not v.supported)
            st.markdown(f"**NLI 验证:** {len(verdicts)} 条断言, {unsupported_count} 条未通过")

        if "external" in layer_results:
            ext: list[ClaimVerdict] = layer_results["external"]  # type: ignore[assignment]
            ext_unsupported = sum(1 for v in ext if not v.supported)
            st.markdown(f"**外部验证:** {len(ext)} 条, {ext_unsupported} 条未通过")


def _get_messages() -> list[dict]:
    cid = st.session_state.current_conversation_id
    conversations: dict = st.session_state.conversations
    return conversations.setdefault(cid, [])


def render_chat_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        "<h1>Financial RAG — Query Terminal</h1>"
        '<div class="subtitle">Retrieval-Augmented Generation for Financial Intelligence</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    messages = _get_messages()

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("agent_steps"):
                with st.expander("Agent 思考过程", expanded=False):
                    for i, step in enumerate(msg["agent_steps"]):
                        thought = step.get("thought", "")
                        action = step.get("action", "")
                        observation = step.get("observation", "")
                        st.markdown(f"**Step {i + 1}**")
                        if thought:
                            st.markdown(f"```\n{thought}\n```")
                        if action:
                            st.markdown(f"**Action:** `{action}`")
                        if observation:
                            st.markdown(f"**Observation:** {observation}")
                        if i < len(msg["agent_steps"]) - 1:
                            st.divider()
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("References", expanded=False):
                    for src in msg["sources"]:
                        title = src.get("metadata", {}).get("title", src.get("metadata", {}).get("source", "未知来源"))
                        score = src.get("score", 0)
                        st.markdown(f"**📄 {title}** （相似度: {score:.2f}）")
                        content = src.get("content", "")
                        if msg.get("query"):
                            keywords = _extract_keywords(msg["query"])
                            content = _highlight_keywords(content, keywords)
                        st.markdown(content)
                        st.divider()
            if msg["role"] == "assistant" and msg.get("correction"):
                _render_correction_info(msg["correction"])

    if st.session_state.retrieval_results:
        with st.expander("Retrieval Details", expanded=False):
            query = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    query = m["content"]
                    break
            keywords = _extract_keywords(query) if query else []
            for i, result in enumerate(st.session_state.retrieval_results):
                meta = result.get("metadata", {})
                title = meta.get("title", meta.get("source", f"文档 {i + 1}"))
                score = result.get("score", 0)
                st.markdown(f"**[{i + 1}] 📄 {title}** — 相似度: `{score:.4f}`")
                content = result.get("content", "")
                if keywords:
                    content = _highlight_keywords(content, keywords)
                st.markdown(content)
                st.divider()

    if prompt := st.chat_input("请输入您的问题..."):
        api_key = st.session_state.api_key
        if not api_key:
            st.error("请先在侧边栏输入 API Key")
            return

        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Retrieving relevant documents...")
            try:
                pipeline = _build_rag_pipeline(api_key=api_key)
                use_correction = getattr(st.session_state, "self_correction_enabled", False)
                is_agent_mode = getattr(st.session_state, "query_mode", "qa") == "agent"

                sources: list[dict] = []
                answer = ""
                correction_data: CorrectionResult | None = None

                if is_agent_mode:
                    status.update(label="Agent reasoning...")
                    result = pipeline.agent_query(task=prompt)  # type: ignore[union-attr]
                    answer = str(result.get("answer", ""))
                    agent_steps = result.get("steps", [])
                    st.markdown(answer)
                    if agent_steps:
                        with st.expander("Agent 思考过程", expanded=False):
                            for i, step in enumerate(agent_steps):
                                thought = step.get("thought", "")
                                action = step.get("action", "")
                                observation = step.get("observation", "")
                                st.markdown(f"**Step {i + 1}**")
                                if thought:
                                    st.markdown(f"```\n{thought}\n```")
                                if action:
                                    st.markdown(f"**Action:** `{action}`")
                                if observation:
                                    st.markdown(f"**Observation:** {observation}")
                                if i < len(agent_steps) - 1:
                                    st.divider()
                    status.update(label="Agent complete", state="complete")

                elif use_correction:
                    status.update(label="Generating and verifying response...")
                    result = pipeline.query(
                        question=prompt,
                        chat_history=[m for m in messages[:-1]],
                    )
                    answer = str(result.get("answer", ""))
                    sources = result.get("sources", [])  # type: ignore[assignment]
                    st.session_state.retrieval_results = sources
                    if "correction" in result and result["correction"] is not None:
                        correction_data = result["correction"]  # type: ignore[assignment]
                    st.markdown(answer)
                    status.update(label="Response complete", state="complete")
                else:
                    answer_parts = []
                    status.update(label="Generating response...")
                    answer_placeholder = st.empty()

                    for chunk in pipeline.stream_query(
                        question=prompt,
                        chat_history=[m for m in messages[:-1]],
                    ):
                        if chunk["type"] == "sources":
                            sources = chunk["sources"]
                            st.session_state.retrieval_results = sources
                        elif chunk["type"] == "answer":
                            answer_parts.append(chunk["content"])
                            answer_placeholder.markdown("".join(answer_parts) + "▌")
                    answer = "".join(answer_parts)
                    answer_placeholder.markdown(answer)
                    status.update(label="Response complete", state="complete")

                if correction_data:
                    _render_correction_info(correction_data)

                msg_data: dict[str, object] = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "query": prompt,
                }
                if correction_data:
                    msg_data["correction"] = correction_data
                if is_agent_mode and agent_steps:
                    msg_data["agent_steps"] = agent_steps
                messages.append(msg_data)

            except LLMAuthError as e:
                logger.error("LLM Auth 错误: %s", e, exc_info=True)
                st.error("API Key 无效，请在侧边栏重新输入")
                status.update(label="Auth failed", state="error")
                msg = {"role": "assistant", "content": "抱歉，API Key 无效，请在侧边栏重新输入。", "sources": []}
                messages.append(msg)
            except LLMQuotaError as e:
                logger.error("LLM 额度耗尽: %s", e, exc_info=True)
                st.error("API 额度已用完，请到 SiliconFlow 控制台检查用量或更换 Key")
                status.update(label="Quota exceeded", state="error")
                msg = {"role": "assistant", "content": "抱歉，API 额度已用完，请充值或更换 Key。", "sources": []}
                messages.append(msg)
            except LLMTimeoutError as e:
                logger.error("LLM 超时: %s", e, exc_info=True)
                st.error("请求超时，请检查网络连接后重试")
                status.update(label="Timeout", state="error")
                msg = {"role": "assistant", "content": "抱歉，请求超时，请检查网络连接后重试。", "sources": []}
                messages.append(msg)
            except LLMRateLimitError as e:
                logger.error("LLM 频率限制: %s", e, exc_info=True)
                st.error("请求过于频繁，请等待几秒后重试")
                status.update(label="Rate limited", state="error")
                messages.append({"role": "assistant", "content": "抱歉，请求过于频繁，请稍后再试。", "sources": []})
            except LLMError as e:
                logger.error("LLM 错误: %s", e, exc_info=True)
                st.error(f"LLM 错误: {e}")
                status.update(label="LLM error", state="error")
                messages.append({"role": "assistant", "content": f"抱歉，LLM 错误: {e}", "sources": []})
            except RAGPipelineError as e:
                logger.error("RAG 流程错误: %s", e, exc_info=True)
                st.error(f"RAG 流程错误: {e}")
                status.update(label="Pipeline error", state="error")
                messages.append({"role": "assistant", "content": f"抱歉，RAG 流程错误: {e}", "sources": []})
            except Exception as e:
                logger.error("未知错误: %s", e, exc_info=True)
                st.error("发生未知错误，请刷新页面重试")
                status.update(label="Unknown error", state="error")
                messages.append({"role": "assistant", "content": "抱歉，发生未知错误，请刷新页面重试。", "sources": []})
