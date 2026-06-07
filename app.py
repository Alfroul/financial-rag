from __future__ import annotations

from uuid import uuid4

import streamlit as st

from src.config import Config
from src.ui.styles import _CUSTOM_CSS
from src.ui.sidebar import render_sidebar
from src.ui.chat_tab import render_chat_tab
from src.ui.doc_tab import render_doc_management_tab
from src.ui.eval_tab import render_eval_tab

config = Config()

st.set_page_config(page_title=config.app.page_title, page_icon="💰", layout="wide")

st.markdown(f"<style>{_CUSTOM_CSS}</style>", unsafe_allow_html=True)

if "current_conversation_id" not in st.session_state:
    _first_id = uuid4().hex[:8]
    st.session_state.current_conversation_id = _first_id
    st.session_state.conversations = {_first_id: []}

_session_defaults = {
    "api_key": "",
    "conversations": {},
    "retrieval_results": [],
    "indexed_files": [],
    "model": config.llm.model,
    "temperature": config.llm.temperature,
    "max_tokens": config.llm.max_tokens,
    "top_k": config.retriever.top_k,
    "score_threshold": config.retriever.score_threshold,
    "chunk_size": config.chunker.chunk_size,
    "chunk_overlap": config.chunker.chunk_overlap,
    "retrieve_strategy": config.hybrid.strategy,
    "reranker_enabled": config.reranker.enabled,
    "reranker_top_n": config.reranker.top_n,
    "query_rewrite": config.rag.query_rewrite,
    "cache_enabled": config.cache.enabled,
    "self_correction_enabled": config.self_correction.enabled,
    "query_mode": "qa",
}
for _key, _default in _session_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

render_sidebar()

tab_chat, tab_docs, tab_eval = st.tabs(["智能问答", "文档管理", "系统评估"])

with tab_chat:
    render_chat_tab()

with tab_docs:
    render_doc_management_tab()

with tab_eval:
    render_eval_tab()
