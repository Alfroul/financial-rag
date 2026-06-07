from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

import streamlit as st

from src.config import Config
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.fact_cache.store import FactCacheStore
from src.fact_cache.sync import CacheSynchronizer, DataSourceWatcher
from src.fact_extractor.extractor import FactExtractor
from src.metrics.collector import MetricsCollector
from src.ui.services import _init_vectorstore

logger = logging.getLogger(__name__)

config = Config()


def _render_sync_section() -> None:
    """渲染缓存同步区域。"""
    if "last_sync_time" not in st.session_state:
        st.session_state.last_sync_time = None
    if "last_sync_result" not in st.session_state:
        st.session_state.last_sync_result = None

    # 显示上次同步信息
    if st.session_state.last_sync_time:
        st.caption(f"上次同步: {st.session_state.last_sync_time}")
    if st.session_state.last_sync_result:
        r = st.session_state.last_sync_result
        st.caption(
            f"变更: +{r.added} ~{r.modified} -{r.deleted} | "
            f"更新 {r.facts_updated} 条fact"
        )

    if st.button("同步缓存", use_container_width=True, key="sync_cache"):
        with st.spinner("正在同步..."):
            try:
                api_key = st.session_state.api_key or ""
                if not api_key:
                    st.warning("请先配置 API Key")
                    return

                embedder = SiliconFlowEmbedder(api_key=api_key, model=config.embedding.model)
                fact_cache = FactCacheStore(
                    embedder=embedder,
                    collection_name=config.fact_cache.collection_name,
                )
                fact_extractor = FactExtractor(api_key=api_key)

                watcher = DataSourceWatcher(
                    data_directory=config.sync.data_directory,
                    hash_file=config.sync.hash_file,
                )
                synchronizer = CacheSynchronizer(
                    cache=fact_cache,
                    extractor=fact_extractor,
                    watcher=watcher,
                    data_directory=config.sync.data_directory,
                    judge_api_key=api_key,
                    judge_model=config.sync.judge_model,
                    judge_base_url=config.sync.judge_base_url,
                )
                result = synchronizer.manual_sync()
                st.session_state.last_sync_result = result
                st.session_state.last_sync_time = datetime.now().strftime("%H:%M:%S")
                st.success(f"同步完成: {result.facts_updated} 条fact已更新")
            except Exception as e:
                logger.error("Sync failed: %s", e, exc_info=True)
                st.error(f"同步失败: {e}")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="margin-bottom:0.6rem;">'
            '<div style="font-size:0.85rem;color:#C9A84C;font-weight:700;'
            'text-transform:uppercase;letter-spacing:3px;">Financial RAG</div>'
            '<div style="font-size:0.6rem;color:#4B5563;letter-spacing:1px;'
            'text-transform:uppercase;margin-top:2px;">Intelligent Query System</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.5rem 0;">', unsafe_allow_html=True)

        st.subheader("CONVERSATIONS")

        if st.button("NEW SESSION", use_container_width=True, key="new_conv"):
            new_id = uuid4().hex[:8]
            st.session_state.current_conversation_id = new_id
            st.session_state.conversations[new_id] = []
            st.session_state.retrieval_results = []
            st.rerun()

        conv_list = list(st.session_state.conversations.keys())
        if conv_list:
            current_id = st.session_state.current_conversation_id
            display_names = []
            for cid in conv_list:
                msgs = st.session_state.conversations.get(cid, [])
                if msgs:
                    first_user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "EMPTY")
                    display_names.append(first_user_msg[:20] + ("..." if len(first_user_msg) > 20 else ""))
                else:
                    display_names.append("— New Session —")

            selected_idx = conv_list.index(current_id) if current_id in conv_list else 0
            selected = st.radio(
                "HISTORY",
                range(len(conv_list)),
                format_func=lambda i: display_names[i],
                index=selected_idx,
                label_visibility="collapsed",
            )
            if conv_list[selected] != current_id:
                st.session_state.current_conversation_id = conv_list[selected]
                st.session_state.retrieval_results = []
                st.rerun()

            if st.button("CLEAR SESSION", use_container_width=True, key="clear_conv"):
                st.session_state.conversations[current_id] = []
                st.session_state.retrieval_results = []
                st.rerun()

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("API CONFIGURATION")
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            key="api_key_input",
        )
        st.session_state.api_key = api_key

        if api_key:
            st.markdown(
                '<div style="color:#22C55E;font-size:0.78rem;letter-spacing:0.5px;padding:0.3rem 0;">● CONNECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#EAB308;font-size:0.78rem;letter-spacing:0.5px;padding:0.3rem 0;">'
                "● AWAITING KEY</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("MODEL PARAMETERS")
        model = st.selectbox(
            "Model",
            ["mimo-v2-pro", "mimo-v2.5", "mimo-v2.5-Pro"],
            index=["mimo-v2-pro", "mimo-v2.5", "mimo-v2.5-Pro"].index(config.llm.model)
            if config.llm.model in ("mimo-v2-pro", "mimo-v2.5", "mimo-v2.5-Pro")
            else 0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, config.llm.temperature, 0.1)
        max_tokens = st.slider("Max Tokens", 256, 4096, config.llm.max_tokens, 256)

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("QUERY MODE")
        query_mode = st.radio(
            "Query Mode",
            ["问答模式", "分析模式"],
            index=0,
            help="问答模式：单次检索回答。分析模式：Agent 多步推理，支持工具调用。",
        )
        st.session_state.query_mode = "agent" if query_mode == "分析模式" else "qa"

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("RETRIEVAL")
        retrieve_strategy = st.radio(
            "Retrieval Mode",
            ["混合检索", "向量检索", "BM25 检索"],
            index=["hybrid", "vector", "bm25"].index(st.session_state.retrieve_strategy)
            if st.session_state.retrieve_strategy in ("hybrid", "vector", "bm25")
            else 0,
        )
        strategy_map = {"混合检索": "hybrid", "向量检索": "vector", "BM25 检索": "bm25"}
        top_k = st.slider("Top-K", 1, 20, config.retriever.top_k, 1)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, config.retriever.score_threshold, 0.05)

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("RERANKER")
        reranker_enabled = st.checkbox("Enable Reranker", value=st.session_state.reranker_enabled)
        reranker_top_n = st.slider("Rerank Top-N", 1, 20, config.reranker.top_n, 1)

        query_rewrite_enabled = st.checkbox(
            "Enable Query Rewriting",
            value=st.session_state.query_rewrite,
            help="多轮对话时自动改写指代性问题为独立检索 query（如「那它的市盈率呢？」→「沪深300的市盈率是多少？」）",
        )

        cache_enabled = st.checkbox(
            "Enable Query Cache",
            value=st.session_state.cache_enabled,
            help="启用语义缓存，相似问题直接返回缓存结果，减少 API 调用",
        )

        self_correction_enabled = st.checkbox(
            "自我修正",
            value=st.session_state.self_correction_enabled,
            help="启用多层幻觉检测与修正：检索质量门控 → 规则预检 → NLI 验证 → 外部模型二判定",
        )
        if self_correction_enabled:
            st.markdown(
                '<div style="color:#C9A84C;font-size:0.72rem;padding:0.2rem 0;">'
                "四层检测已启用：L0 门控 · L2 规则 · L3 NLI · L4 外部"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("CHUNKING")
        chunk_size = st.slider("Chunk Size", 128, 2048, config.chunker.chunk_size, 64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 512, config.chunker.chunk_overlap, 32)
        if chunk_overlap >= chunk_size:
            st.warning("Chunk Overlap 应小于 Chunk Size，否则分块将无意义。请调小 Overlap 或增大 Size。")

        st.session_state.model = model
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens
        st.session_state.top_k = top_k
        st.session_state.score_threshold = score_threshold
        st.session_state.retrieve_strategy = strategy_map[retrieve_strategy]
        st.session_state.reranker_enabled = reranker_enabled
        st.session_state.reranker_top_n = reranker_top_n
        st.session_state.query_rewrite = query_rewrite_enabled
        st.session_state.cache_enabled = cache_enabled
        st.session_state.self_correction_enabled = self_correction_enabled
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("CACHE SYNC")
        if config.fact_cache.enabled:
            _render_sync_section()
        else:
            st.caption("Fact Cache disabled — sync unavailable")

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("SYSTEM METRICS")
        try:
            _collector = MetricsCollector()
            _summary = _collector.summary()
            if _summary["total_queries"] > 0:
                total_tokens = _summary["total_input_tokens"] + _summary["total_output_tokens"]
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#9CA3AF;">'
                    f'<div>Queries: <b style="color:#C9A84C;">{_summary["total_queries"]}</b></div>'
                    f'<div>Avg Latency: <b style="color:#C9A84C;">{_summary["avg_latency_ms"]:.0f}ms</b></div>'
                    f'<div>P95 Latency: <b style="color:#C9A84C;">{_summary["p95_latency_ms"]:.0f}ms</b></div>'
                    f'<div>Cache Hit: <b style="color:#C9A84C;">{_summary["cache_hit_rate"] * 100:.1f}%</b></div>'
                    f'<div>Tokens: <b style="color:#C9A84C;">{total_tokens}</b></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No queries yet")
        except Exception as e:
            logger.debug("Metrics display failed: %s", e)

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:0.6rem 0;">', unsafe_allow_html=True)
        st.subheader("KNOWLEDGE BASE")
        try:
            store = _init_vectorstore()
            stats = store.get_stats()
            st.markdown(
                f'<div class="sidebar-kb-stat">'
                f'<div class="stat-label">Indexed Chunks</div>'
                f'<div class="stat-value">{stats["document_count"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            logger.error("Vector store connection failed in sidebar: %s", e, exc_info=True)
            st.markdown(
                '<div style="color:#EF4444;font-size:0.78rem;">● CONNECTION FAILED</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="branding-footer">'
            '<div class="brand-name">Financial RAG</div>'
            '<div class="brand-version">v1.0 // Powered by SiliconFlow</div>'
            "</div>",
            unsafe_allow_html=True,
        )
