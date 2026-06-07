"""Langfuse Tracer — 封装 Langfuse SDK 的 trace/span 管理。

当 enabled=False 或 SDK 不可用时，所有方法为 no-op，零性能开销。
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None  # type: ignore[assignment, misc]


@dataclass
class _SpanRecord:
    trace_id: str
    span_id: str
    name: str
    langfuse_span: Any  # langfuse Span 对象，None 表示 no-op


@dataclass
class _TraceRecord:
    trace_id: str
    langfuse_trace: Any  # langfuse Trace 对象，None 表示 no-op


class LangfuseTracer:
    """封装 Langfuse SDK，为 RAG query 提供全链路 trace。"""

    def __init__(self, enabled: bool = False, public_key: str = "",
                 secret_key: str = "", host: str = "https://cloud.langfuse.com") -> None:
        self._enabled = enabled
        self._client: Any = None
        self._traces: dict[str, _TraceRecord] = {}
        self._spans: dict[str, _SpanRecord] = {}

        if not enabled:
            return

        # 优先从环境变量读取
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or public_key
        sk = os.environ.get("LANGFUSE_SECRET_KEY") or secret_key
        lf_host = os.environ.get("LANGFUSE_HOST") or host

        if not pk or not sk:
            logger.warning("Langfuse enabled but keys not configured; falling back to no-op")
            self._enabled = False
            return

        if Langfuse is None:
            logger.warning("langfuse package not installed; falling back to no-op")
            self._enabled = False
            return

        try:
            self._client = Langfuse(public_key=pk, secret_key=sk, host=lf_host)
            logger.info("Langfuse client initialized (host=%s)", lf_host)
        except Exception as e:
            logger.warning("Langfuse init failed: %s; falling back to no-op", e)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_trace(self, query: str, metadata: dict[str, Any] | None = None) -> str:
        """创建新 trace，返回 trace_id。"""
        trace_id = uuid.uuid4().hex
        if not self._enabled or self._client is None:
            self._traces[trace_id] = _TraceRecord(trace_id=trace_id, langfuse_trace=None)
            return trace_id

        try:
            langfuse_trace = self._client.trace(
                name="rag_query",
                input=query,
                metadata=metadata or {},
            )
            self._traces[trace_id] = _TraceRecord(trace_id=trace_id, langfuse_trace=langfuse_trace)
        except Exception as e:
            logger.warning("Langfuse start_trace failed: %s", e)
            self._traces[trace_id] = _TraceRecord(trace_id=trace_id, langfuse_trace=None)

        return trace_id

    def start_span(self, trace_id: str, name: str, input_data: Any = None) -> str:
        """创建子 span，返回 span_id。"""
        span_id = uuid.uuid4().hex
        if not self._enabled:
            self._spans[span_id] = _SpanRecord(trace_id=trace_id, span_id=span_id, name=name, langfuse_span=None)
            return span_id

        trace_record = self._traces.get(trace_id)
        if trace_record is None or trace_record.langfuse_trace is None:
            self._spans[span_id] = _SpanRecord(trace_id=trace_id, span_id=span_id, name=name, langfuse_span=None)
            return span_id

        try:
            langfuse_span = trace_record.langfuse_trace.span(
                name=name,
                input=input_data,
            )
            self._spans[span_id] = _SpanRecord(
                trace_id=trace_id, span_id=span_id,
                name=name, langfuse_span=langfuse_span,
            )
        except Exception as e:
            logger.warning("Langfuse start_span(%s) failed: %s", name, e)
            self._spans[span_id] = _SpanRecord(trace_id=trace_id, span_id=span_id, name=name, langfuse_span=None)

        return span_id

    def end_span(self, span_id: str, output_data: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """结束 span，记录结果。"""
        span = self._spans.get(span_id)
        if span is None or span.langfuse_span is None:
            return

        try:
            span.langfuse_span.end(output=output_data, metadata=metadata)
        except Exception as e:
            logger.warning("Langfuse end_span(%s) failed: %s", span.name, e)
        finally:
            self._spans.pop(span_id, None)

    def end_trace(self, trace_id: str, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """结束 trace。"""
        trace = self._traces.get(trace_id)
        if trace is None:
            return

        if trace.langfuse_trace is not None:
            try:
                trace.langfuse_trace.update(output=output, metadata=metadata)
            except Exception as e:
                logger.warning("Langfuse end_trace failed: %s", e)

        self._traces.pop(trace_id, None)

    def record_llm_call(
        self,
        span_id: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency: float = 0.0,
    ) -> None:
        """在 span 内记录 LLM 调用详情。"""
        span = self._spans.get(span_id)
        if span is None or span.langfuse_span is None:
            return

        try:
            span.langfuse_span.update(
                metadata={
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": round(latency, 2),
                },
            )
        except Exception as e:
            logger.warning("Langfuse record_llm_call failed: %s", e)

    def flush(self) -> None:
        """强制上传 trace 数据。"""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning("Langfuse flush failed: %s", e)
