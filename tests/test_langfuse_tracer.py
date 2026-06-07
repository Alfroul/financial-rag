"""LangfuseTracer 单元测试 — trace/span 生命周期、disabled 模式、Mock SDK。"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.observability.langfuse_tracer import LangfuseTracer


class TestDisabledMode:
    """enabled=False 时所有方法为 no-op。"""

    def test_disabled_tracer_enabled_property(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        assert not tracer.enabled

    def test_disabled_start_trace_returns_id(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        trace_id = tracer.start_trace("test query")
        assert isinstance(trace_id, str)
        assert len(trace_id) > 0

    def test_disabled_start_span_returns_id(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        trace_id = tracer.start_trace("q")
        span_id = tracer.start_span(trace_id, "retrieval")
        assert isinstance(span_id, str)

    def test_disabled_end_span_no_error(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        trace_id = tracer.start_trace("q")
        span_id = tracer.start_span(trace_id, "retrieval")
        tracer.end_span(span_id, output_data={"ms": 100})

    def test_disabled_end_trace_no_error(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        trace_id = tracer.start_trace("q")
        tracer.end_trace(trace_id, output="answer")

    def test_disabled_record_llm_call(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        trace_id = tracer.start_trace("q")
        span_id = tracer.start_span(trace_id, "gen")
        tracer.record_llm_call(span_id, model="MiMo", prompt_tokens=10, completion_tokens=20)

    def test_disabled_flush(self) -> None:
        tracer = LangfuseTracer(enabled=False)
        tracer.flush()


class TestMissingKeys:
    """enabled=True 但 keys 为空时回退到 no-op。"""

    def test_no_keys_falls_back_to_disabled(self) -> None:
        tracer = LangfuseTracer(enabled=True, public_key="", secret_key="", host="https://cloud.langfuse.com")
        assert not tracer.enabled

    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})
    def test_env_vars_used_when_config_empty(self) -> None:
        with patch("src.observability.langfuse_tracer.Langfuse") as mock_lf_cls:
            mock_client = MagicMock()
            mock_lf_cls.return_value = mock_client
            tracer = LangfuseTracer(enabled=True, public_key="", secret_key="")
            assert tracer.enabled
            mock_lf_cls.assert_called_once_with(
                public_key="pk", secret_key="sk", host="https://cloud.langfuse.com"
            )


class TestMockedSDK:
    """Mock Langfuse SDK，验证调用参数正确。"""

    def _make_tracer(self) -> tuple[LangfuseTracer, MagicMock, MagicMock]:
        with patch("src.observability.langfuse_tracer.Langfuse") as mock_lf_cls:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_span = MagicMock()
            mock_client.trace.return_value = mock_trace
            mock_trace.span.return_value = mock_span
            mock_lf_cls.return_value = mock_client
            tracer = LangfuseTracer(
                enabled=True,
                public_key="pk-test",
                secret_key="sk-test",
                host="https://cloud.langfuse.com",
            )
            return tracer, mock_trace, mock_span

    def test_start_trace_calls_sdk(self) -> None:
        tracer, mock_trace, _ = self._make_tracer()
        tracer._client.trace.reset_mock()
        tracer.start_trace("hello world", metadata={"user": "test"})
        tracer._client.trace.assert_called_once()
        call_kwargs = tracer._client.trace.call_args[1]
        assert call_kwargs["name"] == "rag_query"
        assert call_kwargs["input"] == "hello world"

    def test_start_span_creates_child(self) -> None:
        tracer, mock_trace, _ = self._make_tracer()
        trace_id = tracer.start_trace("q")
        mock_trace.span.reset_mock()
        tracer.start_span(trace_id, "retrieval", input_data={"query": "q"})
        mock_trace.span.assert_called_once()
        call_kwargs = mock_trace.span.call_args[1]
        assert call_kwargs["name"] == "retrieval"
        assert call_kwargs["input"] == {"query": "q"}

    def test_end_span_calls_end(self) -> None:
        tracer, _, mock_span = self._make_tracer()
        trace_id = tracer.start_trace("q")
        span_id = tracer.start_span(trace_id, "retrieval")
        mock_span.end.reset_mock()
        tracer.end_span(span_id, output_data={"num_results": 5}, metadata={"ms": 120})
        mock_span.end.assert_called_once()
        call_kwargs = mock_span.end.call_args[1]
        assert call_kwargs["output"] == {"num_results": 5}
        assert call_kwargs["metadata"] == {"ms": 120}

    def test_end_trace_calls_update(self) -> None:
        tracer, mock_trace, _ = self._make_tracer()
        trace_id = tracer.start_trace("q")
        mock_trace.update.reset_mock()
        tracer.end_trace(trace_id, output="answer", metadata={"len": 6})
        mock_trace.update.assert_called_once()

    def test_record_llm_call_updates_span(self) -> None:
        tracer, _, mock_span = self._make_tracer()
        trace_id = tracer.start_trace("q")
        span_id = tracer.start_span(trace_id, "generation")
        mock_span.update.reset_mock()
        tracer.record_llm_call(span_id, model="mimo-v2-pro", prompt_tokens=100, completion_tokens=50, latency=1.5)
        mock_span.update.assert_called_once()
        call_kwargs = mock_span.update.call_args[1]
        assert call_kwargs["metadata"]["model"] == "mimo-v2-pro"
        assert call_kwargs["metadata"]["prompt_tokens"] == 100
        assert call_kwargs["metadata"]["completion_tokens"] == 50

    def test_flush_calls_client(self) -> None:
        tracer, _, _ = self._make_tracer()
        tracer._client.flush.reset_mock()
        tracer.flush()
        tracer._client.flush.assert_called_once()

    def test_full_lifecycle(self) -> None:
        tracer, mock_trace, mock_span = self._make_tracer()
        mock_trace.span.return_value = mock_span

        trace_id = tracer.start_trace("test query")
        retrieval_span = tracer.start_span(trace_id, "retrieval", {"query": "test"})
        tracer.end_span(retrieval_span, output_data={"num_results": 3})
        gen_span = tracer.start_span(trace_id, "generation")
        tracer.record_llm_call(gen_span, model="MiMo", prompt_tokens=10, completion_tokens=20)
        tracer.end_span(gen_span, output_data={"answer_len": 100})
        tracer.end_trace(trace_id, output="final answer")

        # trace created once, 2 spans created
        assert tracer._client.trace.call_count == 1
        assert mock_trace.span.call_count == 2
        assert mock_span.end.call_count == 2
        assert mock_trace.update.call_count == 1


class TestSDKInitFailure:
    """SDK 初始化失败时优雅降级。"""

    def test_sdk_init_exception_falls_back(self) -> None:
        with patch("src.observability.langfuse_tracer.Langfuse", side_effect=Exception("connection error")):
            tracer = LangfuseTracer(enabled=True, public_key="pk", secret_key="sk")
            assert not tracer.enabled


class TestSDKNotInstalled:
    """langfuse 包未安装时回退到 no-op。"""

    def test_no_package_falls_back(self) -> None:
        with patch("src.observability.langfuse_tracer.Langfuse", None):
            tracer = LangfuseTracer(enabled=True, public_key="pk", secret_key="sk")
            assert not tracer.enabled
