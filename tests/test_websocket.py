"""WebSocket 流式查询测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen(items):
    for item in items:
        yield item


_STREAM_CHUNKS = [
    {"type": "sources", "sources": [{"content": "GDP 定义", "score": 0.95, "metadata": {"source": "test"}}]},
    {"type": "answer", "content": "GDP"},
    {"type": "answer", "content": "是"},
    {"type": "answer", "content": "国内生产总值"},
]


def _make_pipeline(chunks):
    pipeline = MagicMock()
    pipeline.astream_query = MagicMock(return_value=_async_gen(chunks))
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ip_connections():
    import src.api.routes.query as q

    q._ip_connections.clear()
    yield
    q._ip_connections.clear()


@pytest.fixture()
def pipeline_mock():
    pipeline = _make_pipeline(_STREAM_CHUNKS)
    with patch("src.api.routes.query.get_pipeline", return_value=pipeline):
        yield pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketChat:
    def test_normal_streaming(self, pipeline_mock):
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_json({"type": "query", "query": "什么是GDP？"})

            # sources frame
            msg = ws.receive_json()
            assert msg["type"] == "sources"
            assert len(msg["data"]) == 1

            # token frames
            tokens = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens.append(msg["content"])
                else:
                    break

            assert tokens == ["GDP", "是", "国内生产总值"]

            # metrics frame
            assert msg["type"] == "metrics"
            assert msg["data"]["tokens"] == 3
            assert msg["data"]["latency_ms"] > 0

            # done frame
            done = ws.receive_json()
            assert done["type"] == "done"

    def test_error_invalid_json(self, pipeline_mock):
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_text("not json")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "invalid JSON" in msg["message"]

    def test_error_unknown_type(self, pipeline_mock):
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_json({"type": "unknown"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown message type" in msg["message"]

    def test_multiple_queries_in_one_connection(self, pipeline_mock):
        chunks1 = [
            {"type": "sources", "sources": []},
            {"type": "answer", "content": "答案1"},
        ]
        chunks2 = [
            {"type": "sources", "sources": []},
            {"type": "answer", "content": "答案2"},
        ]

        pipeline_mock.astream_query = MagicMock(side_effect=[
            _async_gen(chunks1),
            _async_gen(chunks2),
        ])

        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            # First query
            ws.send_json({"type": "query", "query": "问题1"})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break

            # Second query
            ws.send_json({"type": "query", "query": "问题2"})
            msg = ws.receive_json()
            assert msg["type"] == "sources"
            msg = ws.receive_json()
            assert msg["type"] == "token"
            assert msg["content"] == "答案2"

    def test_disconnect_cleanup(self, pipeline_mock):
        import src.api.routes.query as q

        with TestClient(app) as client:
            assert len(q._ip_connections) == 0
            with client.websocket_connect("/api/v1/ws/chat"):
                pass
            assert len(q._ip_connections) == 0

    def test_query_with_options(self, pipeline_mock):
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_json({"type": "query", "query": "test", "options": {"top_k": 5}})

            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break

            pipeline_mock.astream_query.assert_called_once()
            call_kwargs = pipeline_mock.astream_query.call_args
            assert call_kwargs.kwargs.get("top_k") == 5 or ("top_k", 5) in call_kwargs.args

    def test_cancel_ignored_when_not_streaming(self, pipeline_mock):
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_json({"type": "cancel"})
            # cancel without active stream should not crash — just continue
            ws.send_json({"type": "query", "query": "test"})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "done":
                    break
