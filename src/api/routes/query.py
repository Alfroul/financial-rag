"""查询 API 路由 — 同步查询、SSE 流式查询、WebSocket 流式查询、健康检查、Metrics。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse

from src.api.deps import get_pipeline, get_store
from src.api.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentStep,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import Config
from src.correction.pipeline import SelfCorrectingPipeline
from src.correction.types import CorrectionResult
from src.metrics.collector import MetricsCollector
from src.rag_pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """同步查询：检索 + 重排序（可选）+ LLM 生成。"""
    t0 = time.perf_counter()
    try:
        effective_pipeline: RAGPipeline | SelfCorrectingPipeline = pipeline
        if body.self_correction:
            config = Config()
            effective_pipeline = SelfCorrectingPipeline(
                pipeline=pipeline,
                config=config.self_correction,
                api_key=config.siliconflow_api_key,
                base_url=config.self_correction.verifier_base_url,
                model=config.self_correction.verifier_model,
            )

        kwargs: dict = {}
        if body.top_k is not None:
            kwargs["top_k"] = body.top_k

        result = await effective_pipeline.aquery(
            question=body.question,
            chat_history=body.chat_history,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        correction_data = None
        raw_correction = result.get("correction")
        if raw_correction is not None and isinstance(raw_correction, CorrectionResult):
            c = raw_correction
            correction_data = {
                "passed": c.passed,
                "flagged_claims": c.flagged_claims,
                "confidence": c.confidence,
                "layer_results": {
                    k: str(v) for k, v in (c.layer_results or {}).items()
                },
            }

        return QueryResponse(
            answer=str(result.get("answer", "")),
            sources=result.get("sources", []) or [],  # type: ignore[arg-type]
            cached=False,
            latency_ms=round(latency_ms, 2),
            correction=correction_data,
        )
    except Exception as e:
        logger.error("查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {e}") from e


@router.post("/query/stream")
async def query_stream(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EventSourceResponse:
    """SSE 流式查询：先推送 sources，再逐 token 推送 answer。"""

    async def _generate() -> AsyncGenerator[dict, None]:
        kwargs: dict = {}
        if body.top_k is not None:
            kwargs["top_k"] = body.top_k

        try:
            async for chunk in pipeline.astream_query(
                question=body.question,
                chat_history=body.chat_history,
                **kwargs,
            ):
                if chunk.get("type") == "sources":
                    import json

                    yield {"event": "sources", "data": json.dumps(chunk["sources"], ensure_ascii=False)}
                elif chunk.get("type") == "answer":
                    yield {"event": "answer", "data": chunk.get("content", "")}
        except Exception as e:
            logger.error("流式查询失败: %s", e, exc_info=True)
            import json

            yield {"event": "error", "data": json.dumps({"detail": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(_generate())


@router.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(
    body: AgentQueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AgentQueryResponse:
    """Agent 模式查询：多步推理 + 工具调用。"""
    t0 = time.perf_counter()
    try:
        result = await pipeline.aagent_query(
            task=body.task,
            max_steps=body.max_steps,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Agent query completed in %.0fms, %d steps", latency_ms, len(result.get("steps", [])))

        steps = [AgentStep(**s) for s in result.get("steps", [])]
        return AgentQueryResponse(
            answer=result.get("answer", ""),
            steps=steps,
        )
    except Exception as e:
        logger.error("Agent 查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent 查询失败: {e}") from e


@router.get("/health", response_model=HealthResponse)
async def health(
    store: ChromaStore = Depends(get_store),
) -> HealthResponse:
    """健康检查：返回系统状态和已索引文档块数。"""
    stats = store.get_stats()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        indexed_chunks=stats.get("document_count", 0),
    )


@router.post("/eval", response_model=EvalResponse)
async def run_eval(
    body: EvalRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvalResponse:
    """运行 RAGAS 评估（需要 API Key 和充足额度）。"""
    try:
        from pathlib import Path

        from src.evaluation.ragas_eval import RAGEvaluator

        config = pipeline._llm  # noqa: SLF001 — 获取配置信息
        eval_path = body.eval_path or "data/eval/financial_qa_eval.json"
        eval_file = Path(eval_path)
        if not eval_file.exists():
            raise HTTPException(status_code=404, detail=f"评估数据集不存在: {eval_path}")

        import json

        with open(eval_file, encoding="utf-8") as f:
            eval_data = json.load(f)

        questions = [d["question"] for d in eval_data]
        # 简化：使用 pipeline 生成回答后评估
        responses = []
        contexts_list = []
        for q in questions:
            result = await pipeline.aquery(q)
            responses.append(result.get("answer", ""))
            sources = result.get("sources", [])
            contexts_list.append([s.get("content", "") for s in sources])

        references = [d.get("reference", "") for d in eval_data]

        t0 = time.perf_counter()
        evaluator = RAGEvaluator(
            api_key=config._api_key,  # noqa: SLF001
            model=config._model,  # noqa: SLF001
        )
        scores = evaluator.evaluate(
            questions=questions,
            responses=responses,
            contexts=contexts_list,
            references=references if any(references) else None,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return EvalResponse(
            scores=scores,
            sample_count=len(questions),
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("评估失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估失败: {e}") from e


@router.get("/metrics")
async def get_metrics() -> dict:
    """返回 MetricsCollector 的汇总统计数据。"""
    collector = MetricsCollector()
    return collector.summary()


@router.delete("/metrics")
async def clear_metrics() -> dict:
    """清空所有 Metrics 记录。"""
    collector = MetricsCollector()
    collector.clear()
    return {"status": "ok", "message": "Metrics cleared"}


# ---------------------------------------------------------------------------
# WebSocket 流式查询
# ---------------------------------------------------------------------------

_ip_connections: dict[str, int] = {}
_MAX_CONNECTIONS_PER_IP = 5
_IDLE_TIMEOUT = 60


def _check_and_add_ip(ip: str) -> bool:
    count = _ip_connections.get(ip, 0)
    if count >= _MAX_CONNECTIONS_PER_IP:
        return False
    _ip_connections[ip] = count + 1
    return True


def _remove_ip(ip: str) -> None:
    count = _ip_connections.get(ip, 0)
    if count <= 1:
        _ip_connections.pop(ip, None)
    else:
        _ip_connections[ip] = count - 1


async def _listen_for_cancel(websocket: WebSocket, cancel: asyncio.Event) -> None:
    try:
        while not cancel.is_set():
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                if data.get("type") == "cancel":
                    cancel.set()
                    return
            except json.JSONDecodeError:
                pass
    except (WebSocketDisconnect, RuntimeError):
        cancel.set()
    except Exception:
        cancel.set()


async def _ws_stream_response(
    websocket: WebSocket,
    pipeline: RAGPipeline,
    query_text: str,
    options: dict,
) -> None:
    cancel = asyncio.Event()
    listener = asyncio.create_task(_listen_for_cancel(websocket, cancel))

    t0 = time.perf_counter()
    token_count = 0

    try:
        kwargs: dict = {}
        if "top_k" in options:
            kwargs["top_k"] = options["top_k"]
        if "chat_history" in options:
            kwargs["chat_history"] = options["chat_history"]

        async for chunk in pipeline.astream_query(question=query_text, **kwargs):
            if cancel.is_set():
                break
            chunk_type = chunk.get("type")
            if chunk_type == "sources":
                await websocket.send_json({"type": "sources", "data": chunk["sources"]})
            elif chunk_type == "answer":
                token_count += 1
                await websocket.send_json({"type": "token", "content": chunk.get("content", "")})

        if not cancel.is_set():
            latency_ms = (time.perf_counter() - t0) * 1000
            await websocket.send_json({
                "type": "metrics",
                "data": {"latency_ms": round(latency_ms, 2), "tokens": token_count},
            })
            await websocket.send_json({"type": "done"})
    except Exception as e:
        logger.error("WebSocket streaming error: %s", e, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
        raise
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket 流式查询：逐 token 推送，支持取消和断连清理。"""
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"

    if not _check_and_add_ip(client_ip):
        await websocket.close(code=1008, reason="too many connections")
        return

    try:
        pipeline = get_pipeline()

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=_IDLE_TIMEOUT)
            except TimeoutError:
                await websocket.close(code=1000, reason="idle timeout")
                break

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "invalid JSON"})
                continue

            msg_type = data.get("type")
            if msg_type == "cancel":
                continue
            if msg_type != "query":
                await websocket.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})
                continue

            query_text = data.get("query", "")
            options = data.get("options") or {}
            await _ws_stream_response(websocket, pipeline, query_text, options)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_ip)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", client_ip, e, exc_info=True)
    finally:
        _remove_ip(client_ip)
