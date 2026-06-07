"""Pydantic 数据模型 — API 请求/响应 Schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 查询相关
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """查询请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    chat_history: list[dict[str, str]] | None = Field(default=None, description="历史对话")
    top_k: int | None = Field(default=None, ge=1, le=50, description="检索返回数量")
    strategy: str | None = Field(default=None, description="检索策略: vector / bm25 / hybrid")
    use_reranker: bool | None = Field(default=None, description="是否启用重排序")
    self_correction: bool = Field(default=False, description="是否启用自我修正")


class SourceItem(BaseModel):
    """单条检索来源。"""

    content: str
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """查询响应。"""

    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0
    correction: dict | None = Field(default=None, description="自我修正报告（启用时返回）")


# ---------------------------------------------------------------------------
# 文档管理相关
# ---------------------------------------------------------------------------


class DocumentUploadRequest(BaseModel):
    """文档上传请求。"""

    filename: str = Field(..., min_length=1, description="文件名")
    content: str = Field(..., min_length=1, description="文件内容（base64 编码）")
    doc_type: str = Field(default="text", description="文档类型: text / pdf / qa")


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""

    success: bool
    message: str
    chunk_count: int = 0


class DocumentStatsResponse(BaseModel):
    """文档统计响应。"""

    total_chunks: int = 0
    unique_sources: int = 0
    source_distribution: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
    version: str = "1.0.0"
    indexed_chunks: int = 0


# ---------------------------------------------------------------------------
# 评估相关
# ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    """评估请求。"""

    eval_path: str | None = Field(default=None, description="评估数据集路径")
    strategy: str = Field(default="hybrid", description="检索策略")
    use_reranker: bool = Field(default=False, description="是否启用重排序")


class EvalResponse(BaseModel):
    """评估响应。"""

    scores: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0
    latency_ms: float = 0.0


class MetricsSummaryResponse(BaseModel):
    """Metrics 汇总响应。"""

    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    strategy_distribution: dict[str, int] = Field(default_factory=dict)
    recent_queries: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 相关
# ---------------------------------------------------------------------------


class AgentQueryRequest(BaseModel):
    """Agent 查询请求。"""

    task: str = Field(..., min_length=1, max_length=2000, description="分析任务描述")
    max_steps: int = Field(default=6, ge=1, le=20, description="最大推理步数")


class AgentStep(BaseModel):
    """Agent 单步推理记录。"""

    thought: str = ""
    action: str = ""
    observation: str = ""


class AgentQueryResponse(BaseModel):
    """Agent 查询响应。"""

    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
