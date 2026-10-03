"""Request/response models for the HTTP API."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.answer import Status
from app.retrieval.hybrid import Mode


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    stream: bool = False
    top_k: int = Field(8, ge=1, le=20)
    mode: Mode = "hybrid"
    use_rerank: bool = True
    use_graph: bool = True
    use_cache: bool = Field(True, description="Serve and store exact-match cached answers")


class SourceOut(BaseModel):
    label: str
    chunk_id: str
    doc_id: str
    company: str | None
    issue_date: str | None
    section: str
    url: str | None
    origin: Literal["search", "graph"]
    rerank_score: float | None
    snippet: str


class GraphOut(BaseModel):
    template: str
    params: dict
    rows: list[dict]
    error: str | None = None


class AskResponse(BaseModel):
    question: str
    status: Status
    answer: str
    citations: list[SourceOut]
    flags: list[str]
    route: str
    plan: dict
    graph: GraphOut | None
    sources: list[SourceOut]
    draft: str | None = Field(None, description="Model output when the answer was rejected")
    timings_ms: dict[str, int]
    usage: dict[str, int]
    cost_usd: float = Field(0.0, description="List-price cost of all LLM calls for this answer")
    trace_id: str | None = Field(None, description="Langfuse trace id, when tracing is enabled")
    cached: bool = Field(False, description="Served from the answer cache (no LLM calls)")


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    ordinal: int
    section: str
    observation_id: str | None
    text: str


class DocumentOut(BaseModel):
    id: str
    title: str
    company: str | None
    cms_number: str | None
    issue_date: str | None
    subject: str | None
    product: str | None
    issuing_office: str | None
    location: str | None
    url: str
    observations: list[dict]
    chunks: list[ChunkOut]


class IngestRequest(BaseModel):
    fetch: bool = Field(False, description="Download new letters from fda.gov before ingesting")
    limit: int = Field(10, ge=1, le=15, description="Target number of letters on disk (max 15)")
    refresh: bool = Field(False, description="Re-run LLM extraction instead of using the cache")


class IngestJob(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    request: IngestRequest
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
