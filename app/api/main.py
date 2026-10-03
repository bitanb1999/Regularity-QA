"""HTTP API.

    uv run uvicorn app.api.main:app --reload
"""
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app import observability
from app.db import neo4j, pg_pool
from app.graph.queries import document_observations
from app.ingest import fda_scraper, pipeline
from app.llm.client import QuotaExhausted
from app.models.api import AskRequest, AskResponse, ChunkOut, DocumentOut, IngestJob, IngestRequest
from app.qa import ask, ask_stream
from app.retrieval.embeddings import model as embedding_model
from app.retrieval.hybrid import _reranker

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Load models up front; otherwise the first /ask pays ~30s of cold start.
    embedding_model()
    _reranker()
    pg_pool()
    yield
    observability.flush()  # export any buffered traces before exit
    pg_pool().close()
    neo4j().close()


app = FastAPI(title="Regulatory Q&A", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    checks: dict[str, str] = {}
    try:
        with pg_pool().connection() as conn:
            docs, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)").fetchone()
        checks["postgres"] = f"ok ({docs} documents, {chunks} chunks)"
    except Exception as e:  # noqa: BLE001 - health reports any failure instead of raising
        checks["postgres"] = f"error: {type(e).__name__}"
    try:
        neo4j().verify_connectivity()
        checks["neo4j"] = "ok"
    except Exception as e:  # noqa: BLE001 - health reports any failure instead of raising
        checks["neo4j"] = f"error: {type(e).__name__}"
    ok = all(v.startswith("ok") for v in checks.values())
    if not ok:
        raise HTTPException(503, detail={"status": "degraded", "checks": checks})
    return {"status": "ok", "checks": checks}


@app.post("/ask", response_model=AskResponse,
          responses={200: {"content": {"text/event-stream": {}}, "description": "JSON, or SSE when stream=true"}})
def ask_endpoint(req: AskRequest):
    if req.stream:
        return StreamingResponse(ask_stream(req), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        return ask(req)
    except QuotaExhausted as e:
        raise HTTPException(503, detail={"code": "quota_exhausted", "message": str(e),
                                         "retry_in": e.retry_in}) from e


@app.get("/documents")
def list_documents():
    with pg_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title, company, issue_date, subject FROM documents ORDER BY issue_date DESC").fetchall()
    return [{"id": r[0], "title": r[1], "company": r[2], "issue_date": str(r[3]), "subject": r[4]}
            for r in rows]


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str):
    with pg_pool().connection() as conn:
        d = conn.execute(
            """SELECT id, title, company, cms_number, issue_date, subject, product, issuing_office,
                      concat_ws(', ', street, city, region, postal_code, country), url
               FROM documents WHERE id = %s""", (doc_id,)).fetchone()
        if not d:
            raise HTTPException(404, f"document {doc_id!r} not found")
        chunks = conn.execute(
            "SELECT id, doc_id, ordinal, section, observation_id, text FROM chunks WHERE doc_id = %s ORDER BY ordinal",
            (doc_id,)).fetchall()
    return DocumentOut(
        id=d[0], title=d[1], company=d[2], cms_number=d[3], issue_date=str(d[4]) if d[4] else None,
        subject=d[5], product=d[6], issuing_office=d[7], location=d[8] or None, url=d[9],
        observations=document_observations(neo4j(), doc_id),
        chunks=[ChunkOut(id=c[0], doc_id=c[1], ordinal=c[2], section=c[3], observation_id=c[4], text=c[5])
                for c in chunks],
    )


@app.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str):
    with pg_pool().connection() as conn:
        c = conn.execute(
            "SELECT id, doc_id, ordinal, section, observation_id, text FROM chunks WHERE id = %s",
            (chunk_id,)).fetchone()
    if not c:
        raise HTTPException(404, f"chunk {chunk_id!r} not found")
    return ChunkOut(id=c[0], doc_id=c[1], ordinal=c[2], section=c[3], observation_id=c[4], text=c[5])


# --- ingestion: one background job at a time, status kept in memory ---
_jobs: dict[str, IngestJob] = {}
_ingest_lock = threading.Lock()


def _run_ingest(job: IngestJob) -> None:
    job.status, job.started_at = "running", datetime.now(UTC)
    try:
        if job.request.fetch:
            try:
                fda_scraper.main(job.request.limit)
            except SystemExit as e:  # the scraper exits on robots.txt / bot-protection stops
                if e.code not in (None, 0):
                    raise RuntimeError(str(e.code)) from None
        pipeline.run(refresh=job.request.refresh)
        job.status = "succeeded"
    except Exception as e:
        log.exception("ingest job %s failed", job.id)
        job.status, job.error = "failed", f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = datetime.now(UTC)
        _ingest_lock.release()


@app.post("/ingest", response_model=IngestJob, status_code=202)
def start_ingest(req: IngestRequest, background: BackgroundTasks):
    if not _ingest_lock.acquire(blocking=False):
        running = next((j for j in _jobs.values() if j.status in ("queued", "running")), None)
        raise HTTPException(409, f"ingest job {running.id if running else '?'} is already running")
    job = IngestJob(id=uuid.uuid4().hex[:12], status="queued", request=req)
    _jobs[job.id] = job
    background.add_task(_run_ingest, job)
    return job


@app.get("/ingest/{job_id}", response_model=IngestJob)
def get_ingest(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"job {job_id!r} not found")
    return _jobs[job_id]
