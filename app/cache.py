"""Exact-match answer cache in Postgres.

The key covers everything that can change an answer: the normalized question, the request's
retrieval settings, the models and prompts, and a data version derived from the ingested letters.
Re-ingesting or changing a prompt therefore invalidates old entries without manual flushing.

Deliberately not a semantic (embedding-similarity) cache: "What did FDA find at Bentley?" and
"What did FDA find at Babikian?" embed almost identically, and a near-duplicate hit would return
another company's violations, which is worse than a miss in a compliance tool.
"""
import hashlib
import json
import logging
import re
from functools import lru_cache

from app.config import settings
from app.db import pg_pool
from app.models.api import AskRequest, AskResponse

log = logging.getLogger(__name__)

CACHEABLE = {"answered", "flagged", "refused"}  # never cache errors or rejected drafts

SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    key         TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    response    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    hits        INT NOT NULL DEFAULT 0,
    last_hit_at TIMESTAMPTZ
);
"""


def normalize(question: str) -> str:
    q = re.sub(r"\s+", " ", question).strip().lower()
    return q.rstrip("?.! ")


@lru_cache
def _ensure_schema() -> None:
    with pg_pool().connection() as conn:
        conn.execute(SCHEMA)


def _config_version() -> str:
    from app.llm import answer, planner
    parts = [settings.answer_model, settings.router_model, answer.SYSTEM, planner.SYSTEM,
             settings.embedding_model, settings.rerank_model]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def data_version() -> str:
    """Changes whenever letters are (re)ingested."""
    with pg_pool().connection() as conn:
        n, latest, chunks = conn.execute(
            "SELECT count(*), max(stored_at), (SELECT count(*) FROM chunks) FROM documents").fetchone()
    return f"{n}:{chunks}:{latest.isoformat() if latest else '-'}"


def key(req: AskRequest) -> str:
    material = json.dumps({
        "q": normalize(req.question), "mode": req.mode, "top_k": req.top_k,
        "rerank": req.use_rerank, "graph": req.use_graph,
        "config": _config_version(), "data": data_version(),
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def get(req: AskRequest) -> AskResponse | None:
    try:
        _ensure_schema()
        k = key(req)
        with pg_pool().connection() as conn:
            row = conn.execute(
                "UPDATE answer_cache SET hits = hits + 1, last_hit_at = now() WHERE key = %s RETURNING response",
                (k,)).fetchone()
        return AskResponse.model_validate(row[0]) if row else None
    except Exception as e:  # noqa: BLE001 - a cache failure must degrade to a miss, not an error
        log.warning("answer cache read failed: %s", e)
        return None


def put(req: AskRequest, resp: AskResponse) -> None:
    if resp.status not in CACHEABLE:
        return
    try:
        _ensure_schema()
        payload = resp.model_dump(mode="json", exclude={"cached", "trace_id"})
        with pg_pool().connection() as conn:
            conn.execute(
                """INSERT INTO answer_cache (key, question, response) VALUES (%s, %s, %s)
                   ON CONFLICT (key) DO UPDATE SET response = EXCLUDED.response, created_at = now()""",
                (key(req), req.question, json.dumps(payload)))
    except Exception as e:  # noqa: BLE001
        log.warning("answer cache write failed: %s", e)


def delete(req: AskRequest) -> None:
    _ensure_schema()
    with pg_pool().connection() as conn:
        conn.execute("DELETE FROM answer_cache WHERE key = %s", (key(req),))


def clear() -> int:
    _ensure_schema()
    with pg_pool().connection() as conn:
        return conn.execute("DELETE FROM answer_cache").rowcount


def stats() -> dict:
    _ensure_schema()
    with pg_pool().connection() as conn:
        n, hits = conn.execute("SELECT count(*), coalesce(sum(hits), 0) FROM answer_cache").fetchone()
    return {"entries": n, "hits": hits}
