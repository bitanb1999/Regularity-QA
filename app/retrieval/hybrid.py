"""Hybrid retrieval over `chunks`: pgvector + Postgres full-text, fused with RRF, then reranked.

`mode` exists so the eval harness can compare vector-only / keyword-only / hybrid.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np
from sentence_transformers import CrossEncoder

from app.config import settings
from app.db import pg_pool
from app.retrieval.embeddings import embed_query

Mode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60
CANDIDATES = 30

_COLUMNS = """c.id, c.doc_id, d.company, d.issue_date, d.url, c.section, c.observation_id, c.text"""
_FILTER = "(%(doc_ids)s::text[] IS NULL OR c.doc_id = ANY(%(doc_ids)s))"

VECTOR_SQL = f"""
SELECT {_COLUMNS}, 1 - (c.embedding <=> %(vec)s) AS score
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE {_FILTER}
ORDER BY c.embedding <=> %(vec)s
LIMIT %(k)s
"""

# Question lexemes are OR'd (plainto/websearch would AND them, which kills recall for
# natural-language questions). 'simple' re-parses the already-normalized lexemes as-is, so
# exact tokens like '211.192' survive.
KEYWORD_SQL = f"""
WITH q AS (
    SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS q
    FROM unnest(to_tsvector('english', %(text)s))
)
SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q.q) AS score
FROM chunks c JOIN documents d ON d.id = c.doc_id, q
WHERE q.q IS NOT NULL AND c.tsv @@ q.q AND {_FILTER}
ORDER BY score DESC
LIMIT %(k)s
"""


CITATION_WEIGHT = 3.0  # an exact section match is the most precise signal we have
# Exact CFR section lookup ("211.192", "1.502"). In OR'd keyword search the common tokens
# "21"/"cfr" outrank the one rare token that matters, so sections get their own ranked list.
SECTION = re.compile(r"\b\d{1,3}\.\d{1,4}\b")
CITATION_SQL = f"""
SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q) AS score
FROM chunks c JOIN documents d ON d.id = c.doc_id,
     to_tsquery('simple', %(tsq)s) q
WHERE c.tsv @@ q AND {_FILTER}
ORDER BY score DESC
LIMIT %(k)s
"""


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    company: str | None
    issue_date: str | None
    url: str | None
    section: str
    observation_id: str | None
    text: str
    scores: dict[str, float] = field(default_factory=dict)


def _rows(sql: str, params: dict, key: str) -> list[Hit]:
    with pg_pool().connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        Hit(r[0], r[1], r[2], str(r[3]) if r[3] else None, r[4], r[5], r[6], r[7], {key: float(r[8])})
        for r in rows
    ]


def vector_search(query: str, k: int = CANDIDATES, doc_ids: list[str] | None = None) -> list[Hit]:
    vec = np.asarray(embed_query(query), dtype=np.float32)
    return _rows(VECTOR_SQL, {"vec": vec, "k": k, "doc_ids": doc_ids}, "vector")


def keyword_search(query: str, k: int = CANDIDATES, doc_ids: list[str] | None = None) -> list[Hit]:
    return _rows(KEYWORD_SQL, {"text": query, "k": k, "doc_ids": doc_ids}, "keyword")


def citation_search(query: str, k: int = CANDIDATES, doc_ids: list[str] | None = None) -> list[Hit]:
    sections = sorted(set(SECTION.findall(query)))
    if not sections:
        return []
    tsq = " | ".join(f"'{sec}'" for sec in sections)  # digits and dots only: safe to quote
    return _rows(CITATION_SQL, {"tsq": tsq, "k": k, "doc_ids": doc_ids}, "citation")


def rrf(*ranked: list[Hit], weights: list[float] | None = None, k: int = RRF_K) -> list[Hit]:
    """Weighted reciprocal rank fusion: score = sum over lists of w / (k + rank)."""
    weights = weights or [1.0] * len(ranked)
    merged: dict[str, Hit] = {}
    for hits, w in zip(ranked, weights):
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.chunk_id, Hit(**{**h.__dict__, "scores": {}}))
            m.scores.update(h.scores)
            m.scores["rrf"] = m.scores.get("rrf", 0.0) + w / (k + rank)
    return sorted(merged.values(), key=lambda h: h.scores["rrf"], reverse=True)


@lru_cache
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return hits
    scores = _reranker().predict([(query, f"{h.section}\n{h.text}") for h in hits])
    for h, s in zip(hits, scores):
        h.scores["rerank"] = float(s)
    return sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)


def _candidates(query: str, mode: Mode, doc_ids: list[str] | None) -> list[list[Hit]]:
    if mode == "vector":
        return [vector_search(query, doc_ids=doc_ids)]
    if mode == "keyword":
        return [citation_search(query, doc_ids=doc_ids), keyword_search(query, doc_ids=doc_ids)]
    return [citation_search(query, doc_ids=doc_ids), vector_search(query, doc_ids=doc_ids),
            keyword_search(query, doc_ids=doc_ids)]


def _weights(mode: Mode) -> list[float]:
    return {"vector": [1.0], "keyword": [CITATION_WEIGHT, 1.0]}.get(mode, [CITATION_WEIGHT, 1.0, 1.0])


def search(
    query: str,
    top_k: int = 8,
    mode: Mode = "hybrid",
    use_rerank: bool = True,
    doc_ids: list[str] | None = None,
    extra_queries: list[str] | None = None,
    order_by_rerank: bool = False,
) -> list[Hit]:
    """Fused candidates for `query` (plus any `extra_queries`, e.g. the planner's rewrite).

    With `use_rerank`, every hit gets a cross-encoder score (used by callers as a relevance gate),
    but the order stays RRF unless `order_by_rerank`: on the eval set, letting the MiniLM reranker
    reorder lowered recall@5 (0.89 -> 0.86) and MRR (0.85 -> 0.81).
    """
    lists: list[list[Hit]] = []
    weights: list[float] = []
    for q in [query, *(e for e in (extra_queries or []) if e and e != query)]:
        lists += _candidates(q, mode, doc_ids)
        weights += _weights(mode)
    hits = rrf(*lists, weights=weights) if len(lists) > 1 else lists[0]
    if not use_rerank:
        return hits[:top_k]
    if not order_by_rerank:
        return rerank_scores(query, hits[:top_k])
    hits = rerank(query, hits)
    # The cross-encoder can't read section numbers, so chunks that literally cite a section the
    # user named keep up to half the slots regardless of rerank score.
    pinned = [h for h in hits if "citation" in h.scores][: top_k // 2]
    rest = [h for h in hits if h not in pinned]
    return sorted(pinned + rest[: top_k - len(pinned)], key=lambda h: h.scores["rerank"], reverse=True)


def rerank_scores(query: str, hits: list[Hit]) -> list[Hit]:
    """Attach cross-encoder scores without changing order."""
    order = [h.chunk_id for h in hits]
    by_id = {h.chunk_id: h for h in rerank(query, list(hits))}
    return [by_id[c] for c in order]


def fetch_chunks(chunk_ids: list[str]) -> list[Hit]:
    """Load specific chunks (e.g. graph evidence) in the order given."""
    if not chunk_ids:
        return []
    sql = f"SELECT {_COLUMNS}, 0.0 FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ANY(%(ids)s)"
    by_id = {h.chunk_id: h for h in _rows(sql, {"ids": chunk_ids, "doc_ids": None}, "graph")}
    return [by_id[i] for i in chunk_ids if i in by_id]
