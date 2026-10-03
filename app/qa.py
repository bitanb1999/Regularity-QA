"""Question answering: plan -> retrieve (graph and/or hybrid search) -> gate -> generate -> check.

Refusals happen at three points, cheapest first:
  1. before generation: unknown company, empty graph result, or nothing relevant retrieved
  2. by the model: it answers NOT_FOUND when the sources don't cover the question
  3. after generation: an answer without a valid citation is rejected
"""
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import openai

from app import cache
from app.config import settings
from app.db import neo4j, pg_pool
from app.graph.queries import AGGREGATE_TEMPLATES, GraphCall, InvalidGraphCall
from app.graph.queries import run as run_graph
from app.llm import answer as ans
from app.llm.client import QuotaExhausted, Usage, chat, chat_stream, raise_if_quota
from app.llm.planner import Plan, plan
from app.models.api import AskRequest, AskResponse, GraphOut, SourceOut
from app.observability import NOOP, cost, generation_update, start_trace
from app.retrieval.hybrid import Hit, fetch_chunks, rerank, search

log = logging.getLogger(__name__)

MIN_RERANK = -5.0  # ms-marco logit; off-topic questions score around -11, weak-but-relevant around -1
# Graph rows whose first evidence chunk is loaded as a source. A row without evidence can't be
# cited, so the model drops it; for rankings that silently changes the answer. Aggregates are
# therefore capped at the same number of rows. Sized to stay under Groq's 8k tokens/min.
GRAPH_EVIDENCE_ROWS = 10
AGGREGATE_ROWS = 10
MAX_SOURCES = 14
NO_SOURCES_MSG = "I couldn't find anything relevant to this question in the indexed FDA warning letters."


@dataclass
class Prepared:
    req: AskRequest
    plan: Plan
    plan_error: str | None
    graph: GraphOut | None
    sources: list[SourceOut]
    refusal: str | None  # set when we refuse before calling the answer model
    refusal_flag: str | None
    timings: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Usage] = field(default_factory=dict)  # per model, for cost accounting


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _company_doc_ids(company: str) -> list[str]:
    with pg_pool().connection() as conn:
        rows = conn.execute("SELECT id FROM documents WHERE company ILIKE %s", (f"%{company}%",)).fetchall()
    return [r[0] for r in rows]


def _document_count() -> int:
    with pg_pool().connection() as conn:
        return conn.execute("SELECT count(*) FROM documents").fetchone()[0]


def _graph_evidence(question: str, rows: list[dict]) -> list[Hit]:
    """One evidence chunk per graph row: the row's chunk that best matches the question (not just
    the observation's first chunk), with rows reordered so different letters come first."""
    rows = [r for r in rows if r.get("chunk_ids")]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r.get("doc_id") or r.get("regulation") or r.get("topic") or "", []).append(r)
    diverse: list[dict] = []
    while len(diverse) < len(rows):
        for group in by_doc.values():
            if group:
                diverse.append(group.pop(0))
    diverse = diverse[:GRAPH_EVIDENCE_ROWS]

    chunks = {h.chunk_id: h for h in rerank(question, fetch_chunks(
        sorted({c for r in diverse for c in r["chunk_ids"]})))}
    picked: list[Hit] = []
    for r in diverse:
        best = max((chunks[c] for c in r["chunk_ids"] if c in chunks),
                   key=lambda h: h.scores["rerank"], default=None)
        if best and best not in picked:
            picked.append(best)
    return picked


def _source(label: str, h: Hit, origin: str) -> SourceOut:
    return SourceOut(
        label=label, chunk_id=h.chunk_id, doc_id=h.doc_id, company=h.company, issue_date=h.issue_date,
        section=h.section, url=h.url, origin=origin, rerank_score=h.scores.get("rerank"),
        snippet=h.text,
    )


def prepare(req: AskRequest, planned: tuple[Plan, str | None] | None = None, trace=NOOP) -> Prepared:
    """`planned` lets the eval harness inject a cached plan so retrieval runs are reproducible.
    `trace` is the request's root span; each stage records a child observation on it."""
    timings: dict[str, int] = {}
    usage: dict[str, Usage] = {}
    t0 = time.perf_counter()
    span = trace.start_observation(name="plan", as_type="generation", input=req.question,
                                   model=settings.router_model)
    if planned:
        p, plan_error = planned
        span.update(metadata={"cached": True})
    else:
        usage[settings.router_model] = Usage()
        p, plan_error = plan(req.question, usage[settings.router_model])
        u = usage[settings.router_model]
        generation_update(span, settings.router_model, u.prompt_tokens, u.completion_tokens)
    span.update(output=p.model_dump(), level="WARNING" if plan_error else None, status_message=plan_error)
    span.end()
    timings["plan"] = _ms(t0)

    def refuse(msg: str, flag: str, graph: GraphOut | None = None) -> Prepared:
        return Prepared(req, p, plan_error, graph, [], msg, flag, timings, usage)

    doc_ids: list[str] | None = None
    if p.company:
        doc_ids = _company_doc_ids(p.company)
        if not doc_ids:
            return refuse(f"There are no indexed warning letters for “{p.company}”.", "unknown_company")

    # Graph
    graph: GraphOut | None = None
    graph_hits: list[Hit] = []
    if req.use_graph and p.route != "semantic" and p.graph:
        t0 = time.perf_counter()
        call = GraphCall(**p.graph.model_dump())
        if call.template in AGGREGATE_TEMPLATES:
            call.limit = AGGREGATE_ROWS
        span = trace.start_observation(name="graph", as_type="tool", input=p.graph.model_dump())
        try:
            rows = run_graph(neo4j(), call)
            graph = GraphOut(template=call.template, params=call.validated().params(), rows=rows)
            span.update(output={"rows": len(rows)})
        except InvalidGraphCall as e:
            graph = GraphOut(template=call.template, params=p.graph.model_dump(), rows=[], error=str(e))
            span.update(level="WARNING", status_message=f"rejected by whitelist: {e}")
        span.end()
        timings["graph"] = _ms(t0)
        if graph.rows:
            graph_hits = _graph_evidence(req.question, graph.rows)
            # "both": restrict text search to the letters the graph matched
            row_docs = sorted({r["doc_id"] for r in graph.rows if r.get("doc_id")})
            if p.route == "both" and row_docs:
                doc_ids = sorted(set(doc_ids) & set(row_docs)) if doc_ids else row_docs
        elif p.route == "structural" and not graph.error:
            n = _document_count()
            return refuse(f"None of the {n} indexed warning letters has {call.validated().describe()}.",
                          "graph_empty", graph)

    # Text search on every route: for structural questions it supplies passages (e.g. a letter's
    # intro) that cite what the graph rows only summarize. Both the question as asked and the
    # planner's rewrite are searched: the rewrite helped some eval questions and dropped key
    # terms ("importers", "FSVP") on others.
    t0 = time.perf_counter()
    span = trace.start_observation(name="search", as_type="retriever", input={
        "query": req.question, "rewrite": p.search_query, "mode": req.mode, "doc_filter": doc_ids})
    hits = search(req.question, top_k=req.top_k, mode=req.mode, use_rerank=req.use_rerank,
                  doc_ids=doc_ids, extra_queries=[p.search_query])
    span.update(output=[{"chunk": h.chunk_id, "rerank": round(h.scores.get("rerank", 0.0), 2)} for h in hits])
    span.end()
    timings["search"] = _ms(t0)
    if req.use_rerank and not graph_hits:
        hits = [h for h in hits if h.scores.get("rerank", 0.0) >= MIN_RERANK]

    if not graph_hits and not hits:
        return refuse(NO_SOURCES_MSG, "no_relevant_sources", graph)

    # Structural: graph evidence leads (it is the answer). Both: graph evidence and text hits
    # compete on relevance, so a company-wide graph result can't crowd out the specific passage.
    # One graph chunk per distinct letter stays pinned first, so list questions ("which firms…")
    # still show every matched letter.
    ranked = [*((g, "graph") for g in graph_hits), *((h, "search") for h in hits)]
    if p.route == "both" and req.use_rerank:
        pinned, docs = [], set()
        for g in graph_hits:
            if g.doc_id not in docs:
                docs.add(g.doc_id)
                pinned.append((g, "graph"))
        rest = sorted((x for x in ranked if x not in pinned),
                      key=lambda x: x[0].scores.get("rerank", float("-inf")), reverse=True)
        ranked = pinned + rest

    seen: set[str] = set()
    sources: list[SourceOut] = []
    for h, origin in ranked:
        if h.chunk_id in seen or len(sources) >= MAX_SOURCES:
            continue
        seen.add(h.chunk_id)
        sources.append(_source(f"S{len(sources) + 1}", h, origin))
    return Prepared(req, p, plan_error, graph, sources, None, None, timings, usage)


def _prompt(prep: Prepared) -> str:
    labels = {s.chunk_id: s.label for s in prep.sources}
    gblock = ans.graph_block(prep.graph.template, prep.graph.rows, labels) if prep.graph and prep.graph.rows else None
    psources = [ans.PromptSource(s.label, s.company, s.issue_date, s.section, s.snippet) for s in prep.sources]
    return ans.build_prompt(prep.req.question, psources, gblock)


def _response(prep: Prepared, status: str, text: str, flags: list[str], cited: list[str],
              draft: str | None, usage: Usage, trace=NOOP) -> AskResponse:
    by_label = {s.label: s for s in prep.sources}
    flags = [*([prep.plan_error] if prep.plan_error else []), *flags]
    if prep.graph and prep.graph.error:
        flags.append(f"graph_invalid: {prep.graph.error}")
    return AskResponse(
        question=prep.req.question, status=status, answer=text,
        citations=[by_label[c] for c in cited if c in by_label], flags=flags,
        route=prep.plan.route, plan=prep.plan.model_dump(), graph=prep.graph, sources=prep.sources,
        draft=draft, timings_ms=prep.timings,
        usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
        cost_usd=round(sum((cost(m, u.prompt_tokens, u.completion_tokens) or {"total": 0.0})["total"]
                           for m, u in prep.usage.items()), 6),
        trace_id=trace.trace_id,
    )


def _finish(prep: Prepared, text: str, usage: Usage, trace=NOOP) -> AskResponse:
    span = trace.start_observation(name="citation_check", as_type="guardrail", input=text)
    c = ans.check(text, {s.label for s in prep.sources})
    span.update(output={"status": c.status, "cited": c.cited, "flags": c.flags, "coverage": c.coverage},
                level="WARNING" if c.status in ("rejected", "flagged") else None)
    span.end()
    draft = text if c.status == "rejected" else None
    return _response(prep, c.status, c.answer, c.flags, c.cited, draft, usage, trace)


def _generation_span(prep: Prepared, trace):
    return trace.start_observation(
        name="answer", as_type="generation", model=settings.answer_model,
        input=[{"role": "system", "content": ans.SYSTEM}, {"role": "user", "content": _prompt(prep)}])


def _close_trace(trace, resp: AskResponse | None, error: str | None = None) -> None:
    if resp is not None:
        trace.update(
            output={"status": resp.status, "answer": resp.answer},
            metadata={"route": resp.route, "flags": resp.flags, "timings_ms": resp.timings_ms,
                      "cost_usd": resp.cost_usd, "cached": resp.cached,
                      "citations": [c.chunk_id for c in resp.citations]},
            level="WARNING" if resp.status in ("rejected", "flagged") else None,
        )
    if error:
        trace.update(level="ERROR", status_message=error)
    trace.end()


def _cached(req: AskRequest, trace, t_all: float) -> AskResponse | None:
    if not req.use_cache:
        return None
    span = trace.start_observation(name="cache", as_type="span", input={"question": req.question})
    hit = cache.get(req)
    span.update(output={"hit": hit is not None})
    span.end()
    if hit is None:
        return None
    hit.cached, hit.trace_id = True, trace.trace_id
    hit.cost_usd, hit.usage = 0.0, {"prompt_tokens": 0, "completion_tokens": 0}
    hit.timings_ms = {"cache": _ms(t_all), "total": _ms(t_all)}
    return hit


def ask(req: AskRequest, planned: tuple[Plan, str | None] | None = None) -> AskResponse:
    t_all = time.perf_counter()
    trace = start_trace("ask", input=req.question, metadata=req.model_dump(exclude={"question"}))
    if (hit := _cached(req, trace, t_all)) is not None:
        _close_trace(trace, hit)
        return hit
    try:
        prep = prepare(req, planned, trace)
        usage = Usage()
        if prep.refusal:
            resp = _response(prep, "refused", prep.refusal, [prep.refusal_flag], [], None, usage, trace)
        else:
            t0 = time.perf_counter()
            span = _generation_span(prep, trace)
            try:
                text = chat(ans.SYSTEM, _prompt(prep), settings.answer_model, usage)
            except openai.RateLimitError as e:
                span.update(level="ERROR", status_message=str(e)[:300])
                span.end()
                raise_if_quota(e, settings.answer_model)
                raise
            generation_update(span, settings.answer_model, usage.prompt_tokens, usage.completion_tokens,
                              output=text)
            span.end()
            prep.usage[settings.answer_model] = usage
            prep.timings["generate"] = _ms(t0)
            resp = _finish(prep, text, usage, trace)
        resp.timings_ms["total"] = _ms(t_all)
    except Exception as e:
        _close_trace(trace, None, f"{type(e).__name__}: {e}"[:300])
        raise
    if req.use_cache:
        cache.put(req, resp)
    _close_trace(trace, resp)
    return resp


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def ask_stream(req: AskRequest) -> Iterator[str]:
    """Server-sent events: plan, sources, token*, done (the checked final answer + citations)."""
    t_all = time.perf_counter()
    trace = start_trace("ask", input=req.question, metadata={**req.model_dump(exclude={"question"})})
    if (hit := _cached(req, trace, t_all)) is not None:
        _close_trace(trace, hit)
        yield _sse("plan", {"route": hit.route, "plan": hit.plan,
                            "graph": hit.graph.model_dump() if hit.graph else None, "cached": True})
        yield _sse("sources", [s.model_dump(exclude={"snippet"}) for s in hit.sources])
        yield _sse("done", hit.model_dump(exclude={"sources"}))
        return
    try:
        prep = prepare(req, trace=trace)
    except Exception as e:
        _close_trace(trace, None, f"{type(e).__name__}: {e}"[:300])
        raise
    yield _sse("plan", {"route": prep.plan.route, "plan": prep.plan.model_dump(),
                        "graph": prep.graph.model_dump() if prep.graph else None})
    yield _sse("sources", [s.model_dump(exclude={"snippet"}) for s in prep.sources])
    usage = Usage()
    if prep.refusal:
        resp = _response(prep, "refused", prep.refusal, [prep.refusal_flag], [], None, usage, trace)
    else:
        t0 = time.perf_counter()
        parts: list[str] = []
        span = _generation_span(prep, trace)
        try:
            for delta in chat_stream(ans.SYSTEM, _prompt(prep), settings.answer_model, usage):
                parts.append(delta)
                yield _sse("token", {"text": delta})
        except openai.RateLimitError as e:
            span.update(level="ERROR", status_message=str(e)[:300])
            span.end()
            _close_trace(trace, None, "rate limited")
            try:
                raise_if_quota(e, settings.answer_model)
            except QuotaExhausted as q:
                yield _sse("error", {"message": "The answer model's daily token quota is used up"
                                     + (f"; try again in {q.retry_in}." if q.retry_in else "."),
                                     "code": "quota_exhausted"})
                return
            log.exception("answer stream failed")
            yield _sse("error", {"message": "The answer model is rate-limited; try again shortly.",
                                 "code": "rate_limited"})
            return
        except Exception as e:
            log.exception("answer stream failed")
            span.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
            span.end()
            _close_trace(trace, None, f"{type(e).__name__}: {e}"[:300])
            yield _sse("error", {"message": f"generation failed: {type(e).__name__}"})
            return
        generation_update(span, settings.answer_model, usage.prompt_tokens, usage.completion_tokens,
                          output="".join(parts))
        span.end()
        prep.usage[settings.answer_model] = usage
        prep.timings["generate"] = _ms(t0)
        resp = _finish(prep, "".join(parts), usage, trace)
    resp.timings_ms["total"] = _ms(t_all)
    if req.use_cache:
        cache.put(req, resp)
    _close_trace(trace, resp)
    # The client should replace the streamed text with `answer`: it may be stripped or rejected.
    yield _sse("done", resp.model_dump(exclude={"sources"}))
