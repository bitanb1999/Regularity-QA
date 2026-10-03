"""Evaluation harness: retrieval quality per system, and (opt-in) end-to-end answer quality.

    python -m eval.run_eval                 # retrieval comparison only (no LLM calls once plans are cached)
    python -m eval.run_eval --answers       # + full-system answers, judged (slow: Groq rate limits)
    python -m eval.run_eval --refresh-plans # re-run the planner instead of using eval/cache/plans.json

Relevance labels are (doc, phrase) items rather than chunk ids, so they survive re-chunking:
a retrieved chunk satisfies an item if it belongs to that doc and contains the phrase (or, for
doc-only items, belongs to the doc).

  recall@k  = satisfied items in the top k / all items
  MRR@10    = 1 / rank of the first chunk satisfying any item (0 if none in the top 10)
  hit@k     = 1 if any item is satisfied in the top k
"""
import argparse
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import openai
from pydantic import BaseModel, ConfigDict

from app.config import settings
from app.db import pg_pool
from app.llm import answer as ans
from app.llm import planner
from app.llm.client import structured
from app.models.api import AskRequest
from app.qa import ask, prepare
from app.retrieval.hybrid import search

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
PLAN_CACHE = EVAL_DIR / "cache" / "plans.json"
RESULTS = EVAL_DIR / "results"
K = 5
DEPTH = 10

# name -> how to retrieve. "raw" systems search with the question as typed; "planned" systems go
# through qa.prepare (query rewrite + company filter, and optionally the graph).
SYSTEMS: dict[str, dict] = {
    "vector": {"kind": "raw", "mode": "vector", "use_rerank": False},
    "keyword": {"kind": "raw", "mode": "keyword", "use_rerank": False},
    "hybrid": {"kind": "raw", "mode": "hybrid", "use_rerank": False},
    "hybrid+rerank": {"kind": "raw", "mode": "hybrid", "use_rerank": True, "order_by_rerank": True},
    # Full pipeline (qa.prepare): RRF order, reranker as gate, planner query + company filter.
    "hybrid+planner": {"kind": "planned", "use_graph": False},
    "hybrid+planner+graph": {"kind": "planned", "use_graph": True},
}


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("‑", "-").replace("‐", "-")).lower()


# --- planner cache -----------------------------------------------------------------------------

def _plan_key(question: str) -> str:
    h = hashlib.sha256(f"{settings.router_model}\n{planner.SYSTEM}\n{question}".encode()).hexdigest()
    return h[:16]


def load_plans(questions: list[dict], refresh: bool = False) -> dict[str, tuple[planner.Plan, str | None]]:
    cache = json.loads(PLAN_CACHE.read_text()) if PLAN_CACHE.exists() and not refresh else {}
    out, dirty = {}, False
    for q in questions:
        key = _plan_key(q["question"])
        if key not in cache:
            p, err = planner.plan(q["question"])
            if err:  # don't cache a fallback plan
                out[q["id"]] = (p, err)
                continue
            cache[key] = {"question": q["question"], "plan": p.model_dump()}
            dirty = True
        out[q["id"]] = (planner.Plan.model_validate(cache[key]["plan"]), None)
    if dirty:
        PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PLAN_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return out


# --- retrieval ---------------------------------------------------------------------------------

@dataclass
class Corpus:
    doc_of: dict[str, str]
    text_of: dict[str, str]

    @classmethod
    def load(cls) -> "Corpus":
        with pg_pool().connection() as conn:
            rows = conn.execute("SELECT id, doc_id, text FROM chunks").fetchall()
        return cls({r[0]: r[1] for r in rows}, {r[0]: _norm(r[2]) for r in rows})

    def satisfies(self, chunk_id: str, item: dict) -> bool:
        if self.doc_of.get(chunk_id) != item["doc"]:
            return False
        return "phrase" not in item or _norm(item["phrase"]) in self.text_of[chunk_id]


def retrieve(system: dict, q: dict, plan: tuple | None) -> list[str]:
    if system["kind"] == "raw":
        hits = search(q["question"], top_k=DEPTH, mode=system["mode"], use_rerank=system["use_rerank"],
                      order_by_rerank=system.get("order_by_rerank", False))
        return [h.chunk_id for h in hits]
    req = AskRequest(question=q["question"], top_k=DEPTH, use_graph=system["use_graph"])
    return [s.chunk_id for s in prepare(req, plan).sources]


def score_retrieval(corpus: Corpus, ranked: list[str], items: list[dict]) -> dict:
    top_k = ranked[:K]
    satisfied = [any(corpus.satisfies(c, it) for c in top_k) for it in items]
    first = next((i for i, c in enumerate(ranked[:DEPTH], 1) if any(corpus.satisfies(c, it) for it in items)), None)
    return {"recall@5": sum(satisfied) / len(items), "mrr@10": 1 / first if first else 0.0,
            "hit@5": float(any(satisfied))}


def run_retrieval(questions: list[dict], plans: dict, systems: list[str] | None = None) -> dict:
    corpus = Corpus.load()
    labelled = [q for q in questions if q["answerable"] and q["relevant"]]
    out: dict = {"n": len(labelled), "systems": {}}
    for name in systems or SYSTEMS:
        per_q = {}
        for q in labelled:
            ranked = retrieve(SYSTEMS[name], q, plans.get(q["id"]))
            per_q[q["id"]] = {**score_retrieval(corpus, ranked, q["relevant"]),
                              "category": q["category"], "ranked": ranked[:DEPTH]}
        out["systems"][name] = {"per_question": per_q, **_aggregate(per_q, ("recall@5", "mrr@10", "hit@5"))}
    return out


def _aggregate(per_q: dict, metrics: tuple[str, ...]) -> dict:
    def mean(rows, m):
        vals = [r[m] for r in rows if r.get(m) is not None]
        return round(statistics.mean(vals), 3) if vals else None
    rows = list(per_q.values())
    cats = sorted({r["category"] for r in rows})
    return {"overall": {m: mean(rows, m) for m in metrics},
            "by_category": {c: {m: mean([r for r in rows if r["category"] == c], m) for m in metrics}
                            for c in cats}}


# --- answers -----------------------------------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(_Strict):
    claim: str
    supported: bool


class Judgement(_Strict):
    claims: list[Claim]


# A different family (qwen3.8-27b) would limit self-preference, but this Groq tier caps it at 1,000
# output tokens/min, so judge calls never fit. gpt-oss-20b has its own rate bucket and is unused
# during evals (planner output is cached). Same family as the answerer: some self-preference risk.
JUDGE_MODEL = "openai/gpt-oss-20b"
JUDGE_SYSTEM = """You check whether an ANSWER is faithful to the CONTEXT it was written from.
The CONTEXT is exactly what the answering model saw: numbered sources (each with a header giving
company, letter date and section) and, sometimes, knowledge-graph results.
List every factual claim the ANSWER makes (skip citations, headings and filler). For each, set
supported=true if the CONTEXT states it or it follows directly from it. Paraphrase is fine.
Claims that report what a letter, firm or website said (quotes, "the firm claimed ...") are
supported when the CONTEXT contains that statement; do not judge whether the quoted statement is
true. Added specifics (numbers, names, causes, dates) absent from the CONTEXT are unsupported."""


def judge_context(question: str, sources, graph) -> str:
    """Rebuild the answer model's prompt context (same labels, headers and graph rows)."""
    labels = {s.chunk_id: s.label for s in sources}
    gblock = ans.graph_block(graph.template, graph.rows, labels) if graph and graph.rows else None
    psources = [ans.PromptSource(s.label, s.company, s.issue_date, s.section, s.snippet) for s in sources]
    return ans.build_prompt(question, psources, gblock)


def judge(answer: str, context: str) -> Judgement:
    return structured(JUDGE_SYSTEM, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}", Judgement,
                      model=JUDGE_MODEL, max_tokens=1500)


def score_answer(corpus: Corpus, q: dict, resp) -> dict:
    """Deterministic metrics; faithfulness is added separately by _judge_row."""
    abstained = resp.status in ("refused", "rejected")
    text = _norm(resp.answer)
    row: dict = {"category": q["category"], "status": resp.status, "flags": resp.flags,
                 "answer": resp.answer, "cited": [c.chunk_id for c in resp.citations],
                 "latency_ms": resp.timings_ms.get("total"),
                 "tokens": resp.usage["prompt_tokens"] + resp.usage["completion_tokens"]}
    if not q["answerable"]:
        mentions = any(m in text for m in q.get("accept_if_mentions", []))
        row["correct_refusal"] = float(abstained or mentions)
        return row
    row["false_refusal"] = float(abstained)
    if abstained:
        return row
    if q["key_facts"]:
        row["key_fact_recall"] = sum(any(_norm(a) in text for a in group) for group in q["key_facts"]) / len(q["key_facts"])
    if q["relevant"] and row["cited"]:
        row["citation_relevance"] = sum(
            any(corpus.satisfies(c, it) for it in q["relevant"]) for c in row["cited"]) / len(row["cited"])
    row["context"] = judge_context(q["question"], resp.sources, resp.graph)
    return row


def _needs_judging(row: dict) -> bool:
    return row["status"] in ("answered", "flagged")


def _judge_row(q: dict, row: dict, plan) -> None:
    if not row.get("context"):  # rows saved before contexts were stored: rebuild deterministically
        prep = prepare(AskRequest(question=q["question"]), plan)
        row["context"] = judge_context(q["question"], prep.sources, prep.graph)
    try:
        j = judge(row["answer"], row["context"])
        row["faithfulness"] = (sum(c.supported for c in j.claims) / len(j.claims)) if j.claims else None
        row["unsupported_claims"] = [c.claim for c in j.claims if not c.supported]
        row.pop("judge_error", None)
        row["judge_key"] = _judge_key()
    except Exception as e:  # noqa: BLE001 - a judge failure shouldn't lose the other metrics
        row["judge_error"] = f"{type(e).__name__}: {e}"[:200]


PARTIAL = RESULTS / "answers-partial.jsonl"


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _answer_key() -> str:
    """Saved answers are reused while the answering setup is unchanged (they cost daily quota)."""
    return _hash(settings.answer_model, ans.SYSTEM)


def _judge_key() -> str:
    """Judgements are recomputed whenever the judge changes, without regenerating answers."""
    return _hash(JUDGE_MODEL, JUDGE_SYSTEM)


def _save_partial(rows: dict[str, dict]) -> None:
    key = _answer_key()
    PARTIAL.write_text("".join(json.dumps({"answer_key": key, "id": i, "row": r}) + "\n" for i, r in rows.items()))


def run_answers(questions: list[dict], plans: dict, pause_s: float = 0.0, resume: bool = True) -> dict:
    """Answers are rate-limited (per-minute and per-day on Groq's free tier), so each finished
    question is saved to answers-partial.jsonl and reused on the next run with the same answer
    setup. Saved answers are re-judged if the judge changed."""
    corpus = Corpus.load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if resume and PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("answer_key") == _answer_key():
                done[rec["id"]] = rec["row"]
    elif PARTIAL.exists():
        PARTIAL.unlink()

    per_q: dict[str, dict] = {}
    for i, q in enumerate(questions, 1):
        row = done.get(q["id"])
        fresh = row is None
        if fresh:
            try:
                resp = ask(AskRequest(question=q["question"], use_cache=False), plans.get(q["id"]))
            except openai.RateLimitError as e:
                if "per day" not in str(e):
                    raise
                # Daily quota: retrying for minutes won't help. Keep what's done; rerun later resumes.
                print(f"  daily token quota reached at {q['id']}; {len(per_q)} of {len(questions)} done. "
                      "Rerun later to resume.", flush=True)
                break
            row = score_answer(corpus, q, resp)
        if _needs_judging(row) and (row.get("judge_key") != _judge_key() or row.get("judge_error")):
            _judge_row(q, row, plans.get(q["id"]))
        per_q[q["id"]] = row
        done[q["id"]] = row
        _save_partial(done)
        print(f"  [{i}/{len(questions)}] {q['id']:7} {row['status']:9} "
              f"{'new' if fresh else 'saved':5} faith={row.get('faithfulness')}", flush=True)
        if fresh:
            time.sleep(pause_s)
    if not per_q:
        raise SystemExit("no answers yet (daily token quota?) — rerun later")
    metrics = ("correct_refusal", "false_refusal", "key_fact_recall", "citation_relevance",
               "faithfulness")
    agg = _aggregate(per_q, metrics)
    rows = list(per_q.values())
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    agg["overall"].update({
        "answered": round(sum(r["status"] == "answered" for r in rows) / len(rows), 3),
        "flagged": round(sum(r["status"] == "flagged" for r in rows) / len(rows), 3),
        "rejected": round(sum(r["status"] == "rejected" for r in rows) / len(rows), 3),
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
    })
    return {"n": len(rows), "complete": len(rows) == len(questions), "per_question": per_q, **agg}


# --- report ------------------------------------------------------------------------------------

def _fmt(v) -> str:
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def summary_markdown(retrieval: dict | None, answers: dict | None) -> str:
    lines = [f"# Evaluation results\n\n_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC._\n"]
    if retrieval:
        cats = sorted({c for s in retrieval["systems"].values() for c in s["by_category"]})
        lines += [f"## Retrieval ({retrieval['n']} answerable questions with relevance labels)\n",
                  "| System | Recall@5 | MRR@10 | Hit@5 | " + " | ".join(f"R@5 {c}" for c in cats) + " |",
                  "|---" * (4 + len(cats)) + "|"]
        for name, s in retrieval["systems"].items():
            o = s["overall"]
            lines.append(f"| {name} | {_fmt(o['recall@5'])} | {_fmt(o['mrr@10'])} | {_fmt(o['hit@5'])} | "
                         + " | ".join(_fmt(s["by_category"].get(c, {}).get("recall@5")) for c in cats) + " |")
        lines.append("")
    if answers:
        o = answers["overall"]
        partial = "" if answers.get("complete", True) else " — INCOMPLETE, daily quota hit; rerun to resume"
        lines += [f"## Answers (full system, {answers['n']} questions{partial})\n",
                  ("_Latency includes Groq free-tier rate-limit waits during the eval run; direct API "
                   "calls measured 2.5–3.5 s._\n"), "| Metric | Value |", "|---|---|"]
        for label, key in [("Correct refusals (unanswerable)", "correct_refusal"),
                           ("False refusals (answerable)", "false_refusal"),
                           ("Key-fact recall", "key_fact_recall"), ("Citation relevance", "citation_relevance"),
                           ("Faithfulness (LLM judge)", "faithfulness"), ("Answered", "answered"),
                           ("Flagged", "flagged"), ("Rejected by citation check", "rejected"),
                           ("Latency p50 (ms)", "latency_p50_ms"), ("Latency p95 (ms)", "latency_p95_ms"),
                           ("Avg tokens / question", "avg_tokens")]:
            lines.append(f"| {label} | {_fmt(o.get(key))} |")
        lines += ["", "### By category\n", "| Category | Key facts | Faithfulness | False refusals | Correct refusals |",
                  "|---|---|---|---|---|"]
        for c, m in answers["by_category"].items():
            lines.append(f"| {c} | {_fmt(m['key_fact_recall'])} | {_fmt(m['faithfulness'])} | "
                         f"{_fmt(m['false_refusal'])} | {_fmt(m['correct_refusal'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also run and judge full-system answers")
    ap.add_argument("--refresh-plans", action="store_true")
    ap.add_argument("--only", nargs="*", help="question ids to run")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between answer requests")
    ap.add_argument("--fresh", action="store_true", help="discard saved partial answers and start over")
    args = ap.parse_args()

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    plans = load_plans(questions, refresh=args.refresh_plans)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    # Subset runs (--only) never overwrite the full-run "latest" files or summary.md.
    tag = "subset" if args.only else "latest"

    print(f"retrieval: {len(SYSTEMS)} systems x {len(questions)} questions")
    retrieval = run_retrieval(questions, plans)
    if not args.only:
        (RESULTS / f"retrieval-{stamp}.json").write_text(json.dumps(retrieval, indent=1))
    (RESULTS / f"retrieval-{tag}.json").write_text(json.dumps(retrieval, indent=1))

    answers = None
    if args.answers:
        print("answers:")
        answers = run_answers(questions, plans, args.pause, resume=not args.fresh)
        if not args.only:
            (RESULTS / f"answers-{stamp}.json").write_text(json.dumps(answers, indent=1))
        (RESULTS / f"answers-{tag}.json").write_text(json.dumps(answers, indent=1))
    elif (RESULTS / "answers-latest.json").exists():
        answers = json.loads((RESULTS / "answers-latest.json").read_text())

    md = summary_markdown(retrieval, answers)
    if not args.only:
        (RESULTS / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
