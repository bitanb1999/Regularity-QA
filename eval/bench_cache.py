"""Before/after numbers for the answer cache.

    python -m eval.bench_cache          # clears the cache, then runs each question cold and warm

Each question is asked twice through the full pipeline (qa.ask): the first call misses and fills
the cache, the second must hit it. Writes eval/results/cache.md.
"""
import json
import statistics
import sys
import time
from pathlib import Path

from app import cache
from app.llm.client import QuotaExhausted
from app.models.api import AskRequest
from app.qa import ask

QUESTIONS = [
    "What data-integrity problems did FDA find at Curia?",
    "What pathogens were found in Raaw Energy's dog food?",
    "Which companies were cited under 21 CFR Part 211, and for what?",
    "What did FDA say about Epicur's claim that its tablets are manufactured to FDA specifications?",
    "Which firms were cited under 21 CFR 1.502?",
    "How did FDA describe the tofu cooling problem at Binh Minh Tofu?",
    "What did FDA find at Pfizer's manufacturing site?",  # refused before any answer-model call
    "Which sites had repeat observations from a previous inspection?",  # refused from the graph
]
OUT = Path(__file__).resolve().parent / "results" / "cache.md"


PARTIAL = OUT.with_name("cache-partial.jsonl")


def run(pause_s: float = 5.0, fresh: bool = False) -> list[dict]:
    """Resumable: finished questions are kept in cache-partial.jsonl (cold runs cost daily quota).
    Only the entry for the question being measured is evicted, so other cached answers survive."""
    if fresh and PARTIAL.exists():
        PARTIAL.unlink()
    done = {r["question"]: r for r in map(json.loads, PARTIAL.read_text().splitlines())} if PARTIAL.exists() else {}
    rows = []
    for q in QUESTIONS:
        if q in done:
            rows.append(done[q])
            continue
        cache.delete(AskRequest(question=q))  # guarantee the cold run is a miss
        try:
            cold = ask(AskRequest(question=q))
        except QuotaExhausted as e:
            print(f"stopping: {e} ({len(rows)}/{len(QUESTIONS)} done; rerun to resume)")
            break
        warm = ask(AskRequest(question=q.upper()))  # different casing: must still hit
        r = {
            "question": q, "status": cold.status, "hit": warm.cached,
            "same_answer": warm.answer == cold.answer
            and [c.chunk_id for c in warm.citations] == [c.chunk_id for c in cold.citations],
            "cold_ms": cold.timings_ms["total"], "warm_ms": warm.timings_ms["total"],
            "cold_tokens": cold.usage["prompt_tokens"] + cold.usage["completion_tokens"],
            "warm_tokens": warm.usage["prompt_tokens"] + warm.usage["completion_tokens"],
            "cold_cost": cold.cost_usd, "warm_cost": warm.cost_usd,
        }
        rows.append(r)
        with PARTIAL.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  {r['status']:9} hit={r['hit']} same={r['same_answer']} "
              f"{r['cold_ms']:>6}ms -> {r['warm_ms']:>4}ms  tokens {r['cold_tokens']} -> {r['warm_tokens']}  {q[:50]}")
        time.sleep(pause_s)
    return rows


def report(rows: list[dict]) -> str:
    def med(k: str) -> float:
        return statistics.median(r[k] for r in rows)

    lines = [
        "# Answer cache: before / after\n",
        (f"{len(rows)} questions, each asked cold (cache miss) then warm (cache hit, different casing). "
         "Latency is end-to-end inside the API process; cost is the Groq list-price equivalent.\n"),
        "| Question | Status | Cold (ms) | Warm (ms) | Tokens cold → warm | Cost cold → warm | Identical |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['question'][:60]} | {r['status']} | {r['cold_ms']:,} | {r['warm_ms']:,} | "
                     f"{r['cold_tokens']:,} → {r['warm_tokens']} | ${r['cold_cost']:.5f} → ${r['warm_cost']:.0f} | "
                     f"{'yes' if r['same_answer'] else 'NO'} |")
    lines += [
        "",
        f"- **Median latency:** {med('cold_ms'):,.0f} ms cold → {med('warm_ms'):,.0f} ms warm",
        f"- **LLM tokens per repeated question:** {statistics.mean(r['cold_tokens'] for r in rows):,.0f} → 0",
        f"- **Cost per repeated question:** ${statistics.mean(r['cold_cost'] for r in rows):.5f} → $0",
        (f"- **Hits:** {sum(r['hit'] for r in rows)}/{len(rows)}; identical answers and citations: "
         f"{sum(r['same_answer'] for r in rows)}/{len(rows)}"),
        "",
        ("Cold latency includes any Groq free-tier rate-limit waits. The cache only helps repeated "
         "questions; it is exact-match by design (see app/cache.py)."),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = run(fresh="--fresh" in sys.argv)
    if rows:
        md = report(rows)
        OUT.write_text(md)
        print("\n" + md)
