"""Regression gate over the eval set (eval/questions.jsonl).

    pytest -m eval                         # retrieval gate: live Postgres/Neo4j, cached planner output
    RUN_ANSWER_EVAL=1 pytest -m eval       # + end-to-end answers with the LLM judge (slow, rate-limited)

Thresholds live in eval/thresholds.json; raise them deliberately when the system improves.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

THRESHOLDS = json.loads((Path(__file__).resolve().parents[1] / "eval" / "thresholds.json").read_text())


def _services_up() -> str | None:
    try:
        from app.db import neo4j, pg_pool
        with pg_pool().connection() as conn:
            if conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
                return "no chunks ingested (run python -m app.ingest.pipeline)"
        neo4j().verify_connectivity()
    except Exception as e:  # noqa: BLE001 - any failure here means "environment not ready"
        return f"services unavailable: {type(e).__name__}"
    return None


@pytest.fixture(scope="module")
def questions():
    if reason := _services_up():
        pytest.skip(reason)
    from eval.run_eval import load_questions
    return load_questions()


@pytest.fixture(scope="module")
def plans(questions):
    from eval.run_eval import load_plans
    return load_plans(questions)


@pytest.fixture(scope="module")
def retrieval(questions, plans):
    from eval.run_eval import run_retrieval
    return run_retrieval(questions, plans)


@pytest.mark.parametrize("system", sorted(THRESHOLDS["retrieval"]))
def test_retrieval_thresholds(retrieval, system):
    got = retrieval["systems"][system]["overall"]
    for metric, minimum in THRESHOLDS["retrieval"][system].items():
        assert got[metric] >= minimum, f"{system} {metric} = {got[metric]} < {minimum}"


def test_full_system_beats_vector_baseline(retrieval):
    s = retrieval["systems"]
    margin = THRESHOLDS["min_recall_gain_over_vector"]
    assert s["hybrid+planner+graph"]["overall"]["recall@5"] >= s["vector"]["overall"]["recall@5"] + margin
    assert s["hybrid"]["overall"]["recall@5"] >= s["vector"]["overall"]["recall@5"]


def test_exact_citation_lookup_never_regresses(retrieval):
    got = retrieval["systems"]["hybrid+planner+graph"]["by_category"]["citation"]["recall@5"]
    assert got == 1.0, f"citation-question recall@5 dropped to {got}"


@pytest.mark.skipif(os.environ.get("RUN_ANSWER_EVAL") != "1", reason="set RUN_ANSWER_EVAL=1 (slow, calls LLMs)")
def test_answer_thresholds(questions, plans):
    from eval.run_eval import run_answers
    got = run_answers(questions, plans, pause_s=6)["overall"]
    for metric, bound in THRESHOLDS["answers"].items():
        if metric.endswith("_max"):
            name = metric.removesuffix("_max")
            assert got[name] <= bound, f"{name} = {got[name]} > {bound}"
        else:
            assert got[metric] >= bound, f"{metric} = {got[metric]} < {bound}"


@pytest.mark.skipif(os.environ.get("RUN_ANSWER_EVAL") != "1", reason="set RUN_ANSWER_EVAL=1 (calls the judge LLM)")
def test_judge_catches_planted_fabrications():
    """A judge that scores everything 1.0 is useless; it must flag claims absent from the context."""
    from eval.run_eval import judge
    context = ("Question: What did FDA find?\n\nSources:\n[S1] Curia New York, Inc. — warning letter 2026-09-18 — "
               "Violations\nAnalysts shared usernames and passwords to access GMP computerized systems. "
               "Original records were found in shred bins.\n")
    answer = ("- Analysts shared usernames and passwords for GMP systems [S1].\n"
              "- Original records were discarded in shred bins [S1].\n"
              "- FDA fined Curia $1.2 million for these failures [S1].")
    verdicts = {c.claim: c.supported for c in judge(answer, context).claims}
    assert any("1.2" in claim and not ok for claim, ok in verdicts.items()), verdicts
    assert sum(verdicts.values()) >= 2, verdicts
