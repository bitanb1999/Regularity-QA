# Evaluation

`questions.jsonl` holds 45 hand-written questions over the 10 indexed warning letters:

| Category | n | What it tests |
|---|---|---|
| semantic | 19 | detail questions answered from one letter |
| citation | 6 | exact regulation lookups (`21 CFR 211.192`, `section 503B`) |
| structural | 9 | lists, filters, counts across letters |
| both | 4 | a structural filter plus text detail |
| unanswerable | 7 | out-of-corpus companies, facts the letters don't contain, redacted values |

Each answerable question has:

- `relevant`: `{doc, phrase}` items. A retrieved chunk satisfies an item if it belongs to `doc` and
  contains `phrase` (doc-only items match any chunk of the letter). Labels are phrases, not chunk
  ids, so they survive re-chunking. The harness checks every phrase exists in its letter.
- `key_facts`: groups of alternatives; a group counts if any alternative appears in the answer.

## Running

```bash
python -m eval.run_eval              # retrieval comparison (~1 min; planner output cached)
python -m eval.run_eval --answers    # + end-to-end answers and LLM judge (~30 min on Groq free tier)
pytest -m eval                       # regression gate (thresholds in thresholds.json)
RUN_ANSWER_EVAL=1 pytest -m eval     # gate including answers
```

Planner output is cached in `cache/plans.json` (keyed by question, planner prompt and model), so
retrieval runs are deterministic and make no LLM calls. Results go to `results/`; `summary.md` is
the human-readable table.

## Metrics

Retrieval (36 answerable questions with labels):

- **recall@5**: satisfied items in the top 5 / all items
- **MRR@10**: 1 / rank of the first chunk satisfying any item
- **hit@5**: any item satisfied in the top 5

Systems: `vector`, `keyword` (+ exact-citation list), `hybrid` (weighted RRF of citation, vector
and keyword lists), `hybrid+rerank` (MiniLM cross-encoder reorders), `hybrid+planner` (full
`qa.prepare` without the graph: original + rewritten query, company filter, reranker as relevance
gate) and `hybrid+planner+graph` (the production path).

Answers (full system, all 45 questions):

- **correct refusal**: unanswerable question refused or rejected (or, for redacted values, the
  answer says so)
- **false refusal**: answerable question refused or rejected
- **key-fact recall**, **citation relevance** (cited chunks that satisfy a relevance item)
- **faithfulness**: share of the answer's claims that `gpt-oss-20b` judges supported by the cited
  sources. A different model family would limit self-preference bias, but Qwen on this Groq tier
  allows only 1,000 output tokens/min, which no judge call fits in. Treat faithfulness as an
  upper bound and spot-check `unsupported_claims`.

## Caveats

- The retrieval pipeline was tuned against this same question set (see the root README's "what
  failed" notes). Treat the numbers as optimistic until checked on fresh questions.
- 10 letters and 45 questions: one question moves recall@5 by about 0.03.
- The LLM judge is itself a model; spot-check `unsupported_claims` in `results/answers-latest.json`.
