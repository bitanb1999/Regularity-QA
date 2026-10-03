# Evaluation results

_Generated 2026-10-03 06:37 UTC._

## Retrieval (36 answerable questions with relevance labels)

| System | Recall@5 | MRR@10 | Hit@5 | R@5 both | R@5 citation | R@5 semantic | R@5 structural |
|---|---|---|---|---|---|---|---|
| vector | 0.76 | 0.69 | 0.86 | 0.50 | 0.42 | 0.95 | 0.70 |
| keyword | 0.85 | 0.76 | 0.94 | 1.00 | 1.00 | 0.88 | 0.55 |
| hybrid | 0.89 | 0.85 | 0.94 | 0.75 | 1.00 | 0.95 | 0.74 |
| hybrid+rerank | 0.86 | 0.79 | 0.94 | 0.83 | 1.00 | 0.88 | 0.69 |
| hybrid+planner | 0.89 | 0.90 | 0.97 | 0.83 | 1.00 | 0.93 | 0.74 |
| hybrid+planner+graph | 0.97 | 0.83 | 0.97 | 1.00 | 1.00 | 0.95 | 1.00 |

## Answers (full system, 45 questions)

_Latency includes Groq free-tier rate-limit waits during the eval run; direct API calls measured 2.5–3.5 s._

| Metric | Value |
|---|---|
| Correct refusals (unanswerable) | 1.00 |
| False refusals (answerable) | 0.00 |
| Key-fact recall | 1.00 |
| Citation relevance | 0.82 |
| Faithfulness (LLM judge) | 1.00 |
| Answered | 0.82 |
| Flagged | 0.02 |
| Rejected by citation check | 0.00 |
| Latency p50 (ms) | 11452 |
| Latency p95 (ms) | 27148 |
| Avg tokens / question | 2325 |

### By category

| Category | Key facts | Faithfulness | False refusals | Correct refusals |
|---|---|---|---|---|
| both | 1.00 | 1.00 | 0.00 | – |
| citation | 1.00 | 1.00 | 0.00 | – |
| semantic | 1.00 | 1.00 | 0.00 | – |
| structural | 1.00 | 1.00 | 0.00 | – |
| unanswerable | – | – | – | 1.00 |
