# Answer cache: before / after

2 questions, each asked cold (cache miss) then warm (cache hit, different casing). Latency is end-to-end inside the API process; cost is the Groq list-price equivalent.

| Question | Status | Cold (ms) | Warm (ms) | Tokens cold → warm | Cost cold → warm | Identical |
|---|---|---|---|---|---|---|
| What data-integrity problems did FDA find at Curia? | answered | 20,639 | 4 | 2,404 → 0 | $0.00054 → $0 | yes |
| What pathogens were found in Raaw Energy's dog food? | answered | 20,582 | 5 | 2,359 → 0 | $0.00051 → $0 | yes |

- **Median latency:** 20,610 ms cold → 4 ms warm
- **LLM tokens per repeated question:** 2,382 → 0
- **Cost per repeated question:** $0.00053 → $0
- **Hits:** 2/2; identical answers and citations: 2/2

Cold latency includes any Groq free-tier rate-limit waits. The cache only helps repeated questions; it is exact-match by design (see app/cache.py).
