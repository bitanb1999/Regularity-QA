# Answer cache: before / after

8 questions, each asked cold (cache miss) then warm (cache hit, different casing). Latency is end-to-end inside the API process; cost is the Groq list-price equivalent.

| Question | Status | Cold (ms) | Warm (ms) | Answer-model tokens cold → warm | Cost (all LLM calls) cold → warm | Identical |
|---|---|---|---|---|---|---|
| What data-integrity problems did FDA find at Curia? | answered | 20,639 | 4 | 2,404 → 0 | $0.00054 → $0 | yes |
| What pathogens were found in Raaw Energy's dog food? | answered | 20,582 | 5 | 2,359 → 0 | $0.00051 → $0 | yes |
| Which companies were cited under 21 CFR Part 211, and for wh | answered | 29,918 | 3 | 4,444 → 0 | $0.00099 → $0 | yes |
| What did FDA say about Epicur's claim that its tablets are m | answered | 2,046 | 8 | 3,147 → 0 | $0.00061 → $0 | yes |
| Which firms were cited under 21 CFR 1.502? | answered | 2,296 | 3 | 2,216 → 0 | $0.00050 → $0 | yes |
| How did FDA describe the tofu cooling problem at Binh Minh T | answered | 11,627 | 2 | 1,564 → 0 | $0.00046 → $0 | yes |
| What did FDA find at Pfizer's manufacturing site? | refused | 1,523 | 3 | 0 → 0 | $0.00010 → $0 | yes |
| Which sites had repeat observations from a previous inspecti | refused | 2,033 | 6 | 0 → 0 | $0.00010 → $0 | yes |

- **Median latency:** 6,962 ms cold → 4 ms warm
- **Answer-model tokens per repeated question:** 2,017 → 0
- **Cost per repeated question:** $0.00047 → $0
- **Hits:** 8/8; identical answers and citations: 8/8

Cold latency includes any Groq free-tier rate-limit waits. The cache only helps repeated questions; it is exact-match by design (see app/cache.py).
