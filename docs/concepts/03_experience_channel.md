# Exploratory Experience-Aware Retrieval Channel

**Status:** Exploratory contribution · negative-result case study (extractor-fixed v2)
**Position in thesis:** Methodology chapter — recall-pipeline ablation; Discussion chapter — error analysis

---

## Motivation

Dense (SBERT + FAISS) and lexical (BM25) retrieval both score candidates on
the *content* of a résumé against a job description. Neither captures the
*shape* of a career — total tenure, role progression, domain depth, or
employer history. A recruiter's intuition often integrates these signals
("15 years across senior backend roles signals real capability, even if
the literal words don't match"); the question is whether this can be
modelled as a third recall channel and measurably help retrieval.

The hypothesis: a third channel that scores candidates by trajectory will
surface strong-but-keyword-poor candidates that the skill-fit channels
miss, raising recall without sacrificing precision.

## Design

A `ResumeTrajectory` schema captures five structured features per résumé:
years of experience, seniority level, primary domain, a Tier-1 company
flag, and a tenure-depth signal. A symmetric `JobCriteria` schema is
extracted from each JD. A composite `trajectory_score` combines these
with weighted defaults (YOE-match 0.30, seniority-match 0.20, domain
overlap 0.25, tenure 0.10, tier 0.15). The Tier-1 flag is toggleable so
the prestige-bias risk can be ablated explicitly.

Two integration strategies were evaluated:

- **Parallel recall channel.** The experience index returns its own top-N,
  fused with dense and BM25 outputs via reciprocal rank fusion (equal
  weights and weighted variants).
- **Re-ranker over the fused pool.** Dense + BM25 produce a top-50 pool;
  trajectory scores re-order the pool with a configurable β balancing
  skill-fit against trajectory.

## Two extractor generations

The trajectory extractor is rules-based. Two versions were built and
diagnosed against the HF corpus (n=477 résumés):

| Diagnostic | v1 extractor (initial) | v2 extractor (fixed) |
|---|---|---|
| DIRECTOR classification rate | **50.5%** (broken — half the corpus) | **1.7%** (sensible) |
| MID (correct default) | 11.1% | **60.4%** |
| Tier-1 employer flag rate | **54.7%** (false positives from bare-word matching) | **4.0%** |
| YOE extracted (any value) | 26.2% | **34.0%** |

The v2 extractor fixed three concrete bugs:

1. **Tier-1 detection** now requires an employment-context phrase
   (`at <Company>`, `<Company> Inc.`, `<Company>, YYYY`, etc.) rather than
   bare-word matching. Bare-word matching incidentally fires on
   "*used Apple developer tools*" or "*shopped on Amazon*"; the new
   context check eliminates that.
2. **Seniority detection** now requires the seniority word to appear
   adjacent to an engineering / role noun. Substring matching fired on
   "*directed the team*" or "*Director of XYZ Bootcamp*"; the new regex
   requires `\b(senior|staff|principal|junior) (engineer|developer|...)`.
3. **YOE detection** expanded from 3 regex patterns to 10, plus a
   `since YYYY` branch that converts to YOE relative to a fixed corpus
   reference year. Caught ~30% more résumés.

## Results on the labelled benchmark (v2 extractor)

Evaluation used the HuggingFace `cnamuangtoun/resume-job-description-fit`
test split: 61 queries with at least one relevant result, 477 unique
résumés, 1,759 labelled (résumé, JD) pairs with three relevance grades.

| Configuration                        | P@10  | R@10  | NDCG@10 | MRR   |
|--------------------------------------|-------|-------|---------|-------|
| Dense (SBERT)                        | 0.098 | 0.106 | 0.115   | 0.249 |
| BM25                                 | 0.067 | 0.054 | 0.068   | 0.138 |
| **Hybrid 2-channel (Dense + BM25)**  | **0.108** | **0.075** | **0.115** | **0.243** |
| Experience only (v2)                 | 0.041 | 0.023 | 0.038   | 0.104 |
| Hybrid 3-channel (equal-weight RRF)  | 0.087 | 0.060 | 0.085   | 0.191 |
| Weighted RRF (w_exp=0.3)             | 0.107 | 0.074 | 0.107   | 0.230 |
| Trajectory rerank (β=0.9)            | 0.105 | 0.073 | 0.113   | 0.242 |
| Trajectory rerank (β=0.8)            | 0.102 | 0.072 | 0.111   | 0.241 |

At K=5, the best trajectory-aware configuration (Fix B β=0.9) produces a
marginal P@5 improvement over Hybrid-2 (0.125 vs 0.121), within noise on
61 queries. At K=10, no configuration surpasses Hybrid-2.

## Error analysis (now closed off)

The v1 result was confounded by extractor noise: approximately 50% of
résumés were mis-classified as Director-level and 55% were assigned
Tier-1 company status. Manual inspection confirmed widespread false
positives — Electrical Engineers and Financial Accountants were
classified as Director-tier candidates; bare-word matching for company
names triggered on incidental mentions.

The v2 extractor reduces these false-positive rates by an order of
magnitude (DIRECTOR: 50.5% → 1.7%; Tier-1: 54.7% → 4.0%). With a
demonstrably accurate extractor, the trajectory channel **still** does
not improve over Dense + BM25 on the skill-fit benchmark.

This isolates the finding: the negative result is a property of the
*hypothesis applied to this benchmark*, not of the extractor
implementation. The obvious examiner objection ("did you fix the
extractor?") is now closed.

## Conclusion

The channel demonstrated conceptual value by surfacing certain
high-experience candidates missed by lexical and semantic retrieval — a
27-year ML engineer with Tier-1 employer history was promoted from dense
rank 207 to experience rank 9 in one case-study query, the intended
behaviour of the design. However, even with a corrected v2 extractor,
trajectory signal did not produce consistent improvements on benchmark
metrics, and the strongest reading is that **career-trajectory is
genuinely orthogonal to skill-fit labels** on this benchmark — not a
victim of extractor noise.

This is reported as an exploratory contribution rather than as a
component of the final architecture. The two-channel hybrid (Dense +
BM25) remains the production retrieval configuration in this work.

## Future work

Two distinct continuations of this line of work:

1. **Trajectory-labelled evaluation data.** The HuggingFace fit dataset
   rewards skill–role match, which is orthogonal to the career-shape
   signal the channel is designed to surface. A benchmark labelled on
   trajectory-relevance (e.g. promotion prediction, internal mobility
   data) would be a more appropriate test of the hypothesis. The current
   negative result is informative *about this benchmark* but does not
   refute the hypothesis in general.
2. **LLM-assisted structured extraction.** Even with v2 rule
   improvements, YOE was detected from only 34% of résumés — the rest
   are out-of-vocabulary phrasings. Using a local language model (e.g.
   Llama 3.1 8B via Ollama) to parse each résumé into the
   `ResumeTrajectory` schema at index-build time would close the
   remaining extraction gap. A clean "rules vs LLM extractor" ablation
   against the same retrieval pipeline would isolate whether further
   extraction quality (toward 100%) changes the conclusion.

Both are out of scope for the present dissertation. The v2 fix above
demonstrates that *additional rules engineering* on top of v2 will not
move the negative result.
