"""Retrieval-behaviour diagnostic on real LinkedIn job queries.

Run:  OMP_NUM_THREADS=1 uv run python scripts/eval_linkedin_queries.py

What this does:
- Uses the COMBINED résumé corpus (HF + Kaggle = ~2,960) as the candidate pool.
- Samples N=100 real LinkedIn JDs as queries (unlabelled — there is no
  ground-truth relevance for these).
- For each query, retrieves top-K via Dense / BM25 / Hybrid-2.
- Reports:
    1. Inter-channel Jaccard agreement at K — do Dense and BM25 surface the
       same candidates? Convergence signals consistent retrieval; divergence
       signals each channel is contributing different information.
    2. Score-range stability — are similarities in the same range as on the
       HF labelled queries?
    3. Kaggle Category diversity in top-K — how varied is what the system
       surfaces? (Diversity here is a proxy for representational coverage.)

Why this exists:
- The labelled HF benchmark says "Dense > BM25 on P@10/R@10". This script
  asks the complementary question: "Do they at least agree on WHO they
  surface for OOD queries?" If the answer is no, hybrid fusion is doing
  more than just precision lifting — it's reconciling disagreement.

VIVA: "What's the point of an evaluation with no labels?"
- Out-of-distribution behaviour is itself measurable. P@K needs labels;
  Jaccard agreement, score-range stats, and category diversity don't.
- These metrics are reportable in their own right and are standard in
  retrieval-systems papers under "qualitative analysis".
"""

from __future__ import annotations

import statistics
from collections import Counter

from recruit.config import settings
from recruit.data.loaders import (
    load_hf_fit_dataset,
    load_kaggle_resumes,
    load_linkedin_jobs,
)
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    print("Loading résumé corpus (HF + Kaggle)...")
    hf_resumes, _, _ = load_hf_fit_dataset(split="test")
    kg_resumes = load_kaggle_resumes(
        settings.data_path / "raw" / "resumes_kaggle" / "Resume" / "Resume.csv"
    )
    all_resumes = hf_resumes + kg_resumes
    print(f"  corpus size: {len(all_resumes)} résumés "
          f"(HF {len(hf_resumes)} + Kaggle {len(kg_resumes)})")

    # Category map for diversity reporting; HF rows have no Kaggle category.
    category_by_index = [
        (r.target_role if r.target_role else "HF-unlabelled")
        for r in all_resumes
    ]

    print("\nSampling 100 LinkedIn JDs as queries...")
    linkedin_jobs = load_linkedin_jobs(
        settings.data_path / "raw" / "linkedin_jobs" / "postings.csv",
        sample_size=100,
        seed=42,
    )
    print(f"  queries: {len(linkedin_jobs)}")

    print("\nEncoding corpus with SBERT (cached)...")
    encoder = SBertEncoder()
    resume_vecs = embed_with_cache(
        encoder,
        [r.text for r in all_resumes],
        label="combined_hf_plus_kaggle",
    )
    dense_idx = DenseIndex(resume_vecs)
    bm25_idx = BM25Index([r.text for r in all_resumes])

    K = 10
    POOL = 50
    jaccard_dense_bm25: list[float] = []
    jaccard_dense_hybrid: list[float] = []
    jaccard_bm25_hybrid: list[float] = []
    dense_top_scores: list[float] = []
    bm25_top_scores: list[float] = []
    hybrid_top_score_proxy: list[float] = []
    category_counter: Counter = Counter()

    print(f"\nRetrieving top-{K} per channel for each query...")
    for j_idx, job in enumerate(linkedin_jobs):
        qv = encoder.encode([job.description])[0]
        dense_hits = dense_idx.search(qv, k=POOL)
        bm25_hits = bm25_idx.search(job.description, k=POOL)
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)

        dense_top = [h.index for h in dense_hits[:K]]
        bm25_top = [h.index for h in bm25_hits[:K]]
        hybrid_top = [c.resume_idx for c in fused[:K]]

        jaccard_dense_bm25.append(jaccard(dense_top, bm25_top))
        jaccard_dense_hybrid.append(jaccard(dense_top, hybrid_top))
        jaccard_bm25_hybrid.append(jaccard(bm25_top, hybrid_top))

        if dense_hits:
            dense_top_scores.append(dense_hits[0].score)
        if bm25_hits:
            bm25_top_scores.append(bm25_hits[0].score)
        if fused:
            hybrid_top_score_proxy.append(fused[0].rrf_score)

        # Diversity: which categories appear in the hybrid top-K?
        for idx in hybrid_top:
            category_counter[category_by_index[idx]] += 1

    def s(values: list[float]) -> str:
        if not values:
            return "—"
        return (
            f"mean={statistics.fmean(values):.3f}  "
            f"median={statistics.median(values):.3f}  "
            f"p10={sorted(values)[len(values)//10]:.3f}  "
            f"p90={sorted(values)[max(0, len(values)*9//10 - 1)]:.3f}"
        )

    print("\n" + "=" * 72)
    print(f"RETRIEVAL BEHAVIOUR  ·  {len(linkedin_jobs)} LinkedIn queries  ·  K={K}")
    print("=" * 72)

    print("\n1. Inter-channel Jaccard agreement on top-K:")
    print(f"   Dense ↔ BM25     : {s(jaccard_dense_bm25)}")
    print(f"   Dense ↔ Hybrid2  : {s(jaccard_dense_hybrid)}")
    print(f"   BM25  ↔ Hybrid2  : {s(jaccard_bm25_hybrid)}")
    print(
        "\n   Interpretation: low Dense↔BM25 + high Dense↔Hybrid means the "
        "two channels are surfacing different candidates, and the hybrid is "
        "biased toward Dense's picks."
    )

    print("\n2. Top-1 score distributions per channel:")
    print(f"   Dense (cosine, [-1, 1])  : {s(dense_top_scores)}")
    print(f"   BM25  (unbounded)        : {s(bm25_top_scores)}")
    print(f"   Hybrid (RRF, ~[0, 0.05]) : {s(hybrid_top_score_proxy)}")

    print(
        f"\n3. Category diversity in Hybrid-2 top-{K} (aggregated across "
        f"{len(linkedin_jobs)} queries):"
    )
    total_slots = len(linkedin_jobs) * K
    for cat, count in category_counter.most_common(15):
        pct = 100 * count / total_slots
        print(f"   {cat:<28s} {count:>5d}  ({pct:5.1f}%)")
    n_cats = len(category_counter)
    print(f"\n   Categories surfaced: {n_cats} (out of 24 Kaggle + 1 HF-unlabelled)")

    print("\n" + "=" * 72)
    print(
        "Compare to the labelled-eval finding (scripts/eval_combined_corpus.py):\n"
        "Dense and Hybrid-2 produced near-identical P@K/NDCG@K on labelled HF\n"
        "queries. If the Jaccard between them is also high here on unlabelled\n"
        "LinkedIn queries, that's evidence the labelled-eval result generalises.\n"
        "Low Dense↔BM25 + high Dense↔Hybrid is the *expected* and *desired*\n"
        "behaviour pattern."
    )


if __name__ == "__main__":
    main()
