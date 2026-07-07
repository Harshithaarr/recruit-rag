"""IR evaluation on the HF resume-job-description-fit labelled dataset.

Run:  uv run python scripts/eval_hf_fit.py

What this does:
- Loads cnamuangtoun/resume-job-description-fit (test split: 1,759 labelled
  pairs covering 71 unique JDs and 477 unique résumés).
- Builds a corpus of all unique résumés.
- Builds qrels per JD using the three labels (Good Fit=2, Potential Fit=1,
  No Fit=0).
- Runs Dense / BM25 / Hybrid retrievers and prints the comparison.

Why this matters:
- These are the FIRST publishable IR numbers for the matcher chapter.
- The earlier scripts/eval_retrieval.py runs on hand-crafted sample data —
  smoke test only. This one runs on a third-party labelled benchmark.

VIVA: "Is this dataset representative?"
- Limitations: industry mix is US-centric, résumé length skews short, no
  candidate demographics. To be addressed in the limitations chapter.
- It is, however, the canonical labelled fit dataset on HuggingFace for
  this exact task, and is publicly auditable.
"""

from __future__ import annotations

from recruit.data.loaders import load_hf_fit_dataset
from recruit.data.schemas import Job
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.evaluation.runner import (
    Retriever,
    evaluate_retriever,
    format_aggregate_table,
    queries_from_qrels,
)
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.experience import (
    ExperienceIndex,
    extract_job_criteria,
    trajectory_score,
)
from recruit.retrieval.hybrid import (
    reciprocal_rank_fusion,
    rerank_by_trajectory,
    rrf_fuse_indices,
)


def main() -> None:
    print("Loading HF resume-job-description-fit (test split)...")
    resumes, jobs, qrels = load_hf_fit_dataset(split="test")
    print(f"  corpus: {len(resumes)} unique résumés")
    print(f"  pool:   {len(jobs)} unique JDs")
    print(f"  qrels:  {sum(len(v) for v in qrels.values())} labelled pairs")

    resume_id_to_index = {r.resume_id: i for i, r in enumerate(resumes)}
    job_id_to_text = {j.job_id: j.description for j in jobs}

    queries = queries_from_qrels(
        qrels,
        resume_id_to_index=resume_id_to_index,
        job_id_to_text=job_id_to_text,
        min_relevant_per_query=1,
    )
    print(f"  queries with ≥1 relevant item: {len(queries)}")

    print("\nEncoding corpus with SBERT (cached after first run)...")
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="hf_fit_test_resumes")
    print(f"  vectors: shape={resume_vecs.shape}")

    print("Building indexes...")
    dense_index = DenseIndex(resume_vecs)
    bm25_index = BM25Index(resume_texts)
    exp_index = ExperienceIndex(resumes, use_company_tier=True)
    exp_index_no_tier = ExperienceIndex(resumes, use_company_tier=False)

    def dense_retriever(query_text: str, k: int) -> list[int]:
        qv = encoder.encode([query_text])[0]
        return [h.index for h in dense_index.search(qv, k=k)]

    def bm25_retriever(query_text: str, k: int) -> list[int]:
        return [h.index for h in bm25_index.search(query_text, k=k)]

    def experience_retriever(query_text: str, k: int) -> list[int]:
        synthetic_job = Job(job_id="__inline__", title="", description=query_text)
        return [h.index for h in exp_index.search(synthetic_job, k=k)]

    def experience_no_tier_retriever(query_text: str, k: int) -> list[int]:
        synthetic_job = Job(job_id="__inline__", title="", description=query_text)
        return [h.index for h in exp_index_no_tier.search(synthetic_job, k=k)]

    def hybrid2_retriever(query_text: str, k: int) -> list[int]:
        pool = max(k * 4, 50)
        qv = encoder.encode([query_text])[0]
        dense_hits = dense_index.search(qv, k=pool)
        bm25_hits = bm25_index.search(query_text, k=pool)
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
        return [c.resume_idx for c in fused[:k]]

    def hybrid3_equal_retriever(query_text: str, k: int) -> list[int]:
        """Equal-weight RRF across all three channels — the failure-mode baseline."""
        pool = max(k * 4, 50)
        qv = encoder.encode([query_text])[0]
        synthetic_job = Job(job_id="__inline__", title="", description=query_text)
        dense_idx = [h.index for h in dense_index.search(qv, k=pool)]
        bm25_idx = [h.index for h in bm25_index.search(query_text, k=pool)]
        exp_idx = [h.index for h in exp_index.search(synthetic_job, k=pool)]
        return rrf_fuse_indices([dense_idx, bm25_idx, exp_idx])[:k]

    def make_weighted_hybrid3(exp_weight: float) -> Retriever:
        """FIX A — weighted RRF: down-weight the experience channel."""

        def retriever(query_text: str, k: int) -> list[int]:
            pool = max(k * 4, 50)
            qv = encoder.encode([query_text])[0]
            synthetic_job = Job(job_id="__inline__", title="", description=query_text)
            dense_idx = [h.index for h in dense_index.search(qv, k=pool)]
            bm25_idx = [h.index for h in bm25_index.search(query_text, k=pool)]
            exp_idx = [h.index for h in exp_index.search(synthetic_job, k=pool)]
            return rrf_fuse_indices(
                [dense_idx, bm25_idx, exp_idx],
                weights=[1.0, 1.0, exp_weight],
            )[:k]

        return retriever

    def make_experience_rerank(beta: float) -> Retriever:
        """FIX B — Dense+BM25 fused → top-pool re-ranked by trajectory score."""

        def retriever(query_text: str, k: int) -> list[int]:
            pool = max(k * 4, 50)
            qv = encoder.encode([query_text])[0]
            synthetic_job = Job(job_id="__inline__", title="", description=query_text)
            criteria = extract_job_criteria(synthetic_job)

            dense_hits = dense_index.search(qv, k=pool)
            bm25_hits = bm25_index.search(query_text, k=pool)
            fused_candidates = reciprocal_rank_fusion(dense_hits, bm25_hits)
            pool_indices = [c.resume_idx for c in fused_candidates[:pool]]

            trajectory_scores = [
                trajectory_score(
                    exp_index.trajectories[idx],
                    criteria,
                    use_company_tier=True,
                ).total
                for idx in pool_indices
            ]
            reranked = rerank_by_trajectory(pool_indices, trajectory_scores, beta=beta)
            return reranked[:k]

        return retriever

    retrievers: list[tuple[str, Retriever]] = [
        ("Dense only (SBERT + FAISS)",                          dense_retriever),
        ("BM25 only",                                           bm25_retriever),
        ("Experience only",                                     experience_retriever),
        ("Hybrid 2-channel (Dense + BM25)",                     hybrid2_retriever),
        ("Hybrid 3-channel — equal RRF (baseline failure)",     hybrid3_equal_retriever),
        # FIX A — sweep experience-channel weight
        ("Fix A · weighted RRF, w_exp=0.5",                     make_weighted_hybrid3(0.5)),
        ("Fix A · weighted RRF, w_exp=0.3",                     make_weighted_hybrid3(0.3)),
        ("Fix A · weighted RRF, w_exp=0.1",                     make_weighted_hybrid3(0.1)),
        # FIX B — sweep rerank beta (higher β = more weight on skill-fit)
        ("Fix B · trajectory rerank, β=0.9",                    make_experience_rerank(0.9)),
        ("Fix B · trajectory rerank, β=0.8",                    make_experience_rerank(0.8)),
        ("Fix B · trajectory rerank, β=0.7",                    make_experience_rerank(0.7)),
    ]

    ks = [5, 10, 20]

    print()
    print("=" * 64)
    print(f"IR EVALUATION on HF resume-job-description-fit  ·  n_queries={len(queries)}")
    print("=" * 64)
    for label, retriever in retrievers:
        _, aggregates = evaluate_retriever(queries, retriever, ks)
        print()
        print(format_aggregate_table(label, aggregates))
    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
