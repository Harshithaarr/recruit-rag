"""IR evaluation on the combined HF + Kaggle résumé corpus.

Run:  OMP_NUM_THREADS=1 uv run python scripts/eval_combined_corpus.py

What this does:
- Loads HF labelled (résumé, JD) fit pairs (test split).
- Loads the Kaggle snehaanbhawal résumé corpus and adds it to the candidate
  pool — so retrieval has to find the right HF résumé among ~2,960 résumés
  (HF 477 + Kaggle 2,483) instead of just 477.
- Builds Dense + BM25 indexes on the combined pool.
- Re-runs the same 61 queries with relevance labels limited to HF résumés
  (the Kaggle ones are unlabelled noise — that's the point: can we still
  find the labelled relevant items in a much harder pool?).
- Reports the same P@K / R@K / NDCG / MRR table as eval_hf_fit.py for
  direct comparison.

Why this matters:
- A retriever that works on 477 candidates may collapse at 2,960. This is
  the cheapest test of generalisation to a realistic pool size.
- The numbers will drop relative to eval_hf_fit.py — that's expected and
  honest. The headline is whether the *ordering* of channels (Dense vs
  BM25 vs Hybrid) holds up.
"""

from __future__ import annotations

from recruit.config import settings
from recruit.data.loaders import (
    load_hf_fit_dataset,
    load_kaggle_resumes,
)
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.evaluation.runner import (
    Retriever,
    evaluate_retriever,
    format_aggregate_table,
    queries_from_qrels,
)
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion


def main() -> None:
    print("Loading HF fit dataset (queries + labelled positives)...")
    hf_resumes, hf_jobs, hf_qrels = load_hf_fit_dataset(split="test")
    print(f"  HF: {len(hf_resumes)} résumés · {len(hf_jobs)} JDs · "
          f"{sum(len(v) for v in hf_qrels.values())} labelled pairs")

    print("\nLoading Kaggle résumés (as unlabelled distractors)...")
    kg_resumes = load_kaggle_resumes(
        settings.data_path / "raw" / "resumes_kaggle" / "Resume" / "Resume.csv"
    )
    print(f"  Kaggle: {len(kg_resumes)} résumés across 24 categories")

    # Combined pool — HF first so its indices match earlier scripts; Kaggle
    # appended as distractors.
    all_resumes = list(hf_resumes) + list(kg_resumes)
    print(f"\nCombined corpus: {len(all_resumes)} résumés "
          f"(HF {len(hf_resumes)} + Kaggle {len(kg_resumes)})")

    # qrels still index by HF resume_ids — the combined corpus contains
    # them all (HF first), so the resume_id_to_index mapping handles both.
    resume_id_to_index = {r.resume_id: i for i, r in enumerate(all_resumes)}
    job_id_to_text = {j.job_id: j.description for j in hf_jobs}

    queries = queries_from_qrels(
        hf_qrels,
        resume_id_to_index=resume_id_to_index,
        job_id_to_text=job_id_to_text,
        min_relevant_per_query=1,
    )
    print(f"Queries with ≥1 relevant item in pool: {len(queries)}")

    print("\nEncoding combined corpus with SBERT (cached after first run)...")
    encoder = SBertEncoder()
    all_texts = [r.text for r in all_resumes]
    all_vecs = embed_with_cache(encoder, all_texts, label="combined_hf_plus_kaggle")
    print(f"  vectors: shape={all_vecs.shape}")

    print("Building indexes...")
    dense_idx = DenseIndex(all_vecs)
    bm25_idx = BM25Index(all_texts)

    def dense_retriever(query_text: str, k: int) -> list[int]:
        qv = encoder.encode([query_text])[0]
        return [h.index for h in dense_idx.search(qv, k=k)]

    def bm25_retriever(query_text: str, k: int) -> list[int]:
        return [h.index for h in bm25_idx.search(query_text, k=k)]

    def hybrid2_retriever(query_text: str, k: int) -> list[int]:
        pool = max(k * 4, 50)
        qv = encoder.encode([query_text])[0]
        dense_hits = dense_idx.search(qv, k=pool)
        bm25_hits = bm25_idx.search(query_text, k=pool)
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
        return [c.resume_idx for c in fused[:k]]

    retrievers: list[tuple[str, Retriever]] = [
        ("Dense only (SBERT + FAISS)",     dense_retriever),
        ("BM25 only",                      bm25_retriever),
        ("Hybrid 2-channel (Dense + BM25)", hybrid2_retriever),
    ]
    ks = [5, 10, 20]

    print()
    print("=" * 72)
    print(f"IR EVALUATION on COMBINED corpus  ·  pool size {len(all_resumes)}  ·  "
          f"n_queries={len(queries)}")
    print("=" * 72)
    for label, retriever in retrievers:
        _, aggregates = evaluate_retriever(queries, retriever, ks)
        print()
        print(format_aggregate_table(label, aggregates))
    print()
    print("=" * 72)
    print(
        "Compare to scripts/eval_hf_fit.py (pool size 477) for the same channels:\n"
        "the gap shows generalisation behaviour at a more realistic pool size.\n"
        "Channel ORDERING (Dense > BM25; Hybrid ≈ Dense) is what should be stable;\n"
        "absolute numbers will drop because the pool is now 6× larger."
    )


if __name__ == "__main__":
    main()
