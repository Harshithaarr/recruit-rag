"""IR evaluation harness — ablation across Dense / BM25 / Hybrid.

Run:  uv run python scripts/eval_retrieval.py

What this does:
- Loads sample résumés, jobs, and hand-labelled qrels.
- Defines three retrievers — Dense (SBERT+FAISS), BM25, Hybrid (RRF).
- Evaluates each at K ∈ {3, 5, 10} on the qrels.
- Prints a comparison table — P@K, R@K, NDCG@K, MRR.

Why this is the keystone for the dissertation:
- Objective #2 claims the semantic matcher beats BM25 on P@10 and R@10.
- That claim is unmeasurable without this harness.
- Every future component (experience-channel, drop-off-aware ranking, RAG)
  plugs into the same harness — one more retriever, one more table row.

VIVA: "Why ablate Dense, BM25, Hybrid separately?"
- To show each contributes. If Hybrid only matched the best of the two on
  every metric, the hybrid layer would be doing no work. Ablation is the
  only way to defend the architectural choice.
"""

from __future__ import annotations

from recruit.config import settings
from recruit.data.loaders import load_jobs_csv, load_resumes_csv
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.evaluation.runner import (
    Retriever,
    evaluate_retriever,
    format_aggregate_table,
    load_qrels,
)
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion


def main() -> None:
    sample_dir = settings.data_path / "sample"
    resumes = load_resumes_csv(sample_dir / "resumes.csv")
    jobs = load_jobs_csv(sample_dir / "jobs.csv")

    # Build the retrievers' shared state once.
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="sample_resumes")

    dense_index = DenseIndex(resume_vecs)
    bm25_index = BM25Index(resume_texts)

    # Build the index lookup maps the runner needs.
    resume_id_to_index = {r.resume_id: i for i, r in enumerate(resumes)}
    job_id_to_text = {j.job_id: j.description for j in jobs}

    # Load qrels.
    qrels_path = settings.data_path / "eval" / "sample_qrels.json"
    queries = load_qrels(
        qrels_path,
        resume_id_to_index=resume_id_to_index,
        job_id_to_text=job_id_to_text,
    )
    print(f"Loaded {len(queries)} eval queries from {qrels_path.name}\n")

    # Retriever 1: dense only.
    def dense_retriever(query_text: str, k: int) -> list[int]:
        qv = encoder.encode([query_text])[0]
        hits = dense_index.search(qv, k=k)
        return [h.index for h in hits]

    # Retriever 2: BM25 only.
    def bm25_retriever(query_text: str, k: int) -> list[int]:
        hits = bm25_index.search(query_text, k=k)
        return [h.index for h in hits]

    # Retriever 3: hybrid (RRF fusion of dense + BM25).
    def hybrid_retriever(query_text: str, k: int) -> list[int]:
        # Always fuse with a wide-ish pool then slice; otherwise low-k hybrid
        # produces near-identical lists to either alone.
        pool = max(k * 4, 20)
        qv = encoder.encode([query_text])[0]
        dense_hits = dense_index.search(qv, k=pool)
        bm25_hits = bm25_index.search(query_text, k=pool)
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
        return [c.resume_idx for c in fused[:k]]

    retrievers: list[tuple[str, Retriever]] = [
        ("Dense only (SBERT + FAISS)", dense_retriever),
        ("BM25 only", bm25_retriever),
        ("Hybrid (RRF: SBERT + BM25)", hybrid_retriever),
    ]

    ks = [3, 5, 10]

    print("=" * 60)
    print("IR EVALUATION — ablation rows")
    print("=" * 60)
    for label, retriever in retrievers:
        _, aggregates = evaluate_retriever(queries, retriever, ks)
        print()
        print(format_aggregate_table(label, aggregates))
    print()
    print("=" * 60)
    print(
        "Reading the table:\n"
        "  P@K = fraction of top-K that are relevant.\n"
        "  R@K = fraction of all relevant items captured in top-K.\n"
        "  NDCG@K = rank-aware quality (best=1.0).\n"
        "  MRR = 1 / rank of first relevant item.\n"
        "Sample size is intentionally small (5 queries, 8 résumés) for a smoke test.\n"
        "Real evaluation will use HF resume-job-description-fit dataset."
    )


if __name__ == "__main__":
    main()
