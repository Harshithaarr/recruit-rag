"""End-to-end semantic matcher demo on the sample corpus.

Run:  uv run python scripts/demo_matcher.py

What this does:
- Loads sample résumés and jobs.
- Embeds résumés (and caches to indexes/resumes.npy).
- Picks one job (J002 "Machine Learning Engineer - NLP") as the query.
- Runs three rankings side-by-side:
    1. Dense only (SBERT + FAISS)
    2. BM25 only
    3. Hybrid (RRF) + structured filters
- Prints the top results from each so we can see where they agree and differ.

Why this demo is useful:
- It's the smallest piece of code that exercises every concept your matcher
  chapter has to defend: embeddings, FAISS, BM25, RRF, filters.
- The side-by-side print is exactly the qualitative-analysis figure you'll
  drop into the thesis.
"""

from __future__ import annotations

from recruit.config import settings
from recruit.data.loaders import load_jobs_csv, load_resumes_csv
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import filter_candidates, reciprocal_rank_fusion


def _format_row(rank: int, resume_id: str, name: str, score_label: str, score: float) -> str:
    return f"  {rank:>2}. {resume_id:5s} {name:<18s} | {score_label}={score:+.4f}"


def main() -> None:
    sample_dir = settings.data_path / "sample"
    resumes = load_resumes_csv(sample_dir / "resumes.csv")
    jobs = load_jobs_csv(sample_dir / "jobs.csv")

    # Pick the ML/NLP job as the query; this is the one where we expect
    # SBERT to do something BM25 cannot (paraphrase across "data scientist"
    # and "machine learning engineer").
    target_job = next(j for j in jobs if j.job_id == "J002")
    print(f"=== Query job: {target_job.job_id} {target_job.title} ===")
    print(f"    {target_job.description[:140]}...\n")

    # ---- Dense (SBERT + FAISS) ----
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="sample_resumes", force=True)
    job_vec = encoder.encode([target_job.description])[0]

    dense_index = DenseIndex(resume_vecs)
    dense_hits = dense_index.search(job_vec, k=len(resumes))

    print("--- Dense only (SBERT + FAISS) ---")
    for rank, hit in enumerate(dense_hits[:5], start=1):
        r = resumes[hit.index]
        print(_format_row(rank, r.resume_id, r.candidate_name or "?", "cos", hit.score))
    print()

    # ---- BM25 only ----
    bm25_index = BM25Index(resume_texts)
    bm25_hits = bm25_index.search(target_job.description, k=len(resumes))

    print("--- BM25 only ---")
    for rank, hit in enumerate(bm25_hits[:5], start=1):
        r = resumes[hit.index]
        print(_format_row(rank, r.resume_id, r.candidate_name or "?", "bm25", hit.score))
    print()

    # ---- Hybrid (RRF) + structured filters ----
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
    filtered = filter_candidates(fused, resumes, target_job)

    print("--- Hybrid (RRF) + structured filters (location, min YOE, certs) ---")
    for c in filtered[:5]:
        r = resumes[c.resume_idx]
        extras = []
        if c.dense_score is not None:
            extras.append(f"dense={c.dense_score:+.3f}")
        if c.bm25_score is not None:
            extras.append(f"bm25={c.bm25_score:+.2f}")
        extras.append(f"rrf={c.rrf_score:.4f}")
        print(
            f"  {c.rank:>2}. {r.resume_id:5s} {r.candidate_name or '?':<18s} | "
            + " ".join(extras)
        )

    print("\nDone. Try changing target_job.job_id to J001, J005, etc., and rerun.")


if __name__ == "__main__":
    main()
