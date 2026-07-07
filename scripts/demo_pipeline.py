"""End-to-end pipeline demo — Retrieval + Drop-off + Explanation in one command.

Run:  uv run python scripts/demo_pipeline.py [--job N] [--top-k 5]

What this does:
- Loads the HF labelled fit corpus as the candidate pool.
- Builds Hybrid-2 retrieval (Dense + BM25) over all 477 résumés.
- Picks one JD (by index — argparse `--job N`, default 0).
- Retrieves the Top-K candidates.
- For each candidate:
    1. Predicts drop-off probability + SHAP drivers via the trained model.
    2. Composes a templated explanation (matched/missing skills, fit
       sentence, drop-off note, recommendation).
    3. Computes a final fused score:
           final = w_sem · sem_similarity + w_stay · (1 - P_dropoff)
- Re-ranks by final score and prints the top-K with full explanations.

Why this script matters:
- It is the smallest piece of code that exercises THE FUSION NOVELTY of the
  dissertation: Semantic Matcher + Drop-off Predictor + SHAP-grounded
  Explainer working together. Every viva demonstration ultimately reduces
  to one invocation of this script.

VIVA: "Walk me through what happens when a recruiter submits a JD."
- This script IS that walk-through. Read top-to-bottom.
"""

from __future__ import annotations

import argparse

from recruit.config import settings
from recruit.data.loaders import load_hf_fit_dataset
from recruit.dropoff.predict import DropoffPredictor
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.explain.templated import build_explanation
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion


# Fusion weights for the final score.
# Tuned by inspection for the POC; sensitivity sweep is on the to-do list.
W_SEM = 0.6
W_STAY = 0.4


def _final_score(sem_similarity: float, p_dropoff: float) -> float:
    """final = w_sem * sem + w_stay * (1 - P_dropoff)."""
    return W_SEM * sem_similarity + W_STAY * (1.0 - p_dropoff)


def _short(text: str, n: int = 140) -> str:
    """Truncate text for one-line display."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, default=0,
                    help="Index of the JD to query (0..n-1)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many candidates to surface")
    ap.add_argument("--pool", type=int, default=30,
                    help="Hybrid pool size before drop-off scoring + re-fusion")
    args = ap.parse_args()

    print("Loading HF fit corpus...")
    resumes, jobs, _ = load_hf_fit_dataset(split="test")
    print(f"  {len(resumes)} résumés · {len(jobs)} JDs")

    if not 0 <= args.job < len(jobs):
        raise SystemExit(f"--job must be in 0..{len(jobs) - 1}")
    target_job = jobs[args.job]

    print("Building indexes...")
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="hf_fit_test_resumes")
    dense_idx = DenseIndex(resume_vecs)
    bm25_idx = BM25Index(resume_texts)

    print("Loading trained drop-off model...")
    predictor = DropoffPredictor(settings.models_path / "dropoff_v0.joblib")

    # ──────────────────────────────────────────────────────────────────
    # Step 1 — first-phase recall (Dense + BM25 → RRF top-pool)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n=== QUERY JOB [{args.job}]  ·  {target_job.job_id} ===")
    print(_short(target_job.description, n=240))

    qv = encoder.encode([target_job.description])[0]
    dense_hits = dense_idx.search(qv, k=args.pool)
    bm25_hits = bm25_idx.search(target_job.description, k=args.pool)
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)

    # Build a quick (resume_idx → dense / bm25 score) lookup
    dense_scores = {h.index: h.score for h in dense_hits}
    bm25_scores = {h.index: h.score for h in bm25_hits}

    pool = fused[: args.pool]

    # ──────────────────────────────────────────────────────────────────
    # Step 2 — score each pool candidate for drop-off, compose explanation
    # ──────────────────────────────────────────────────────────────────
    print(f"\nScoring {len(pool)} candidates (drop-off + explanation)...")
    scored: list[dict] = []
    for c in pool:
        resume = resumes[c.resume_idx]
        sem = dense_scores.get(c.resume_idx, 0.0)
        bm25 = bm25_scores.get(c.resume_idx, 0.0)

        prediction = predictor.predict(
            resume=resume,
            job=target_job,
            sem_similarity=sem,
            bm25_score=bm25,
        )
        explanation = build_explanation(
            resume=resume,
            job=target_job,
            prediction=prediction,
        )
        final = _final_score(sem, prediction.probability)

        scored.append({
            "resume": resume,
            "sem": sem,
            "bm25": bm25,
            "p_dropoff": prediction.probability,
            "final": final,
            "explanation": explanation,
        })

    # ──────────────────────────────────────────────────────────────────
    # Step 3 — re-rank by final fused score, print the top-K
    # ──────────────────────────────────────────────────────────────────
    scored.sort(key=lambda r: -r["final"])
    top = scored[: args.top_k]

    print(f"\n=== TOP-{args.top_k} FUSED RANKING  ·  final = {W_SEM}·sem + {W_STAY}·(1−P_dropoff) ===\n")
    for rank, row in enumerate(top, start=1):
        r = row["resume"]
        exp = row["explanation"]
        print(f"#{rank}  {r.resume_id}   "
              f"sem={row['sem']:+.3f}  "
              f"P_dropoff={row['p_dropoff']:.2f}  "
              f"final={row['final']:+.3f}  "
              f"recommend={exp.recommendation.upper().replace('_', ' ')}")
        print(f"     skills matched: "
              f"{', '.join(exp.matched_skills) if exp.matched_skills else '—'}")
        if exp.missing_skills:
            print(f"     missing       : "
                  f"{', '.join(exp.missing_skills[:5])}"
                  f"{'…' if len(exp.missing_skills) > 5 else ''}")
        print(f"     fit           : {exp.overall_fit}")
        print(f"     drop-off note : {exp.recommendation_note}")
        print()

    # ──────────────────────────────────────────────────────────────────
    # Step 4 — show one detailed candidate card (the #1 candidate)
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"DETAILED VIEW  ·  #1 candidate ({top[0]['resume'].resume_id})")
    print("=" * 70)
    print(top[0]["explanation"].render())
    print()


if __name__ == "__main__":
    main()
