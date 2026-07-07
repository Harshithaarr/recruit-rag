"""Generate the synthetic drop-off dataset.

Run:  uv run python scripts/generate_dropoff_data.py

Output:
  data/processed/dropoff_v0.parquet  — all 10k rows
  data/processed/dropoff_splits.json — train/val/test row indices

Each row joins a real (résumé, JD) pair from the HF fit corpus with:
  - real match features (cosine, BM25, skill overlap) computed via the
    same matcher used in retrieval evaluation
  - simulated session telemetry (time-on-page, fields skipped, etc.)
  - a label sampled from the §3 simulation rule

The HF fit corpus is used as the source of real (résumé, JD) pairs because
it is the only labelled corpus available; the labels themselves are NOT
used here — only the texts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recruit.config import settings
from recruit.data.loaders import (
    load_hf_fit_dataset,
    load_kaggle_resumes,
    load_linkedin_jobs,
)
from recruit.data.schemas import Job
from recruit.dropoff.simulation import generate_dataset, split_dataset
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.experience import (
    Seniority,
    extract_resume_trajectory,
)
from recruit.retrieval.faiss_index import DenseIndex
from recruit.skills import extract_skills as _extract_skills


def _country_from_text(text: str) -> str:
    """Best-effort country code from common location strings."""
    low = text.lower()
    for needle, code in [
        ("india", "IN"), ("bengaluru", "IN"), ("bangalore", "IN"),
        ("hyderabad", "IN"), ("mumbai", "IN"), ("delhi", "IN"),
        ("united states", "US"), ("usa", "US"), ("california", "US"),
        ("new york", "US"), ("san francisco", "US"),
        ("united kingdom", "UK"), ("london", "UK"),
        ("germany", "DE"), ("berlin", "DE"),
        ("singapore", "SG"),
    ]:
        if needle in low:
            return code
    return "other"


def _load_resumes_for_variant(resume_source: str):
    """Return the résumé list + an SBERT cache label based on `resume_source`.

    resume_source ∈ {'hf', 'combined'}:
      hf       — HF fit corpus résumés only (477).
      combined — HF + Kaggle snehaanbhawal (~2,960).
    """
    if resume_source == "hf":
        resumes, _, _ = load_hf_fit_dataset(split="test")
        return resumes, "hf_fit_test_resumes"
    if resume_source == "combined":
        hf_r, _, _ = load_hf_fit_dataset(split="test")
        kg_r = load_kaggle_resumes(
            settings.data_path / "raw" / "resumes_kaggle" / "Resume" / "Resume.csv"
        )
        return hf_r + kg_r, "combined_hf_plus_kaggle"
    raise ValueError(f"Unknown resume_source: {resume_source}")


def _load_jobs_for_variant(jds_source: str, sample_size: int = 1500) -> list[Job]:
    """Return the JD list based on `jds_source`.

    jds_source ∈ {'hf', 'linkedin'}:
      hf       — 71 JDs from the HF fit corpus.
      linkedin — `sample_size` JDs sampled from real LinkedIn postings.
    """
    if jds_source == "hf":
        _, jobs, _ = load_hf_fit_dataset(split="test")
        return jobs
    if jds_source == "linkedin":
        return load_linkedin_jobs(
            settings.data_path / "raw" / "linkedin_jobs" / "postings.csv",
            sample_size=sample_size,
            seed=42,
        )
    raise ValueError(f"Unknown jds_source: {jds_source}")


def _build_pairs(resume_source: str, jds_source: str) -> list[dict]:
    """Produce one base record per (résumé, JD) pair.

    Each résumé is paired with each JD's top-30 retrieval matches, yielding
    a realistic candidate pool per JD. Total pairs = n_jobs × 30.
    """
    print(f"Loading résumés (source={resume_source})...")
    resumes, cache_label = _load_resumes_for_variant(resume_source)
    print(f"  {len(resumes)} résumés")

    print(f"Loading JDs (source={jds_source})...")
    jobs = _load_jobs_for_variant(jds_source)
    print(f"  {len(jobs)} JDs")

    # Pre-compute trajectory features for résumés and a coarse one for JDs.
    print("Extracting trajectory features...")
    resume_trajs = [extract_resume_trajectory(r) for r in resumes]

    # Pre-compute skill sets and country once.
    print("Extracting skill sets...")
    resume_skills = [_extract_skills(r.text) for r in resumes]
    resume_country = [_country_from_text(r.text) for r in resumes]
    job_skills = [_extract_skills(j.description) for j in jobs]

    # Pre-compute SBERT vectors and BM25 indexes.
    print("Encoding SBERT vectors (cached)...")
    encoder = SBertEncoder()
    resume_vecs = embed_with_cache(
        encoder, [r.text for r in resumes], label=cache_label
    )
    dense_idx = DenseIndex(resume_vecs)
    print("Building BM25 index...")
    bm25_idx = BM25Index([r.text for r in resumes])

    pairs: list[dict] = []
    print("Building pair records (one per JD, top-30 résumés)...")
    for j_idx, job in enumerate(jobs):
        qv = encoder.encode([job.description])[0]
        dense_hits = dense_idx.search(qv, k=30)
        bm25_lookup = {
            h.index: h.score for h in bm25_idx.search(job.description, k=200)
        }

        # JD seniority (coarse) — re-use the experience extractor heuristic.
        # Quick inline lookup rather than another import shuffle.
        job_low = (job.title + " " + job.description).lower()
        if any(k in job_low for k in ("staff engineer", "principal", "tech lead")):
            job_sen = int(Seniority.STAFF.value)
        elif any(k in job_low for k in ("senior", "sr.")):
            job_sen = int(Seniority.SENIOR.value)
        elif any(k in job_low for k in ("director", "head of", "vp ", "cto")):
            job_sen = int(Seniority.DIRECTOR.value)
        elif any(k in job_low for k in ("junior", "intern", "graduate", "entry")):
            job_sen = int(Seniority.JUNIOR.value)
        else:
            job_sen = int(Seniority.MID.value)

        job_min_yoe = float(job.min_years_experience or 2.0)

        for h in dense_hits:
            r_idx = h.index
            resume = resumes[r_idx]
            traj = resume_trajs[r_idx]
            cand_yoe = float(traj.years_experience) if traj.years_experience is not None else 3.0
            pairs.append({
                "resume_id": resume.resume_id,
                "job_id": job.job_id,
                "cand_skills": resume_skills[r_idx],
                "job_skills": job_skills[j_idx],
                "cand_yoe": cand_yoe,
                "job_min_yoe": job_min_yoe,
                "sem_similarity": float(h.score),
                "bm25_score": float(bm25_lookup.get(r_idx, 0.0)),
                "cand_n_past_roles": int(max(1, traj.role_count)),
                "cand_seniority_level": int(traj.seniority.value),
                "cand_domain": str(traj.domain),
                "cand_location_country": resume_country[r_idx],
                "job_seniority_level": job_sen,
            })

    print(f"  built {len(pairs)} base pairs")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0",
                    help="Output filename suffix (e.g. v0, v1).")
    ap.add_argument("--resumes", choices=["hf", "combined"], default="hf",
                    help="Résumé source: hf=HF only (~477), combined=HF+Kaggle (~2,960)")
    ap.add_argument("--jds", choices=["hf", "linkedin"], default="hf",
                    help="JD source: hf=71 fit-corpus JDs, linkedin=real LinkedIn postings")
    ap.add_argument("--n-rows", type=int, default=10_000,
                    help="Number of synthetic samples to generate.")
    ap.add_argument("--target-rate", type=float, default=None,
                    help="If set, autocalibrate intercept to hit this positive rate "
                         "(e.g. 0.25 for HR-analytics-like; 0.55 for outline §3 default).")
    args = ap.parse_args()

    pairs = _build_pairs(resume_source=args.resumes, jds_source=args.jds)

    print(f"\nGenerating synthetic drop-off dataset (n={args.n_rows}, "
          f"variant={args.variant}, target_rate={args.target_rate})...")
    df = generate_dataset(
        pairs,
        n_rows=args.n_rows,
        seed=42,
        target_rate=args.target_rate,
    )
    print(f"  rows: {len(df)}")
    print(f"  positive class share: {df['label'].mean():.3f}")
    print(f"  feature columns: {df.shape[1]} (incl. identity + label)")

    out_dir = settings.data_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dropoff_{args.variant}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size // 1024} KB)")

    train_df, val_df, test_df = split_dataset(df, train_frac=0.70, val_frac=0.15, seed=42)
    print(f"  train: {len(train_df)}   val: {len(val_df)}   test: {len(test_df)}")
    print(f"  pos-rate train/val/test: "
          f"{train_df['label'].mean():.3f} / "
          f"{val_df['label'].mean():.3f} / "
          f"{test_df['label'].mean():.3f}")

    splits_path = out_dir / f"dropoff_splits_{args.variant}.json"
    splits_path.write_text(json.dumps({
        "train": train_df["sample_id"].tolist(),
        "val":   val_df["sample_id"].tolist(),
        "test":  test_df["sample_id"].tolist(),
        "seed":  42,
        "weights_variant": args.variant,
        "resume_source": args.resumes,
        "jds_source": args.jds,
        "target_rate": args.target_rate,
    }, indent=2))
    print(f"Wrote {splits_path}")


if __name__ == "__main__":
    main()
