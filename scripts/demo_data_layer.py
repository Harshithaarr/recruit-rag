"""Smoke-test the data layer.

Run:  uv run python scripts/demo_data_layer.py

What this does:
- Loads the tiny committed sample CSVs.
- Validates each row against the Resume / Job schemas.
- Prints a few records and a one-line summary.

Why it matters in the thesis:
- This is your "data layer contract" — every downstream component (matcher,
  predictor, RAG) consumes Resume / Job objects, not raw CSVs.
"""

from __future__ import annotations

from recruit.config import settings
from recruit.data.loaders import load_jobs_csv, load_resumes_csv


def main() -> None:
    sample_dir = settings.data_path / "sample"
    resumes = load_resumes_csv(sample_dir / "resumes.csv")
    jobs = load_jobs_csv(sample_dir / "jobs.csv")

    print(f"Loaded {len(resumes)} résumés and {len(jobs)} jobs from {sample_dir}")
    print()
    print("--- First résumé ---")
    r = resumes[0]
    print(f"  id        : {r.resume_id}")
    print(f"  name      : {r.candidate_name}")
    print(f"  years exp : {r.years_experience}")
    print(f"  location  : {r.location}")
    print(f"  skills    : {r.skills}")
    print(f"  text[:120]: {r.text[:120]}...")
    print()
    print("--- First job ---")
    j = jobs[0]
    print(f"  id              : {j.job_id}")
    print(f"  title           : {j.title}")
    print(f"  company         : {j.company}")
    print(f"  min years exp   : {j.min_years_experience}")
    print(f"  required skills : {j.required_skills}")
    print(f"  description[:120]: {j.description[:120]}...")


if __name__ == "__main__":
    main()
