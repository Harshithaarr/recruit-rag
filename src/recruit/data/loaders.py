"""Loaders convert raw CSV/JSON/HF datasets into validated `Resume` and `Job` objects.

WHY this layer:
- Real datasets have inconsistent column names, missing fields, encoding issues.
- The loader is the *only* place that knows about those raw formats. Everything
  downstream consumes typed `Resume` / `Job` objects.
- Swapping data sources (Kaggle → HF → O*NET → real ATS export) means rewriting
  only this file, not the matcher, predictor, or UI.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from recruit.data.schemas import Job, Resume


def _split_skills(raw: object) -> list[str]:
    """Best-effort parser for a skills column that may be a list, comma-string, or NaN."""
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def load_resumes_csv(path: Path | str) -> list[Resume]:
    """Load résumés from a CSV.

    Expected columns: resume_id, candidate_name, text, skills, years_experience,
                      location, education, target_role
    Missing optional columns are tolerated.
    """
    df = pd.read_csv(path)
    resumes: list[Resume] = []
    for _, row in df.iterrows():
        resumes.append(
            Resume(
                resume_id=str(row["resume_id"]),
                candidate_name=row.get("candidate_name"),
                text=str(row["text"]),
                skills=_split_skills(row.get("skills")),
                years_experience=row.get("years_experience"),
                location=row.get("location"),
                education=row.get("education"),
                target_role=row.get("target_role"),
            )
        )
    return resumes


def load_jobs_csv(path: Path | str) -> list[Job]:
    """Load job postings from a CSV.

    Expected columns: job_id, title, company, description, required_skills,
                      min_years_experience, location, required_certifications
    """
    df = pd.read_csv(path)
    jobs: list[Job] = []
    for _, row in df.iterrows():
        jobs.append(
            Job(
                job_id=str(row["job_id"]),
                title=str(row["title"]),
                company=row.get("company"),
                description=str(row["description"]),
                required_skills=_split_skills(row.get("required_skills")),
                min_years_experience=row.get("min_years_experience"),
                location=row.get("location"),
                required_certifications=_split_skills(row.get("required_certifications")),
            )
        )
    return jobs


# HuggingFace 'cnamuangtoun/resume-job-description-fit' — labelled (résumé, JD) pairs.
# Used for proper IR evaluation: each unique JD becomes one query whose qrels are
# the labelled résumés (Good Fit=2, Potential Fit=1, No Fit omitted).
_HF_FIT_DATASET = "cnamuangtoun/resume-job-description-fit"
_HF_LABEL_TO_GRADE = {"Good Fit": 2.0, "Potential Fit": 1.0, "No Fit": 0.0}


def _stable_id(prefix: str, text: str) -> str:
    """Deterministic short ID derived from text content.

    Stable across runs so qrels and the corpus index agree on identities even
    when the HF dataset is re-downloaded.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def load_hf_fit_dataset(
    split: str = "test",
) -> tuple[list[Resume], list[Job], dict[str, dict[str, float]]]:
    """Load the HF labelled (résumé, JD) fit dataset for IR evaluation.

    Each row in the dataset is a (résumé_text, JD_text, label) triple where
    label ∈ {Good Fit, Potential Fit, No Fit}. We:
    - Deduplicate résumés and JDs (one row per unique text).
    - Assign stable IDs by content hash so qrels survive re-downloads.
    - Build qrels mapping job_id → {resume_id → grade}; "No Fit" entries are
      stored with grade 0.0 (informative explicit negatives — many IR datasets
      omit negatives entirely, which is weaker).

    Returns:
      resumes: deduplicated `Resume` objects (the corpus to encode).
      jobs:    deduplicated `Job` objects (potential queries).
      qrels:   dict[job_id][resume_id] -> graded relevance (0/1/2).
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_FIT_DATASET, split=split)

    resume_by_id: dict[str, Resume] = {}
    job_by_id: dict[str, Job] = {}
    qrels: dict[str, dict[str, float]] = {}

    for row in ds:
        resume_text = str(row["resume_text"]).strip()
        job_text = str(row["job_description_text"]).strip()
        label = str(row["label"]).strip()
        if not resume_text or not job_text or label not in _HF_LABEL_TO_GRADE:
            continue

        resume_id = _stable_id("hfr", resume_text)
        job_id = _stable_id("hfj", job_text)

        if resume_id not in resume_by_id:
            resume_by_id[resume_id] = Resume(
                resume_id=resume_id,
                text=resume_text,
                # The HF dataset has only raw text — no parsed skills /
                # location / YOE. Downstream parsing (Step 4 in the plan)
                # will populate these. For pure semantic-retrieval eval, the
                # raw text is what SBERT and BM25 consume, so this is fine.
            )
        if job_id not in job_by_id:
            # No title provided in this dataset — synthesize a placeholder.
            # Title is only used in UI; eval uses `description` (=text).
            job_by_id[job_id] = Job(
                job_id=job_id,
                title=f"HF-Fit Query {job_id[-6:]}",
                description=job_text,
            )

        qrels.setdefault(job_id, {})[resume_id] = _HF_LABEL_TO_GRADE[label]

    resumes = list(resume_by_id.values())
    jobs = list(job_by_id.values())
    return resumes, jobs, qrels


# ─────────────────────────────────────────────────────────────────────────
# Kaggle snehaanbhawal/resume-dataset
#   CSV columns: ID, Resume_str, Resume_html, Category
#   ~2,400 résumés across 24 categories. The HTML column is discarded —
#   Resume_str is the plain text that matters.
# ─────────────────────────────────────────────────────────────────────────


def load_kaggle_resumes(
    csv_path: Path | str,
    *,
    categories: list[str] | None = None,
    max_rows: int | None = None,
) -> list[Resume]:
    """Load résumés from the Kaggle snehaanbhawal/resume-dataset CSV.

    `categories` (optional) filters to a subset, e.g.
        ['INFORMATION-TECHNOLOGY', 'BANKING'] for a tech+finance corpus.
    `max_rows` (optional) caps the number of rows loaded for dev iteration.
    """
    df = pd.read_csv(csv_path)
    if categories:
        df = df[df["Category"].isin(categories)]
    if max_rows:
        df = df.head(max_rows)

    resumes: list[Resume] = []
    for _, row in df.iterrows():
        text = str(row["Resume_str"]).strip()
        if not text:
            continue
        resumes.append(
            Resume(
                resume_id=f"kgr-{int(row['ID'])}",
                text=text,
                # `target_role` holds the Kaggle category — useful as a coarse
                # domain tag for stratified splits and fairness slicing.
                target_role=str(row["Category"]),
            )
        )
    return resumes


# ─────────────────────────────────────────────────────────────────────────
# Kaggle arshkon/linkedin-job-postings
#   postings.csv has 3.4M rows; we subsample by default. Schema columns
#   actually used: job_id, title, company_name, description, location,
#   formatted_experience_level, skills_desc, remote_allowed.
# ─────────────────────────────────────────────────────────────────────────


_EXP_LEVEL_TO_MIN_YOE: dict[str, float] = {
    "Internship": 0.0,
    "Entry level": 0.0,
    "Associate": 1.0,
    "Mid-Senior level": 4.0,
    "Director": 8.0,
    "Executive": 10.0,
}


def load_linkedin_jobs(
    csv_path: Path | str,
    *,
    sample_size: int | None = 5000,
    seed: int = 42,
    require_description: bool = True,
) -> list[Job]:
    """Load LinkedIn job postings with optional random subsampling.

    Default is a 5,000-row random sample so SBERT encoding finishes in
    minutes on CPU; pass `sample_size=None` for the full corpus (~3.4M rows,
    do not encode the full set unless you have hours to spare).

    `require_description` drops rows with missing/empty descriptions before
    sampling, since they're unusable for retrieval.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if require_description:
        df = df[df["description"].notna() & (df["description"].str.strip().str.len() > 50)]

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    jobs: list[Job] = []
    for _, row in df.iterrows():
        description = str(row["description"]).strip()
        if not description:
            continue

        # Parse experience level → min YOE; default to 2 if missing/unknown.
        exp_level = str(row.get("formatted_experience_level") or "").strip()
        min_yoe = _EXP_LEVEL_TO_MIN_YOE.get(exp_level, 2.0)

        # skills_desc is a free-text field, not a structured list — parse
        # comma/semicolon-separated tokens, drop anything > 40 chars (probably
        # a sentence rather than a skill).
        skills_raw = str(row.get("skills_desc") or "")
        skills = [
            s.strip() for s in re.split(r"[,;\n]", skills_raw)
            if 0 < len(s.strip()) <= 40
        ]

        jobs.append(
            Job(
                job_id=f"lij-{int(row['job_id'])}",
                title=str(row.get("title") or "").strip() or "Untitled",
                company=str(row.get("company_name") or "").strip() or None,
                description=description,
                required_skills=skills,
                min_years_experience=min_yoe,
                location=str(row.get("location") or "").strip() or None,
            )
        )
    return jobs


# ─────────────────────────────────────────────────────────────────────────
# Kaggle arashnic/hr-analytics-job-change-of-data-scientists
#   aug_train.csv columns: enrollee_id, city, city_development_index, gender,
#   relevent_experience, enrolled_university, education_level, major_discipline,
#   experience, company_size, company_type, last_new_job, training_hours, target
#
#   `target` = 1 means "looking for a job change" — a real-world proxy for
#   the synthetic drop-off label, useful for v2 sensitivity validation.
#   Importantly, `gender` is a real column here (not inferred), so this
#   dataset enables a non-proxy fairness audit.
# ─────────────────────────────────────────────────────────────────────────


def load_hr_analytics(csv_path: Path | str) -> pd.DataFrame:
    """Load the HR analytics dataset as a DataFrame.

    Returns the raw frame; downstream code chooses which columns to use.
    No schema mapping to Resume/Job because this dataset is per-enrollee,
    not per-(résumé, JD) pair. Used for:
    - Real-gender fairness validation of the drop-off classifier
    - Sensitivity comparison: real change-intent labels vs synthetic drop-off
    """
    return pd.read_csv(csv_path)


