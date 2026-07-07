"""Pydantic models for drop-off prediction.

WHY these schemas:
- `DropoffSample` is the contract for one (candidate, job, session) row of the
  training dataset. Every column in §4 of docs/concepts/04_dropoff_design.md
  appears here exactly once with a precise type.
- Locking the schema before writing the simulator or trainer means a column
  added in one place is reflected everywhere via type errors at import time.

VIVA: "Why a Pydantic schema rather than just a Pandas dtype dict?"
- Validation at construction catches synthesis bugs immediately (e.g. a
  negative skill_overlap) instead of later as NaN-cascade in a metric.
- The schema doubles as machine-readable documentation of every feature
  the classifier sees.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DropoffSample(BaseModel):
    """One row of the drop-off training table.

    Mirrors docs/concepts/04_dropoff_design.md §4 exactly. Column order
    follows the four families: match → candidate → job → interaction → label.
    """

    # ── Identity (not features; for traceability) ──────────────────────
    sample_id: str
    resume_id: str
    job_id: str

    # ── Match features ─────────────────────────────────────────────────
    sem_similarity: float = Field(..., ge=-1.0, le=1.0)
    bm25_score: float = Field(..., ge=0.0)
    skill_overlap: float = Field(..., ge=0.0, le=1.0)
    skill_gap: float = Field(..., ge=0.0, le=1.0)
    yoe_gap: float
    under_qualified: bool
    education_match: bool

    # ── Candidate features ─────────────────────────────────────────────
    cand_yoe: float = Field(..., ge=0.0)
    cand_n_past_roles: int = Field(..., ge=0)
    cand_avg_tenure_yrs: float = Field(..., ge=0.0)
    cand_seniority_level: int = Field(..., ge=0, le=5)
    cand_domain: str
    cand_location_country: str

    # ── Job features ───────────────────────────────────────────────────
    job_posting_age_days: int = Field(..., ge=0)
    job_n_required_fields: int = Field(..., ge=1)
    job_has_remote_option: bool
    job_seniority_level: int = Field(..., ge=0, le=5)
    job_n_applicants_so_far: int = Field(..., ge=0)
    job_salary_band: int = Field(..., ge=1, le=4)

    # ── Interaction (session) features ─────────────────────────────────
    session_time_on_page_secs: int = Field(..., ge=0)
    session_hour_of_day: int = Field(..., ge=0, le=23)
    session_is_weekend: bool
    session_device_mobile: bool
    session_n_fields_completed: int = Field(..., ge=0)
    session_n_fields_skipped: int = Field(..., ge=0)
    session_field_completion_rate: float = Field(..., ge=0.0, le=1.0)
    session_navigation_back_count: int = Field(..., ge=0)

    # ── Label ──────────────────────────────────────────────────────────
    p_dropoff_true: float = Field(..., ge=0.0, le=1.0, description="The latent probability from the simulator (kept for diagnostics, not a model feature)")
    label: int = Field(..., ge=0, le=1)

    model_config = {"frozen": True}


# The exact list of columns the model is allowed to see at training time —
# excludes identity (leakage risk) and `p_dropoff_true` (label leakage).
FEATURE_COLUMNS: list[str] = [
    "sem_similarity",
    "bm25_score",
    "skill_overlap",
    "skill_gap",
    "yoe_gap",
    "under_qualified",
    "education_match",
    "cand_yoe",
    "cand_n_past_roles",
    "cand_avg_tenure_yrs",
    "cand_seniority_level",
    "cand_domain",
    "cand_location_country",
    "job_posting_age_days",
    "job_n_required_fields",
    "job_has_remote_option",
    "job_seniority_level",
    "job_n_applicants_so_far",
    "job_salary_band",
    "session_time_on_page_secs",
    "session_hour_of_day",
    "session_is_weekend",
    "session_device_mobile",
    "session_n_fields_completed",
    "session_n_fields_skipped",
    "session_field_completion_rate",
    "session_navigation_back_count",
]


CATEGORICAL_COLUMNS: list[str] = [
    "cand_domain",
    "cand_location_country",
]


LABEL_COLUMN: str = "label"
