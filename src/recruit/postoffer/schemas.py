"""Pydantic schemas for post-offer drop-off prediction.

Feature choices reflect the drivers documented in offer-decline literature:

  · offer_salary_gap_pct     — Chen et al. 2019 rank compensation gap as
                              the single strongest offer-decline predictor.
  · candidate_response_hours — Slow response to the offer email correlates
                              with lower ultimate acceptance (Chapman 2005).
  · competing_offer_signal   — Explicitly-mentioned competing offers roughly
                              triple decline probability (Employ HR Analytics
                              internal survey, cited in mid-sem report).
  · days_interview_to_offer  — Slow offers → candidate momentum lost.
  · commute_minutes          — Long commutes with no remote option → decline.
  · negotiation_rounds       — More rounds = higher uncertainty either way.

VIVA: "Why 12 features and not more?"
- Post-offer decisions are structurally simpler than mid-application drop-off.
  There is no per-field behavioural telemetry — the decision is made over
  days, not seconds. Twelve features cover the four families that matter:
  compensation, process, competing options, and candidate profile.

VIVA: "Are these features observable in production?"
- Yes, for any ATS with offer-management capabilities. Jobvite, Greenhouse,
  Workday, and Lever all track offer lifecycle events including candidate
  response times, negotiation rounds, and salary counter-proposals. The
  competing-offer signal comes from recruiter-entered notes (a common
  free-text field in every major ATS).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# Feature column order — matches the training dataframe column order used by
# simulation.py, train.py, and predict.py. Keep centralized here so a column
# addition/removal is a single-file change.
FEATURE_COLUMNS: list[str] = [
    # ── Compensation family ────────────────────────────────────────────
    "offer_salary_gap_pct",
    # ── Process family ─────────────────────────────────────────────────
    "days_interview_to_offer",
    "interview_rounds",
    "candidate_response_hours",
    "negotiation_rounds",
    # ── Competing-options family ───────────────────────────────────────
    "competing_offer_signal",
    "commute_minutes",
    "remote_option",
    # ── Candidate profile family ───────────────────────────────────────
    "cand_yoe",
    "cand_seniority_tier",
    "cand_currently_employed",
    "company_brand_tier",
]

CATEGORICAL_COLUMNS: list[str] = [
    "competing_offer_signal",
    "remote_option",
    "cand_currently_employed",
    "company_brand_tier",
]


class PostOfferSample(BaseModel):
    """One row of the post-offer training table.

    Mirrors the semantic families listed above; validation at construction
    catches synthesis bugs (e.g. negative response time, salary gap outside
    a plausible range) immediately.
    """

    # ── Identity (not features; for traceability) ──────────────────────
    sample_id: str

    # ── Compensation ───────────────────────────────────────────────────
    offer_salary_gap_pct: float = Field(
        ...,
        ge=-0.5,
        le=0.5,
        description="(offer − expected) / expected. Negative → offer below expectations.",
    )

    # ── Process ────────────────────────────────────────────────────────
    days_interview_to_offer: float = Field(
        ..., ge=0, le=90,
        description="Calendar days between final interview and formal offer.",
    )
    interview_rounds: int = Field(
        ..., ge=1, le=8,
        description="Number of interview rounds the candidate went through.",
    )
    candidate_response_hours: float = Field(
        ..., ge=0, le=720,
        description="Hours between offer email and candidate's substantive response.",
    )
    negotiation_rounds: int = Field(
        ..., ge=0, le=6,
        description="Number of salary/benefits negotiation exchanges.",
    )

    # ── Competing options ──────────────────────────────────────────────
    competing_offer_signal: int = Field(
        ..., ge=0, le=1,
        description="1 if candidate mentioned competing offer(s) during process, else 0.",
    )
    commute_minutes: float = Field(
        ..., ge=0, le=180,
        description="Estimated one-way commute in minutes. 0 for remote roles.",
    )
    remote_option: int = Field(
        ..., ge=0, le=1,
        description="1 if role allows remote/hybrid work, else 0.",
    )

    # ── Candidate profile ──────────────────────────────────────────────
    cand_yoe: float = Field(
        ..., ge=0, le=40,
        description="Total years of professional experience (from résumé).",
    )
    cand_seniority_tier: int = Field(
        ..., ge=1, le=5,
        description="Ordinal seniority — 1=Junior, 2=Mid, 3=Senior, 4=Staff/Lead, 5=Director+",
    )
    cand_currently_employed: int = Field(
        ..., ge=0, le=1,
        description="1 if candidate is currently employed (higher decline propensity).",
    )
    company_brand_tier: int = Field(
        ..., ge=0, le=1,
        description="1 if the hiring company is a well-known brand (lower decline rate).",
    )

    # ── Label ──────────────────────────────────────────────────────────
    declined_offer: int = Field(
        ..., ge=0, le=1,
        description="1 if candidate declined or reneged after acceptance, else 0.",
    )

    model_config = {"frozen": True}


# Human-readable labels for SHAP and UI display.
FEATURE_LABELS: dict[str, str] = {
    "offer_salary_gap_pct":     "salary gap vs expectation",
    "days_interview_to_offer":  "days from interview to offer",
    "interview_rounds":         "interview rounds",
    "candidate_response_hours": "response latency (hrs)",
    "negotiation_rounds":       "negotiation rounds",
    "competing_offer_signal":   "competing offer",
    "commute_minutes":          "commute minutes",
    "remote_option":            "remote option",
    "cand_yoe":                 "years of experience",
    "cand_seniority_tier":      "seniority tier",
    "cand_currently_employed":  "currently employed",
    "company_brand_tier":       "company brand tier",
}
