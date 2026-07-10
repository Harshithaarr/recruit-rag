"""Serving-time wrapper for the drop-off classifier.

WHY this layer:
- The training script handles a DataFrame of N rows. At UI serving time the
  caller has a (Resume, Job, simulated_session) triple and wants one
  probability + one explanation.
- This module is the bridge between the high-level domain objects and the
  flat feature row the model expects.
- It also packages the SHAP→RAG bridge: per-prediction `LocalExplanation`
  ready to be textualised into the RAG explainer prompt.

VIVA: "Where does the session telemetry come from at serving time?"
- In a production ATS it streams from the application form (time-on-page,
  fields completed, navigation). For the POC we simulate a session using
  the same generator as training. The simulator is parameterised so a
  Streamlit slider can override individual session fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recruit.data.schemas import Job, Resume
from recruit.dropoff.explain import DropoffExplainer, LocalExplanation
from recruit.dropoff.schemas import FEATURE_COLUMNS
from recruit.dropoff.simulation import _draw_session_features
from recruit.retrieval.experience import (
    Seniority,
    extract_resume_trajectory,
)
from recruit.skills import extract_skills


@dataclass(frozen=True)
class DropoffPrediction:
    """One end-to-end prediction packaged for the UI / fusion ranker.

    `feature_row` is the exact dict the model scored — kept so the UI can
    regenerate per-prediction artefacts (e.g. the SHAP waterfall) without
    re-running feature extraction.

    ABSTENTION FIELDS (end-sem addition)
    -----------------------------------
    Because our probabilities are isotonic-calibrated, we can turn the
    calibration into a **trust feature**: report how confident the model
    is in its own prediction. Confidence is high when the probability is
    far from 0.5 (the decision boundary); low when it's near 0.5.

    · `confidence`      float in [0, 1] — 1.0 = perfectly certain,
                        0.0 = maximum uncertainty (p = 0.5)
    · `abstention_band` str "confident" | "uncertain" | "abstain"
                        — the interpretable version, for UI badges

    Recommended UI behaviour: when the band is `"abstain"`, present the
    recommendation as "route to human review" rather than trusting the
    model's binary classification. Follows the selective-prediction
    literature (El-Yaniv & Wiener 2010, Geifman & El-Yaniv 2017).
    """

    resume_id: str
    job_id: str
    probability: float
    risk_band: str  # "low" | "moderate" | "high"
    confidence: float  # ∈ [0, 1] — 1 = certain, 0 = maximum entropy at p=0.5
    abstention_band: str  # "confident" | "uncertain" | "abstain"
    explanation: LocalExplanation
    feature_row: dict

    def shap_context_text(self, top_k: int = 3) -> str:
        """Textualised SHAP for the RAG prompt — the SHAP→RAG bridge."""
        return self.explanation.textualize(top_k=top_k)


def _risk_band(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "moderate"
    return "high"


def _confidence(p: float) -> float:
    """Confidence = 1 − normalised Bernoulli entropy.

    Bernoulli entropy H(p) = -p·log(p) - (1-p)·log(1-p).
    Max at p=0.5 (H = log 2 ≈ 0.693), min at p=0 or p=1 (H = 0).
    We normalise to [0, 1] and flip so 1 = confident, 0 = uncertain.
    Guard against log(0) with a small epsilon.
    """
    import math
    eps = 1e-9
    p_c = min(max(p, eps), 1.0 - eps)
    h = -p_c * math.log(p_c) - (1.0 - p_c) * math.log(1.0 - p_c)
    normalized_entropy = h / math.log(2.0)  # ∈ [0, 1]
    return float(1.0 - normalized_entropy)


def _abstention_band(confidence: float) -> str:
    """Bucket confidence into interpretable bands for the UI.

    Thresholds tuned so that the vast majority of predictions land in
    `confident` (i.e. the model is trusted to decide), and only near-
    50-50 predictions get flagged for human review. Concrete boundaries:

      · `confident`  p < 0.30 or p > 0.70   → confidence ≳ 0.12
      · `uncertain`  p ∈ [0.30, 0.40] ∪ [0.60, 0.70]  → confidence 0.03-0.12
      · `abstain`    p ∈ [0.40, 0.60]                  → confidence < 0.03

    An `abstain` verdict is a *feature*, not a failure: the model is
    telling the recruiter "I would coin-flip on this one — please
    review manually." Follows the selective-prediction literature.
    """
    if confidence >= 0.12:
        return "confident"
    if confidence >= 0.03:
        return "uncertain"
    return "abstain"


class DropoffPredictor:
    """Serving-time predictor.

    Loads the joblib pipeline and SHAP explainer once at construction;
    `predict()` accepts domain objects and returns a packaged result.
    """

    def __init__(self, model_path: Path, *, seed: int = 42) -> None:
        self._pipeline = joblib.load(model_path)
        self._explainer = DropoffExplainer(self._pipeline)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        resume: Resume,
        job: Job,
        sem_similarity: float,
        bm25_score: float,
        session_overrides: dict | None = None,
    ) -> DropoffPrediction:
        """Predict drop-off for one (Resume, Job) pair.

        `sem_similarity` and `bm25_score` come from the retrieval layer.
        `session_overrides` lets the caller pin specific session features
        (e.g. from a Streamlit slider); unspecified fields are simulated.
        """
        feature_row = self._build_feature_row(
            resume=resume,
            job=job,
            sem_similarity=sem_similarity,
            bm25_score=bm25_score,
            session_overrides=session_overrides,
        )
        prob = float(
            self._pipeline.predict_proba(
                pd.DataFrame([feature_row])[FEATURE_COLUMNS]
            )[0, 1]
        )

        explanation = self._explainer.explain_row(
            feature_row,
            sample_id=f"{resume.resume_id}::{job.job_id}",
            top_k=5,
        )

        confidence = _confidence(prob)
        return DropoffPrediction(
            resume_id=resume.resume_id,
            job_id=job.job_id,
            probability=prob,
            risk_band=_risk_band(prob),
            confidence=confidence,
            abstention_band=_abstention_band(confidence),
            explanation=explanation,
            feature_row=feature_row,
        )

    @property
    def explainer(self) -> DropoffExplainer:
        """Expose the SHAP explainer for callers needing extra views."""
        return self._explainer

    # ------------------------------------------------------------------
    # Feature assembly — the domain-to-flat-row glue
    # ------------------------------------------------------------------

    def _build_feature_row(
        self,
        *,
        resume: Resume,
        job: Job,
        sem_similarity: float,
        bm25_score: float,
        session_overrides: dict | None = None,
    ) -> dict:
        """Assemble one feature row from domain objects."""
        traj = extract_resume_trajectory(resume)

        cand_yoe = float(traj.years_experience) if traj.years_experience is not None else 3.0
        cand_n_past_roles = max(1, int(traj.role_count))
        cand_avg_tenure_yrs = cand_yoe / cand_n_past_roles

        # Match features — derived from text and structured fields.
        # Fall back to vocabulary-based extraction when structured skill lists
        # are empty (e.g. HF dataset has only raw text). Same vocabulary as
        # the data generator and templated explainer — keeps the model's view
        # of skill_overlap consistent across pipeline stages.
        cand_skill_set = (
            {s.lower() for s in resume.skills}
            if resume.skills else extract_skills(resume.text)
        )
        job_skill_set = (
            {s.lower() for s in job.required_skills}
            if job.required_skills else extract_skills(job.description)
        )
        overlap = (
            len(cand_skill_set & job_skill_set) / len(job_skill_set)
            if job_skill_set else 0.0
        )
        job_min_yoe = float(job.min_years_experience or 2.0)
        yoe_gap = cand_yoe - job_min_yoe

        # Coarse JD seniority (re-uses the same heuristic as the data generator).
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

        # Simulate session, then override with anything the caller pinned.
        session = _draw_session_features(self._rng)
        if session_overrides:
            session.update(session_overrides)

        # Country — best-effort, default "other" if unavailable.
        country = "other"
        if resume.location:
            low = resume.location.lower()
            for needle, code in [
                ("india", "IN"), ("bengaluru", "IN"), ("bangalore", "IN"),
                ("united states", "US"), ("usa", "US"),
                ("united kingdom", "UK"), ("london", "UK"),
                ("germany", "DE"), ("singapore", "SG"),
            ]:
                if needle in low:
                    country = code
                    break

        return {
            # Match features
            "sem_similarity": float(sem_similarity),
            "bm25_score": float(bm25_score),
            "skill_overlap": float(overlap),
            "skill_gap": float(1.0 - overlap),
            "yoe_gap": float(yoe_gap),
            "under_qualified": bool(yoe_gap < 0),
            "education_match": True,
            # Candidate
            "cand_yoe": cand_yoe,
            "cand_n_past_roles": cand_n_past_roles,
            "cand_avg_tenure_yrs": cand_avg_tenure_yrs,
            "cand_seniority_level": int(traj.seniority.value),
            "cand_domain": traj.domain,
            "cand_location_country": country,
            # Job
            "job_posting_age_days": int(self._rng.poisson(lam=18)),
            "job_n_required_fields": session["job_n_required_fields"],
            "job_has_remote_option": "remote" in job_low,
            "job_seniority_level": job_sen,
            "job_n_applicants_so_far": int(self._rng.poisson(lam=80)),
            "job_salary_band": int(self._rng.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.15])),
            # Session
            "session_time_on_page_secs": session["session_time_on_page_secs"],
            "session_hour_of_day": session["session_hour_of_day"],
            "session_is_weekend": session["session_is_weekend"],
            "session_device_mobile": session["session_device_mobile"],
            "session_n_fields_completed": session["session_n_fields_completed"],
            "session_n_fields_skipped": session["session_n_fields_skipped"],
            "session_field_completion_rate": session["session_field_completion_rate"],
            "session_navigation_back_count": session["session_navigation_back_count"],
        }
