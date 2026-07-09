"""Serving-time wrapper for the post-offer decline classifier.

WHY this layer:
- Training scripts operate on a DataFrame of many rows. At serving time
  (the Streamlit UI, or a REST API in future) the caller has a single
  offer situation and wants back a probability + a SHAP explanation.
- This module is the bridge — accepts a typed `OfferSituation` (or a
  plain dict of features), returns a packaged `PostOfferPrediction`.

VIVA: "Where does the feature vector come from in production?"
- In a production ATS integration the offer-lifecycle events (salary
  proposal, days-to-offer, response latency, negotiation rounds,
  competing-offer notes) are emitted directly by the ATS. My module
  simply consumes them. Nothing about the framework changes — only
  the source of the feature values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from recruit.dropoff.explain import LocalExplanation
from recruit.postoffer.explain import PostOfferExplainer
from recruit.postoffer.schemas import FEATURE_COLUMNS


# ─── Domain object: one offer situation ─────────────────────────────────


@dataclass
class OfferSituation:
    """One candidate's offer state — the input to the post-offer predictor.

    All fields mirror the feature schema in `postoffer/schemas.py`. Defaults
    reflect a mid-market "typical" offer so callers can override only the
    fields they care about (useful for the UI's scenario buttons).
    """

    # Compensation
    offer_salary_gap_pct: float = 0.0

    # Process
    days_interview_to_offer: float = 7.0
    interview_rounds: int = 3
    candidate_response_hours: float = 24.0
    negotiation_rounds: int = 1

    # Competing options
    competing_offer_signal: int = 0
    commute_minutes: float = 30.0
    remote_option: int = 0

    # Candidate profile
    cand_yoe: float = 7.0
    cand_seniority_tier: int = 2
    cand_currently_employed: int = 1
    company_brand_tier: int = 0

    def to_feature_row(self) -> dict:
        """Feature-only dict — matches FEATURE_COLUMNS order."""
        return {col: getattr(self, col) for col in FEATURE_COLUMNS}


# ─── Prediction output ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PostOfferPrediction:
    """End-to-end post-offer prediction packaged for the UI / RAG prompt.

    `feature_row` is what the model actually scored — retained so the UI
    can render the SHAP waterfall without re-running feature extraction.
    """

    probability: float
    risk_band: str  # "low" | "moderate" | "high"
    explanation: LocalExplanation
    feature_row: dict

    def shap_context_text(self, top_k: int = 3) -> str:
        """Textualised SHAP context — feeds the RAG explainer prompt."""
        # Wrap the SHAP context in a post-offer-specific header so the
        # LLM doesn't confuse this with a mid-application prediction.
        base = self.explanation.textualize(top_k=top_k)
        return base.replace(
            "Drop-off risk:", "Post-offer decline risk:"
        ).replace("↑ drop-off", "↑ decline").replace(
            "↓ drop-off", "↓ decline"
        )


def _risk_band(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "moderate"
    return "high"


# ─── Predictor class ────────────────────────────────────────────────────


class PostOfferPredictor:
    """Serving-time predictor for post-offer decline probability.

    Loads the joblib pipeline (calibrated by default) and the SHAP
    explainer once at construction. `predict()` accepts an OfferSituation
    (or dict) and returns a packaged PostOfferPrediction.
    """

    def __init__(self, model_path: Path) -> None:
        self._pipeline = joblib.load(model_path)
        self._explainer = PostOfferExplainer(self._pipeline)

    def predict(
        self,
        situation: OfferSituation | dict,
    ) -> PostOfferPrediction:
        """Predict decline probability for one offer situation."""
        if isinstance(situation, OfferSituation):
            feature_row = situation.to_feature_row()
        else:
            feature_row = {col: situation[col] for col in FEATURE_COLUMNS}

        prob = float(
            self._pipeline.predict_proba(
                pd.DataFrame([feature_row])[FEATURE_COLUMNS]
            )[0, 1]
        )
        explanation = self._explainer.explain_row(
            feature_row,
            sample_id="post-offer",
            top_k=5,
        )
        return PostOfferPrediction(
            probability=prob,
            risk_band=_risk_band(prob),
            explanation=explanation,
            feature_row=feature_row,
        )


# ─── Preset scenarios — used by the UI + CLI smoke tests ────────────────


OFFER_SCENARIOS: dict[str, OfferSituation] = {
    "likely_accept": OfferSituation(
        # Strong, well-matched offer — everything the candidate wants.
        offer_salary_gap_pct=0.08,        # 8 % above expectation
        days_interview_to_offer=3.0,      # fast turnaround
        interview_rounds=3,
        candidate_response_hours=4.0,     # responded within 4 hours
        negotiation_rounds=0,
        competing_offer_signal=0,
        commute_minutes=15.0,
        remote_option=1,                  # remote available
        cand_yoe=5.0,
        cand_seniority_tier=2,
        cand_currently_employed=0,        # actively seeking
        company_brand_tier=1,             # strong brand
    ),
    "uncertain": OfferSituation(
        # Reasonable offer, some friction.
        offer_salary_gap_pct=-0.02,       # slightly below expectation
        days_interview_to_offer=10.0,
        interview_rounds=4,
        candidate_response_hours=36.0,    # took a day and a half
        negotiation_rounds=2,
        competing_offer_signal=0,
        commute_minutes=45.0,
        remote_option=0,
        cand_yoe=8.0,
        cand_seniority_tier=3,
        cand_currently_employed=1,        # currently employed, less pressure
        company_brand_tier=0,
    ),
    "likely_decline": OfferSituation(
        # Weak offer + candidate has options + long commute.
        offer_salary_gap_pct=-0.15,       # 15 % below expectation
        days_interview_to_offer=21.0,     # slow process
        interview_rounds=5,
        candidate_response_hours=120.0,   # 5 days
        negotiation_rounds=4,
        competing_offer_signal=1,         # mentioned competing offer
        commute_minutes=90.0,             # 1.5 h each way
        remote_option=0,
        cand_yoe=12.0,
        cand_seniority_tier=4,            # staff / lead — has options
        cand_currently_employed=1,
        company_brand_tier=0,
    ),
}


SCENARIO_LABELS: dict[str, str] = {
    "likely_accept":  "🟢 Likely to accept",
    "uncertain":      "🟡 Uncertain — reasonable but flawed",
    "likely_decline": "🔴 Likely to decline",
}
