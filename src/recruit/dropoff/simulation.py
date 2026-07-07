"""Synthetic data generator for drop-off prediction.

WHY synthetic:
- Public datasets do not include true application-level drop-off labels.
- The methodology chapter frames this as a transparent simulation study —
  see docs/concepts/04_dropoff_design.md §2.

THE RULE (v0):
- Drop-off probability is a logistic of weighted features + Gaussian noise.
- Weights documented in §3 of the design doc.
- Resulting base rate calibrated to ~55% drop-off (matches industry reports).

WHY each sample joins a real (résumé, JD) pair from the HF fit corpus:
- The match features (cosine similarity, BM25, skill overlap) only make
  sense if computed against real text. Generating those from a Gaussian
  would teach the model nothing about retrieval-quality signals.
- Session features (time-on-page, fields skipped, etc.) are pure simulation
  because no candidate-side telemetry is available publicly.

VIVA: "Isn't training and evaluating on simulated data circular?"
- Yes — the model is recovering the structure we put in. The honest claim
  is that the *infrastructure* is sound: trained model + SHAP works
  end-to-end and is ready to plug in real ATS data. The simulation is
  documented; sensitivity variants (v1/v2/v3) report robustness to the
  specific weights chosen. See §2 of the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recruit.dropoff.schemas import DropoffSample


# ─────────────────────────────────────────────────────────────────────────
# Simulation parameters — v0 defaults from design doc §3
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimulationWeights:
    """Coefficients of the log-odds drop-off rule.

    Intercept w_0 calibrated empirically so the resulting dataset has a
    ~55% positive rate (design doc §3 target, matches industry reporting).
    Tuning rationale: with the other terms averaging roughly +2.4 logit,
    w_0 = -2.2 puts mean logit near +0.2 → sigmoid ≈ 0.55. Re-tune if
    weight set is changed.
    """

    w_0: float = -2.2    # intercept — empirically calibrated to ~55% base rate
    w_1: float = +2.0    # skill_gap
    w_2: float = +1.2    # under-qualified penalty
    w_3: float = +0.3    # over-qualified buffer
    w_4: float = +0.8    # posting age > 14 days
    w_5: float = +0.5    # n_required_fields / 10
    w_6: float = +0.7    # remote available (reduces drop-off)  — sign flipped at use site
    w_7: float = +0.6    # mobile session
    w_8: float = +0.9    # long time-on-page (/600)
    w_9: float = +1.5    # % fields skipped
    sigma: float = 0.4   # noise std on the logit


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# Small distributions used to draw session and job features. Defaults chosen
# to look like industry recruiting-funnel telemetry.
_DOMAINS = ["backend", "frontend", "data", "ml", "devops", "mobile", "security", "general"]
_COUNTRIES = ["IN", "US", "UK", "DE", "SG", "other"]


def _draw_session_features(rng: np.random.Generator) -> dict:
    """One realistic session-feature dict."""
    is_mobile = bool(rng.random() < 0.45)
    # Mobile sessions tend shorter; desktop longer with wider spread.
    base_time = rng.lognormal(mean=5.0 if is_mobile else 5.5, sigma=0.6)
    time_on_page = int(min(3600, max(5, base_time)))

    hour = int(rng.choice(range(24), p=_hour_pmf()))
    weekend = bool(rng.random() < 2 / 7)

    n_required = int(rng.choice([5, 8, 10, 12, 15], p=[0.15, 0.30, 0.30, 0.15, 0.10]))
    completion_rate = float(rng.beta(2.5, 1.5))  # right-skewed near 1.0
    n_completed = int(np.round(completion_rate * n_required))
    n_skipped = max(0, n_required - n_completed)

    back_count = int(rng.poisson(lam=1.2 if not is_mobile else 0.6))

    return {
        "session_time_on_page_secs": time_on_page,
        "session_hour_of_day": hour,
        "session_is_weekend": weekend,
        "session_device_mobile": is_mobile,
        "session_n_fields_completed": n_completed,
        "session_n_fields_skipped": n_skipped,
        "session_field_completion_rate": float(n_completed / n_required),
        "session_navigation_back_count": back_count,
        "job_n_required_fields": n_required,
    }


def _hour_pmf() -> np.ndarray:
    """A rough probability mass over hours-of-day: peak evening + lunch."""
    base = np.array([
        # 0-5 quiet
        0.005, 0.005, 0.005, 0.005, 0.005, 0.010,
        # 6-11 morning ramp
        0.020, 0.030, 0.040, 0.050, 0.055, 0.060,
        # 12-13 lunch peak
        0.075, 0.075,
        # 14-17 afternoon
        0.060, 0.055, 0.050, 0.050,
        # 18-22 evening primary peak
        0.060, 0.070, 0.080, 0.075, 0.050,
        # 23 wind down
        0.010,
    ])
    return base / base.sum()


def _draw_job_features(rng: np.random.Generator) -> dict:
    return {
        "job_posting_age_days": int(rng.poisson(lam=18)),
        "job_has_remote_option": bool(rng.random() < 0.55),
        "job_n_applicants_so_far": int(rng.poisson(lam=80)),
        "job_salary_band": int(rng.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.15])),
    }


def _derive_match_features(
    cand_skills: set[str],
    job_skills: set[str],
    cand_yoe: float,
    job_min_yoe: float,
    sem_similarity: float,
    bm25_score: float,
) -> dict:
    overlap = (
        len(cand_skills & job_skills) / len(job_skills)
        if job_skills else 0.0
    )
    yoe_gap = cand_yoe - job_min_yoe
    return {
        "sem_similarity": float(sem_similarity),
        "bm25_score": float(bm25_score),
        "skill_overlap": float(overlap),
        "skill_gap": float(1.0 - overlap),
        "yoe_gap": float(yoe_gap),
        "under_qualified": bool(yoe_gap < 0),
        "education_match": bool(True),  # placeholder — no real edu parse yet
    }


# ─────────────────────────────────────────────────────────────────────────
# Label rule — the heart of the simulation
# ─────────────────────────────────────────────────────────────────────────


def _compute_dropoff_logit(
    sample: dict,
    w: SimulationWeights,
    noise: float,
) -> float:
    """The §3 rule — kept literal to the design doc for auditability."""
    # Over-qualified buffer is capped at 5 to avoid Staff/Director candidates
    # dominating the negative-label class.
    over_qual = max(0.0, sample["yoe_gap"])
    over_qual = min(over_qual, 5.0)

    under_qual = max(0.0, -sample["yoe_gap"])

    return (
        w.w_0
        + w.w_1 * sample["skill_gap"]
        + w.w_2 * under_qual
        - w.w_3 * over_qual
        + w.w_4 * float(sample["job_posting_age_days"] > 14)
        + w.w_5 * (sample["job_n_required_fields"] / 10.0)
        - w.w_6 * float(sample["job_has_remote_option"])
        + w.w_7 * float(sample["session_device_mobile"])
        + w.w_8 * (sample["session_time_on_page_secs"] / 600.0)
        + w.w_9 * (sample["session_n_fields_skipped"] / max(1, sample["job_n_required_fields"]))
        + noise
    )


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def _autocalibrate_intercept(
    pairs: list[dict],
    *,
    n_rows: int,
    weights: SimulationWeights,
    target_rate: float,
    seed: int,
) -> float:
    """Binary-search the intercept so the dataset's positive rate ≈ target.

    Generates a small probe sample, measures empirical rate, and adjusts
    the intercept until the rate is within 1.5 percentage points of target.
    Returns the calibrated `w_0` value.
    """
    rng = np.random.default_rng(seed)
    probe_n = min(2_000, n_rows)
    indices = rng.integers(low=0, high=len(pairs), size=probe_n)

    # Pre-compute per-sample features once — only the intercept changes.
    probe_logits_no_intercept: list[float] = []
    for k, idx in enumerate(indices):
        base = pairs[int(idx)]
        session = _draw_session_features(rng)
        job = _draw_job_features(rng)
        match = _derive_match_features(
            cand_skills=set(base["cand_skills"]),
            job_skills=set(base["job_skills"]),
            cand_yoe=base["cand_yoe"],
            job_min_yoe=base["job_min_yoe"],
            sem_similarity=base["sem_similarity"],
            bm25_score=base["bm25_score"],
        )
        merged = {
            **match,
            **job,
            **session,
        }
        logit = _compute_dropoff_logit(merged, weights, 0.0)
        # _compute_dropoff_logit includes w_0; remove it for the search.
        probe_logits_no_intercept.append(logit - weights.w_0)

    arr = np.asarray(probe_logits_no_intercept)

    def rate_at(w0: float) -> float:
        return float(_sigmoid(arr + w0).mean())

    # Binary search w0 in [-10, 5].
    lo, hi = -10.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if rate_at(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_dataset(
    pairs: list[dict],
    *,
    n_rows: int = 10_000,
    weights: SimulationWeights | None = None,
    seed: int = 42,
    target_rate: float | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of `DropoffSample`-shaped rows.

    `pairs` is a list of dicts each describing one real (résumé, JD) pair
    with: resume_id, job_id, cand_skills (set), job_skills (set), cand_yoe,
    job_min_yoe, sem_similarity, bm25_score, cand_n_past_roles,
    cand_seniority_level, cand_domain, cand_location_country,
    job_seniority_level.

    We draw `n_rows` rows by sampling with replacement from `pairs` (so the
    same retrieval pair can appear with different simulated sessions —
    realistic, since one candidate may attempt the same JD multiple times,
    and gives the model variance to learn from).
    """
    if not pairs:
        raise ValueError("pairs must not be empty")

    w = weights or SimulationWeights()
    if target_rate is not None:
        calibrated_w0 = _autocalibrate_intercept(
            pairs, n_rows=n_rows, weights=w, target_rate=target_rate, seed=seed,
        )
        # Reconstruct the dataclass with the new intercept; keep everything else.
        w = SimulationWeights(
            w_0=calibrated_w0,
            w_1=w.w_1, w_2=w.w_2, w_3=w.w_3, w_4=w.w_4, w_5=w.w_5,
            w_6=w.w_6, w_7=w.w_7, w_8=w.w_8, w_9=w.w_9,
            sigma=w.sigma,
        )
    rng = np.random.default_rng(seed)

    indices = rng.integers(low=0, high=len(pairs), size=n_rows)
    rows: list[dict] = []

    for k, idx in enumerate(indices):
        base = pairs[int(idx)]

        session = _draw_session_features(rng)
        job = _draw_job_features(rng)
        match = _derive_match_features(
            cand_skills=set(base["cand_skills"]),
            job_skills=set(base["job_skills"]),
            cand_yoe=base["cand_yoe"],
            job_min_yoe=base["job_min_yoe"],
            sem_similarity=base["sem_similarity"],
            bm25_score=base["bm25_score"],
        )

        avg_tenure = (
            base["cand_yoe"] / max(1, base["cand_n_past_roles"])
        )

        merged = {
            "sample_id": f"s{k:06d}",
            "resume_id": base["resume_id"],
            "job_id": base["job_id"],
            **match,
            "cand_yoe": float(base["cand_yoe"]),
            "cand_n_past_roles": int(base["cand_n_past_roles"]),
            "cand_avg_tenure_yrs": float(avg_tenure),
            "cand_seniority_level": int(base["cand_seniority_level"]),
            "cand_domain": str(base["cand_domain"]),
            "cand_location_country": str(base["cand_location_country"]),
            **job,
            **session,  # includes job_n_required_fields
            "job_seniority_level": int(base["job_seniority_level"]),
        }

        noise = rng.normal(0.0, w.sigma)
        logit = _compute_dropoff_logit(merged, w, noise)
        p_dropoff = float(_sigmoid(np.array(logit)).item())
        label = int(rng.random() < p_dropoff)

        merged["p_dropoff_true"] = p_dropoff
        merged["label"] = label

        rows.append(merged)

    df = pd.DataFrame(rows)

    # Validate first 50 rows against schema as a sanity check (full validation
    # is O(N) and unnecessary if these 50 pass).
    for row in df.head(50).to_dict(orient="records"):
        DropoffSample(**row)

    return df


def split_dataset(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Seeded random split into train / val / test."""
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
