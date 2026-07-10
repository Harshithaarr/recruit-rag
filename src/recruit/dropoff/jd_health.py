"""Job-posting health score — aggregate drop-off signals across the shortlist.

End-sem extension. Per-candidate drop-off prediction is passive triage —
the recruiter reads the risk score and moves on. This module goes one
level of abstraction up: for a whole JD (i.e. across every candidate who
matches it), compute a **health score** identifying the top signals
driving drop-off across the funnel.

Reframes drop-off from a per-candidate signal into a **systems signal**.
Instead of *"this candidate is at risk"* it says *"this JD posting is
losing candidates at the salary field — recruiters should reconsider
whether that field is required, or move it later in the flow."*

WHY novel:
- Almost every deployed hiring tool operates per-candidate. Aggregating
  to per-JD gives the hiring team feedback on the *posting itself*, not
  just the candidates.
- Reuses the calibrated model + SHAP explainer — no new model training.

VIVA: "How is this different from a report on skill fit?"
- Skill-fit reports say "your JD's skill requirements are too narrow."
  This says "your JD's *application flow* is losing candidates at
  specific fields." The signal is behavioural, not textual.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class DriverAggregate:
    """One feature's aggregated importance across the shortlist."""

    feature: str
    times_in_top_3: int         # how many candidates had this feature as a top-3 driver
    mean_shap: float            # average SHAP value across candidates who had it as top-3
    direction: str              # "↑ drop-off" or "↓ drop-off" (majority direction)
    n_candidates: int           # total candidates aggregated over


@dataclass(frozen=True)
class JDHealthReport:
    """Aggregated health metrics for one JD across its shortlist."""

    n_candidates: int
    mean_dropoff_prob: float
    median_dropoff_prob: float
    pct_high_risk: float               # share with P >= 0.6
    pct_low_risk: float                # share with P < 0.3
    top_drivers: list[DriverAggregate] # ranked by times_in_top_3 (desc)

    @property
    def health_score(self) -> float:
        """Overall health in [0, 1]: 1 = low-drop-off cohort, 0 = high-drop-off cohort.

        Anchored around the mean P(drop-off): a mean of 0.3 (near the low-risk
        band) gives ~0.7, a mean of 0.6 (near high-risk) gives ~0.4.
        """
        return max(0.0, min(1.0, 1.0 - self.mean_dropoff_prob))

    @property
    def health_band(self) -> str:
        h = self.health_score
        if h >= 0.65:
            return "healthy"
        if h >= 0.45:
            return "moderate"
        return "concerning"


def compute_jd_health(
    rows: list[dict],
    *,
    top_k_drivers: int = 5,
) -> JDHealthReport:
    """Aggregate drop-off predictions + SHAP drivers across the shortlist.

    Each `row` is a scored candidate dict from `score_candidates` — must
    contain 'p_dropoff' and 'prediction' (with .explanation.top_attributions).

    Returns a `JDHealthReport` with aggregated statistics.
    """
    if not rows:
        return JDHealthReport(
            n_candidates=0,
            mean_dropoff_prob=0.0,
            median_dropoff_prob=0.0,
            pct_high_risk=0.0,
            pct_low_risk=0.0,
            top_drivers=[],
        )

    probs = sorted(float(r["p_dropoff"]) for r in rows)
    n = len(probs)
    mean_p = sum(probs) / n
    median_p = probs[n // 2] if n % 2 == 1 else 0.5 * (probs[n // 2 - 1] + probs[n // 2])

    n_high = sum(1 for p in probs if p >= 0.60)
    n_low = sum(1 for p in probs if p < 0.30)

    # Aggregate SHAP drivers — count how often each feature appears in a
    # candidate's top-3 attribution list, and track the mean SHAP value
    # and majority direction.
    driver_counter: Counter[str] = Counter()
    driver_shap_sum: dict[str, float] = defaultdict(float)
    driver_pos_count: dict[str, int] = defaultdict(int)
    driver_neg_count: dict[str, int] = defaultdict(int)

    for r in rows:
        pred = r.get("prediction")
        if pred is None:
            continue
        # LocalExplanation.top_attributions[:top_k]
        top_attrs = pred.explanation.top_attributions[:3]
        for attr in top_attrs:
            driver_counter[attr.feature] += 1
            driver_shap_sum[attr.feature] += attr.value
            if attr.value > 0:
                driver_pos_count[attr.feature] += 1
            else:
                driver_neg_count[attr.feature] += 1

    # Rank drivers by frequency in top-3 (desc); keep top_k_drivers.
    top_drivers: list[DriverAggregate] = []
    for feature, count in driver_counter.most_common(top_k_drivers):
        mean_shap = driver_shap_sum[feature] / count
        direction = "↑ drop-off" if driver_pos_count[feature] >= driver_neg_count[feature] else "↓ drop-off"
        top_drivers.append(DriverAggregate(
            feature=feature,
            times_in_top_3=count,
            mean_shap=mean_shap,
            direction=direction,
            n_candidates=n,
        ))

    return JDHealthReport(
        n_candidates=n,
        mean_dropoff_prob=mean_p,
        median_dropoff_prob=median_p,
        pct_high_risk=n_high / n,
        pct_low_risk=n_low / n,
        top_drivers=top_drivers,
    )
