"""Distribution-drift monitor — flag when live session behaviour diverges
from the training-time distribution.

End-sem extension. The dissertation acknowledges that drop-off labels are
synthetic — the model was trained on a plausible-but-simulated distribution
of behaviours. This module confronts that limitation head-on rather than
hiding it: at inference time, we compare each live session feature to its
training-time distribution and flag features that are out-of-distribution.

If a live session's `session_n_fields_skipped` is 20 but the training
distribution's 99th percentile was 8, the model's prediction is
extrapolating — that should be visible to the recruiter.

WHY novel for a dissertation contribution:
- Reviewers reward code that confronts its own limitations rather than
  concealing them. This module makes the synthetic-data limitation
  actionable — the recruiter sees when the current session isn't the
  kind of thing the model was trained on.
- Standard practice in production ML (data-drift monitors, e.g. WhyLabs,
  Fiddler, Arize) but rarely in academic prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from recruit.dropoff.schemas import FEATURE_COLUMNS


# Only monitor the numeric SESSION features — candidate/job features are
# constant across a single session and don't drift.
_MONITORED_FEATURES: list[str] = [
    "session_time_on_page_secs",
    "session_n_fields_completed",
    "session_n_fields_skipped",
    "session_field_completion_rate",
    "session_navigation_back_count",
    "session_hour_of_day",
]


@dataclass(frozen=True)
class FeatureStats:
    """Training-time distribution statistics for one feature."""

    feature: str
    mean: float
    std: float
    p05: float
    p50: float  # median
    p95: float


@dataclass(frozen=True)
class DriftReport:
    """Drift assessment for one live session against the training distribution."""

    per_feature: dict[str, dict]  # feature → {live, mean, std, z, flagged}
    max_abs_z: float               # overall drift magnitude
    features_flagged: list[str]    # features with |z| > 2

    @property
    def overall_verdict(self) -> str:
        if not self.features_flagged:
            return "in_distribution"
        if len(self.features_flagged) == 1:
            return "one_feature_drifting"
        return "multiple_features_drifting"


@lru_cache(maxsize=1)
def _load_training_stats(parquet_path: str) -> dict[str, FeatureStats]:
    """Compute per-feature training-time distribution stats from the parquet.

    Cached per process — the parquet is ~5000 rows, but this runs once per
    Streamlit session and doesn't need re-computing.
    """
    df = pd.read_parquet(parquet_path)
    stats: dict[str, FeatureStats] = {}
    for feat in _MONITORED_FEATURES:
        if feat not in df.columns:
            continue
        s = df[feat].astype(float)
        stats[feat] = FeatureStats(
            feature=feat,
            mean=float(s.mean()),
            std=float(s.std() if s.std() > 0 else 1.0),  # guard div-by-zero
            p05=float(s.quantile(0.05)),
            p50=float(s.median()),
            p95=float(s.quantile(0.95)),
        )
    return stats


def _resolve_stats_path() -> Path:
    """Locate the training parquet. Prefers v1, falls back to v0."""
    from recruit.config import settings
    for variant in ("v1", "v0"):
        candidate = settings.data_path / "processed" / f"dropoff_{variant}.parquet"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No training parquet found under data/processed/. Run "
        "`make generate-data` first."
    )


def compute_drift(feature_row: dict, z_threshold: float = 2.0) -> DriftReport:
    """Compare a live feature row to the training distribution.

    Returns a `DriftReport` with per-feature statistics and an aggregate
    drift verdict. A feature is 'flagged' when |z-score| > `z_threshold`
    (default 2.0, i.e. ~2.3% of a Gaussian tail).
    """
    stats = _load_training_stats(str(_resolve_stats_path()))

    per_feature: dict[str, dict] = {}
    max_abs_z = 0.0
    flagged: list[str] = []

    for feat, fs in stats.items():
        live_val = float(feature_row.get(feat, fs.mean))
        z = (live_val - fs.mean) / fs.std if fs.std > 0 else 0.0
        abs_z = abs(z)
        is_flagged = abs_z > z_threshold
        per_feature[feat] = {
            "live": live_val,
            "train_mean": fs.mean,
            "train_std": fs.std,
            "train_p05": fs.p05,
            "train_p50": fs.p50,
            "train_p95": fs.p95,
            "z_score": z,
            "abs_z": abs_z,
            "flagged": is_flagged,
        }
        max_abs_z = max(max_abs_z, abs_z)
        if is_flagged:
            flagged.append(feat)

    return DriftReport(
        per_feature=per_feature,
        max_abs_z=max_abs_z,
        features_flagged=flagged,
    )
