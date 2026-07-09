"""SHAP explanations for the post-offer decline classifier.

Mirrors `dropoff/explain.py` in structure — same TreeExplainer,
same waterfall design, same textualise-for-RAG contract. Differences:

  · Feature schema — uses `postoffer/schemas.py` FEATURE_COLUMNS
  · Human labels   — pretty-print the post-offer feature names via
                     `postoffer/schemas.py::FEATURE_LABELS`
  · Reused types   — `ShapAttribution` and `LocalExplanation` come from
                     `dropoff/explain.py` (they are schema-agnostic)

VIVA: "Why doesn't this module duplicate ShapAttribution / LocalExplanation?"
- Those classes are generic — one feature's SHAP value and a top-k list of
  them. Duplicating them here would just create two identical dataclasses
  and a divergence risk. Importing keeps the module boundaries clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Reuse the schema-agnostic dataclasses — no reason to duplicate them.
from recruit.dropoff.explain import (
    LocalExplanation,
    ShapAttribution,
)
from recruit.postoffer.schemas import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
)


# ─── Small helpers (copied from dropoff/explain.py — kept local to
# avoid pulling private names across module boundaries) ─────────────────


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _short_value(v: object) -> str:
    """Compact label for the waterfall bar."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}" if -1.0 <= v <= 1.0 else f"{v:.1f}"
    return str(v)


def _format_value(v: object) -> str:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}" if -1.0 <= v <= 1.0 else f"{v:.1f}"
    return repr(v)


def _risk_label(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "moderate"
    return "high"


def _expanded_columns_for(
    feat: str, feature_names_out: list[str]
) -> list[int]:
    """Indices into the one-hot-expanded feature list belonging to `feat`.

    For a numeric feature `num__feat` → the single matching index.
    For a categorical feature `cat__feat_<level>` → all matching indices.
    """
    prefix_num = f"num__{feat}"
    prefix_cat = f"cat__{feat}_"
    return [
        i for i, name in enumerate(feature_names_out)
        if name == prefix_num or name.startswith(prefix_cat)
    ]


# ─── Explainer ──────────────────────────────────────────────────────────


class PostOfferExplainer:
    """TreeExplainer wrapper for the post-offer XGBoost model.

    Constructed from a fitted `Pipeline` produced by `postoffer/train.py`.
    Unwraps the isotonic-calibrated variant automatically — SHAP needs
    the raw tree model, not the calibrated wrapper.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        import shap

        self._pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]

        # Same unwrap trick as dropoff/explain.py — the calibrated wrapper
        # is a monotonic post-hoc transform on probabilities and doesn't
        # affect which features drove the prediction.
        tree_model = clf
        if (
            type(clf).__name__ == "_IsotonicCalibratedWrapper"
            and hasattr(clf, "base")
            and clf.base is not None
        ):
            tree_model = clf.base
        self._clf = tree_model
        self._feature_names_out = list(self._pre.get_feature_names_out())
        self._explainer = shap.TreeExplainer(tree_model)

        ev = self._explainer.expected_value
        ev_arr = np.atleast_1d(ev)
        self._base_value = float(ev_arr[-1])

    # ------------------------------------------------------------------
    # Per-prediction local explanation
    # ------------------------------------------------------------------
    def explain_row(
        self,
        row: pd.Series | dict,
        *,
        sample_id: str | None = None,
        top_k: int = 5,
    ) -> LocalExplanation:
        """Compute SHAP for one row and package into a LocalExplanation."""
        if isinstance(row, dict):
            row_df = pd.DataFrame([row])
        else:
            row_df = row.to_frame().T

        X = self._pre.transform(row_df[FEATURE_COLUMNS])
        sv = np.asarray(self._explainer.shap_values(X)).reshape(-1)

        # Aggregate one-hot expansions back to source feature names so the
        # SHAP output is per-input-feature, not per-encoded-column.
        agg: dict[str, float] = {}
        for feat in FEATURE_COLUMNS:
            cols = _expanded_columns_for(feat, self._feature_names_out)
            if not cols:
                continue
            agg[feat] = float(np.sum(sv[cols]))

        # Sort by absolute contribution, keep top_k.
        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top_k]

        # Reconstruct the probability from the base + all SHAP contributions.
        total_logit = self._base_value + float(sum(agg.values()))
        predicted_prob = _sigmoid(total_logit)

        source = row_df.iloc[0]
        top_attributions = [
            ShapAttribution(
                feature=FEATURE_LABELS.get(f, f),
                value=v,
                feature_value=source.get(f),
            )
            for f, v in ranked
        ]

        return LocalExplanation(
            sample_id=sample_id or "post-offer",
            predicted_probability=float(predicted_prob),
            base_value_logit=float(self._base_value),
            top_attributions=top_attributions,
        )

    # ------------------------------------------------------------------
    # Per-prediction waterfall figure
    # ------------------------------------------------------------------
    def waterfall_figure(
        self,
        row: pd.Series | dict,
        *,
        top_k: int = 8,
        height_inches: float = 3.0,
    ):
        """Matplotlib figure — mirrors dropoff/explain.py's waterfall design.

        Aggregates SHAP values back to source feature names, sorts by
        magnitude, and shows the top_k as a horizontal bar chart with the
        cumulative logit path from `base_value` to `final_logit`.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if isinstance(row, dict):
            row_df = pd.DataFrame([row])
        else:
            row_df = row.to_frame().T

        X = self._pre.transform(row_df[FEATURE_COLUMNS])
        sv = np.asarray(self._explainer.shap_values(X)).reshape(-1)

        agg: dict[str, float] = {}
        for feat in FEATURE_COLUMNS:
            cols = _expanded_columns_for(feat, self._feature_names_out)
            if not cols:
                continue
            agg[feat] = float(np.sum(sv[cols]))

        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
        top = ranked[:top_k]
        rest = ranked[top_k:]

        labels = [
            f"{FEATURE_LABELS.get(f, f)} = {_short_value(row_df.iloc[0].get(f))}"
            for f, _ in top
        ]
        values = [v for _, v in top]
        if rest:
            labels.append(f"{len(rest)} other features")
            values.append(float(sum(v for _, v in rest)))

        # Bottom-up cumulative path so the largest bars sit at the top.
        labels.reverse()
        values.reverse()
        cumulative = [self._base_value]
        for v in values:
            cumulative.append(cumulative[-1] + v)
        final_logit = cumulative[-1]

        fig, ax = plt.subplots(figsize=(7.0, height_inches))
        positions = np.arange(len(values))
        colors = ["#16a34a" if v < 0 else "#dc2626" for v in values]
        left = cumulative[:-1]
        ax.barh(positions, values, left=left, color=colors, edgecolor="white")

        for i, (l, v) in enumerate(zip(left, values)):
            if abs(v) >= 0.08:
                ax.text(
                    l + v / 2.0,
                    i,
                    f"{v:+.2f}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.axvline(self._base_value, color="#94a3b8", linestyle="--", linewidth=1)
        ax.axvline(final_logit, color="#0f172a", linewidth=1.2)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(
            "Log-odds contribution  (green → ↓ decline,  red → ↑ decline)",
            fontsize=9,
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(
            f"SHAP waterfall   ·   base E[f(x)] = {self._base_value:+.2f}   "
            f"→   final logit = {final_logit:+.2f}   "
            f"(P ≈ {_sigmoid(final_logit):.2f})",
            fontsize=10,
            pad=8,
        )
        plt.tight_layout()
        return fig
