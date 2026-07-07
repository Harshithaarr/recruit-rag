"""SHAP explanations for the drop-off classifier.

WHY this module exists (not just inline in train.py):
- SHAP value computation needs the *raw* XGBoost booster, not the sklearn
  Pipeline. The pipeline's preprocessor produces a numpy array that the
  TreeExplainer consumes; the column-name mapping after one-hot encoding
  has to be reconstructed for human-readable output.
- Per-prediction SHAP values are the input to the RAG explainer prompt
  (the SHAP→RAG bridge). That contract is what `textualize_top_k` defines.

VIVA: "Why SHAP and not feature_importance_?"
- feature_importance_ is global only (gain across all splits). SHAP gives
  per-prediction attributions — needed for explaining a single candidate
  to the recruiter. Lundberg & Lee 2017 prove SHAP has the desirable
  properties of local accuracy, missingness, consistency.

VIVA: "Why TreeExplainer specifically?"
- Exact, deterministic, fast for tree models. KernelExplainer would also
  work but is sampling-based and slow. For a tree model TreeExplainer is
  the standard choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from recruit.dropoff.schemas import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShapAttribution:
    """One feature's SHAP contribution for one prediction.

    `value` is the SHAP value in log-odds space. Sign matters:
    positive → pushes prediction toward drop-off=1.
    """

    feature: str
    value: float
    feature_value: object  # the raw feature value for context

    @property
    def direction(self) -> str:
        return "↑ drop-off" if self.value > 0 else "↓ drop-off"

    @property
    def magnitude(self) -> float:
        return abs(self.value)


@dataclass(frozen=True)
class LocalExplanation:
    """Per-prediction SHAP explanation.

    Designed so its `.textualize()` output is the structured context the
    RAG explainer prompt consumes — the SHAP→RAG bridge.
    """

    sample_id: str
    predicted_probability: float
    base_value_logit: float
    top_attributions: list[ShapAttribution]

    def textualize(self, top_k: int = 3) -> str:
        """Compact human-readable summary for use in LLM prompts."""
        prob_pct = self.predicted_probability * 100
        lines = [f"Drop-off risk: {prob_pct:.0f}% ({_risk_label(self.predicted_probability)})"]
        lines.append("Primary drivers:")
        for a in self.top_attributions[:top_k]:
            lines.append(
                f"  · {a.feature} = {_format_value(a.feature_value)}  →  {a.direction}  "
                f"(SHAP {a.value:+.2f})"
            )
        return "\n".join(lines)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _short_value(v: object) -> str:
    """Compact label for the waterfall bar (e.g. 'skill_overlap = 0.62')."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}" if -1.0 <= v <= 1.0 else f"{v:.1f}"
    return str(v)


def _format_value(v: object) -> str:
    """Strip numpy scalar wrappers and round floats for readable prompts."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        # Keep two decimals for ratios, one for larger magnitudes.
        return f"{v:.2f}" if -1.0 <= v <= 1.0 else f"{v:.1f}"
    return repr(v)


def _risk_label(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "moderate"
    return "high"


# ─────────────────────────────────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────────────────────────────────


class DropoffExplainer:
    """Wraps shap.TreeExplainer + the sklearn Pipeline's preprocessor.

    Constructed once from a trained Pipeline. Use:
      - `global_importance(df)`     → DataFrame of mean(|SHAP|) per feature
      - `explain_row(row)`          → LocalExplanation for one row
      - `save_global_plots(df, out_dir)` → beeswarm + bar PNGs
    """

    def __init__(self, pipeline: Pipeline) -> None:
        import shap

        # Pipeline shape from train.py: ('pre', ColumnTransformer) → ('clf', X)
        # where X is normally XGBClassifier — but may be _IsotonicCalibratedWrapper
        # for calibrated variants. SHAP needs the raw tree model, so unwrap.
        self._pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        # The calibrated wrapper exposes the underlying XGBoost as `.base`.
        # SHAP values come from the BASE model (decisions); the calibrator is
        # a post-hoc monotonic transform on probabilities and doesn't change
        # which features drove the prediction.
        #
        # Detect via class-name + attribute (rather than `isinstance`) so this
        # module doesn't have to import from train.py (avoiding a circular
        # dependency) — and so it survives joblib loads of older pickles
        # where the wrapper module path may have shifted.
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
        # Cache the base value (model's expected log-odds output for the dataset).
        # SHAP returns a scalar for binary XGBClassifier, but newer versions wrap
        # it in a length-1 array; handle both.
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
        """Compute SHAP values for one row, returning the top-k by |value|."""
        # Normalise to a single-row DataFrame to keep the ColumnTransformer happy.
        if isinstance(row, dict):
            row_df = pd.DataFrame([row])
        else:
            row_df = row.to_frame().T

        X_transformed = self._pre.transform(row_df[FEATURE_COLUMNS])
        shap_values = self._explainer.shap_values(X_transformed)
        # For binary classifier with `XGBClassifier`, shap_values is shape (1, n_features)
        sv = np.asarray(shap_values).reshape(-1)

        # Aggregate one-hot SHAP values back to their original categorical column
        # so the explanation reads "cand_domain='backend' → ↑ drop-off" rather
        # than "cand_domain_backend → ↑ drop-off".
        agg = _aggregate_categorical_shap(
            sv,
            self._feature_names_out,
            categorical_cols=CATEGORICAL_COLUMNS,
            raw_row=row_df.iloc[0],
        )

        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top_k]
        top = [
            ShapAttribution(
                feature=feat,
                value=float(value),
                feature_value=row_df.iloc[0].get(feat),
            )
            for feat, value in ranked
        ]

        prob = float(self._clf.predict_proba(X_transformed)[0, 1])
        return LocalExplanation(
            sample_id=sample_id or "?",
            predicted_probability=prob,
            base_value_logit=self._base_value,
            top_attributions=top,
        )

    # ------------------------------------------------------------------
    # Global feature importance
    # ------------------------------------------------------------------

    def global_importance(self, df: pd.DataFrame, *, max_samples: int = 1000) -> pd.DataFrame:
        """Mean absolute SHAP value per (aggregated) feature, descending."""
        X = df[FEATURE_COLUMNS].head(max_samples)
        X_trans = self._pre.transform(X)
        shap_matrix = np.asarray(self._explainer.shap_values(X_trans))

        agg_rows: list[dict] = []
        for orig_feat in FEATURE_COLUMNS:
            cols = _expanded_columns_for(orig_feat, self._feature_names_out)
            mean_abs = float(np.mean(np.abs(shap_matrix[:, cols])))
            agg_rows.append({"feature": orig_feat, "mean_abs_shap": mean_abs})

        return (
            pd.DataFrame(agg_rows)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Per-prediction waterfall — for the UI candidate card
    # ------------------------------------------------------------------

    def waterfall_figure(
        self,
        row: pd.Series | dict,
        *,
        top_k: int = 8,
        height_inches: float = 3.0,
    ):
        """Return a matplotlib Figure with the SHAP waterfall for one row.

        Aggregates one-hot SHAP values back to their source feature names so
        the bars are readable ('cand_domain', not 'cat__cand_domain_backend').

        Bars are ordered by |SHAP|; only the top_k contributors are shown,
        with the rest grouped as "X other features".
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

        # Aggregate to original feature names.
        agg: dict[str, float] = {}
        for feat in FEATURE_COLUMNS:
            cols = _expanded_columns_for(feat, self._feature_names_out)
            if not cols:
                continue
            agg[feat] = float(np.sum(sv[cols]))

        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
        top = ranked[:top_k]
        rest = ranked[top_k:]

        labels = [f"{f} = {_short_value(row_df.iloc[0].get(f))}" for f, _ in top]
        values = [v for _, v in top]
        if rest:
            labels.append(f"{len(rest)} other features")
            values.append(sum(v for _, v in rest))

        # Build the figure — horizontal bar chart sorted top-of-plot = largest.
        # Walk the cumulative sum so each bar shows how it shifts the prediction.
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
            # Only draw the value label if the bar is wide enough to hold it.
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

        # Reference lines: base value (dashed grey) and final logit (solid dark).
        ax.axvline(self._base_value, color="#94a3b8", linestyle="--", linewidth=1)
        ax.axvline(final_logit, color="#0f172a", linewidth=1.2)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(
            "Log-odds contribution  (green → ↓ drop-off,  red → ↑ drop-off)",
            fontsize=9,
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Single tidy title on ONE line, no overlap with axis labels.
        ax.set_title(
            f"SHAP waterfall   ·   base E[f(x)] = {self._base_value:+.2f}   "
            f"→   final logit = {final_logit:+.2f}   "
            f"(P ≈ {_sigmoid(final_logit):.2f})",
            fontsize=10,
            pad=8,
        )
        plt.tight_layout()
        return fig

    def save_global_plots(
        self,
        df: pd.DataFrame,
        out_dir: Path,
        *,
        max_samples: int = 500,
    ) -> dict[str, Path]:
        """Save beeswarm + bar PNGs to `out_dir`. Returns the two paths."""
        import matplotlib

        matplotlib.use("Agg")  # headless backend — no Tk needed
        import matplotlib.pyplot as plt
        import shap

        out_dir.mkdir(parents=True, exist_ok=True)
        X = df[FEATURE_COLUMNS].head(max_samples)
        X_trans = self._pre.transform(X)
        shap_values = self._explainer.shap_values(X_trans)

        # Bar plot
        bar_path = out_dir / "shap_bar.png"
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=self._feature_names_out,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120)
        plt.close()

        # Beeswarm plot
        beeswarm_path = out_dir / "shap_beeswarm.png"
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=self._feature_names_out,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=120)
        plt.close()

        return {"bar": bar_path, "beeswarm": beeswarm_path}


# ─────────────────────────────────────────────────────────────────────────
# Helpers — aggregate one-hot SHAP columns back to original feature names
# ─────────────────────────────────────────────────────────────────────────


def _expanded_columns_for(orig_feature: str, feature_names_out: list[str]) -> list[int]:
    """Return positions in `feature_names_out` corresponding to one source column.

    The ColumnTransformer prefixes outputs as 'num__<col>' or 'cat__<col>_<val>'.
    A numeric column has exactly one expanded column; a categorical has many.
    """
    out: list[int] = []
    for i, name in enumerate(feature_names_out):
        # num__sem_similarity, cat__cand_domain_backend, etc.
        body = name.split("__", 1)[-1]
        if body == orig_feature:
            out.append(i)
        elif body.startswith(orig_feature + "_"):
            out.append(i)
    return out


def _aggregate_categorical_shap(
    sv: np.ndarray,
    feature_names_out: list[str],
    *,
    categorical_cols: list[str],
    raw_row: pd.Series,
) -> dict[str, float]:
    """Sum the one-hot SHAP values back to the original feature names."""
    agg: dict[str, float] = {}
    for feat in FEATURE_COLUMNS:
        cols = _expanded_columns_for(feat, feature_names_out)
        if not cols:
            continue
        agg[feat] = float(np.sum(sv[cols]))
    return agg
