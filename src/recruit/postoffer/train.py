"""Train the post-offer decline classifier — PoC extension.

Mirrors `dropoff/train.py` in architecture (preprocessor → XGBoost →
isotonic calibration) with three deliberate simplifications:

  1. No logistic baseline. The dropoff module keeps one as a sanity check
     against the simulator; the post-offer module is explicitly a PoC and
     the extra baseline would not change the story.
  2. Fewer Optuna trials by default (20 vs. 50) — smaller dataset, weaker
     labels, diminishing returns from deeper search.
  3. Reuses `_IsotonicCalibratedWrapper` from `dropoff/train.py` — the
     wrapper is model-agnostic, keeping DRY without introducing surprise
     coupling.

VIVA: "Why not train from scratch inline in a notebook?"
- A module-level trainer is reproducible from `make train-postoffer` and
  version-controlled. Notebook-only work is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse the exact same isotonic wrapper — same sklearn-1.8 dance, same
# duck-typing contract. It has no drop-off-specific logic inside.
from recruit.dropoff.train import _IsotonicCalibratedWrapper
from recruit.postoffer.schemas import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


LABEL_COLUMN: str = "declined_offer"


# ─── Metrics container ──────────────────────────────────────────────────
@dataclass(frozen=True)
class PostOfferReport:
    """Compact metrics for one trained model on one split."""
    name: str
    split: str
    roc_auc: float
    pr_auc: float
    brier: float
    f1_at_0_5: float
    n: int

    def as_row(self) -> str:
        return (
            f"  {self.name:<18s} | {self.split:<5s} | "
            f"AUC={self.roc_auc:.3f}  PR-AUC={self.pr_auc:.3f}  "
            f"Brier={self.brier:.3f}  F1@0.5={self.f1_at_0_5:.3f}  n={self.n}"
        )


# ─── Preprocessing ──────────────────────────────────────────────────────
def _numeric_columns() -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def _build_preprocessor() -> ColumnTransformer:
    """Standard scale numerics, one-hot the categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _numeric_columns()),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


# ─── Train / tune / calibrate ───────────────────────────────────────────
def fit_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.06,
    seed: int = 42,
    **extra_params,
) -> Pipeline:
    """XGBoost with sensible defaults + early stopping on validation AUC."""
    from xgboost import XGBClassifier

    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    pre = _build_preprocessor().fit(train_df[FEATURE_COLUMNS])
    X_train = pre.transform(train_df[FEATURE_COLUMNS])
    X_val = pre.transform(val_df[FEATURE_COLUMNS])

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **extra_params,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return Pipeline([("pre", pre), ("clf", clf)])


def tune_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_trials: int = 20,
    seed: int = 42,
) -> dict:
    """Optuna TPE search — smaller budget than dropoff, fits the PoC scope."""
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pre = _build_preprocessor().fit(train_df[FEATURE_COLUMNS])
    X_train = pre.transform(train_df[FEATURE_COLUMNS])
    X_val = pre.transform(val_df[FEATURE_COLUMNS])
    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        clf = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, p))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def calibrate_pipeline(pipeline: Pipeline, val_df: pd.DataFrame) -> Pipeline:
    """Wrap a fitted pipeline with isotonic probability calibration.

    Same pattern as `dropoff/train.py:calibrate_pipeline` — fits the
    IsotonicRegression on the fitted model's validation predictions,
    then wraps everything in the `_IsotonicCalibratedWrapper`.
    """
    from sklearn.isotonic import IsotonicRegression

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    X_val = pre.transform(val_df[FEATURE_COLUMNS])
    y_val = val_df[LABEL_COLUMN].values
    p_val = clf.predict_proba(X_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)

    wrapped = _IsotonicCalibratedWrapper(base=clf, calibrator=iso)
    return Pipeline([("pre", pre), ("clf", wrapped)])


# ─── Evaluation ─────────────────────────────────────────────────────────
def evaluate(name: str, split: str, model: Pipeline, df: pd.DataFrame) -> PostOfferReport:
    """Compute AUC / PR-AUC / Brier / F1 for one model on one split."""
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].values
    p = model.predict_proba(X)[:, 1]

    return PostOfferReport(
        name=name,
        split=split,
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        f1_at_0_5=float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        n=len(df),
    )


# ─── I/O ────────────────────────────────────────────────────────────────
def save_model(model: Pipeline, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def make_splits(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split.

    Stratified on the label so all three splits carry a similar decline rate.
    Returns (train_df, val_df, test_df) in that order.
    """
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df[LABEL_COLUMN],
        random_state=seed,
    )
    val_size_relative = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_relative,
        stratify=temp_df[LABEL_COLUMN],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
