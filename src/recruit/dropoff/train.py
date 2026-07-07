"""Train and evaluate the drop-off classifier.

WHY logistic regression as a baseline:
- Always have something simple to beat. If XGBoost only marginally beats
  logistic regression, the tree-based model's complexity is not justified.
- A monotonic-coefficient logistic is also a sanity check on the simulation:
  feature signs should match the rule's signs.

WHY XGBoost as the main model:
- Tabular, mixed-type (continuous + categorical + boolean) features.
- Natural fit for SHAP via TreeExplainer.
- Plan §7 specifies XGBoost as the chapter target.

WHY no Optuna in this POC pass:
- Default hyperparameters are good enough for a working demo; tuning is
  ~1 day extra effort and the marginal AUC lift is small at this dataset
  size. Deferred to final-submission scope per the timeline cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recruit.dropoff.schemas import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)


@dataclass(frozen=True)
class ModelReport:
    """Summary metrics for one trained model on one split."""

    name: str
    split: str
    roc_auc: float
    pr_auc: float
    f1_at_0_5: float
    f1_optimal: float
    threshold_optimal: float
    brier: float
    n: int

    def as_row(self) -> str:
        return (
            f"  {self.name:<14s} | {self.split:<5s} | "
            f"AUC={self.roc_auc:.3f}  PR-AUC={self.pr_auc:.3f}  "
            f"F1@0.5={self.f1_at_0_5:.3f}  F1*={self.f1_optimal:.3f} "
            f"(τ={self.threshold_optimal:.2f})  Brier={self.brier:.3f}  n={self.n}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing — shared between logistic baseline and XGBoost
# ─────────────────────────────────────────────────────────────────────────


def _numeric_columns() -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def _build_preprocessor() -> ColumnTransformer:
    """One-hot for categoricals; standardise numerics for the logistic
    baseline. XGBoost does not need scaling but it is harmless.
    """
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


# ─────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────


def fit_logistic(train_df: pd.DataFrame) -> Pipeline:
    """Logistic regression baseline."""
    pipe = Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ])
    pipe.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    return pipe


def tune_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Optuna TPE search over XGBoost hyperparameters.

    Returns the best-trial params dict. Use these with `fit_xgboost(...)`
    to materialise the model.

    Search space follows the plan §7 list — depth, learning rate,
    estimators, regularisation. Objective is validation ROC-AUC.
    """
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pre = _build_preprocessor().fit(train_df[FEATURE_COLUMNS])
    X_train = pre.transform(train_df[FEATURE_COLUMNS])
    X_val = pre.transform(val_df[FEATURE_COLUMNS])
    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values

    n_pos = int((train_df[LABEL_COLUMN] == 1).sum())
    n_neg = int((train_df[LABEL_COLUMN] == 0).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
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


from sklearn.base import BaseEstimator, ClassifierMixin


class _IsotonicCalibratedWrapper(ClassifierMixin, BaseEstimator):
    """Wraps (fitted base classifier + fitted IsotonicRegression).

    sklearn 1.8 removed `CalibratedClassifierCV(cv='prefit')`. We instead
    fit an IsotonicRegression directly on the base classifier's validation
    predictions and apply it at inference. This class duck-types as a fitted
    sklearn classifier so a `Pipeline([pre, this])` works end-to-end.
    """

    def __init__(self, base=None, calibrator=None):
        self.base = base
        self.calibrator = calibrator

    def __sklearn_is_fitted__(self) -> bool:
        return self.base is not None and self.calibrator is not None

    @property
    def classes_(self):
        return getattr(self.base, "classes_", np.array([0, 1]))

    def fit(self, X, y=None):
        """No-op — the base and calibrator are constructed already fitted.

        sklearn 1.8's check_is_fitted requires a `.fit` attribute on the
        estimator even for prefit-style wrappers; this satisfies that
        contract without retraining.
        """
        return self

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_pipeline(
    pipeline: Pipeline,
    val_df: pd.DataFrame,
) -> Pipeline:
    """Wrap a fitted pipeline with isotonic probability calibration.

    The underlying XGBoost is not refit; only the isotonic mapper is learned
    on the validation set's predictions. Returns a new Pipeline whose
    classifier step is the calibrated wrapper.
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


def fit_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    seed: int = 42,
    **extra_params,
) -> Pipeline:
    """XGBoost with reasonable defaults + early stopping on val AUC.

    Wrapped in the same ColumnTransformer pipeline so callers don't need
    different preprocessing paths for the two models.
    """
    from xgboost import XGBClassifier

    n_pos = int((train_df[LABEL_COLUMN] == 1).sum())
    n_neg = int((train_df[LABEL_COLUMN] == 0).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    pre = _build_preprocessor().fit(train_df[FEATURE_COLUMNS])
    X_train = pre.transform(train_df[FEATURE_COLUMNS])
    X_val = pre.transform(val_df[FEATURE_COLUMNS])
    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values

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

    # Wrap so .predict_proba(df) works at serving time
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────


def _f1_optimal_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Search F1 over a fine grid of thresholds; return (F1, τ)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        y_pred = (p >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = float(score), float(thr)
    return best_f1, best_thr


def evaluate(name: str, split: str, model: Pipeline, df: pd.DataFrame) -> ModelReport:
    """Run all the metrics for one trained model on one DataFrame."""
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].values
    p = model.predict_proba(X)[:, 1]

    f1_05 = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
    f1_opt, thr_opt = _f1_optimal_threshold(y, p)

    return ModelReport(
        name=name,
        split=split,
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        f1_at_0_5=f1_05,
        f1_optimal=f1_opt,
        threshold_optimal=thr_opt,
        brier=float(brier_score_loss(y, p)),
        n=len(df),
    )


def save_model(model: Pipeline, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def confusion_at(threshold: float, y_true: np.ndarray, p: np.ndarray) -> dict:
    """Confusion-matrix counts at a given threshold."""
    y_pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
