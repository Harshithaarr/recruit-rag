"""Fairness audit on REAL demographic data (HR analytics dataset).

Run:  OMP_NUM_THREADS=1 uv run python scripts/audit_fairness_hr.py

WHY a separate audit script for HR analytics:
- Our main drop-off model's features (sem_similarity, BM25 score, skill
  overlap, session telemetry) don't exist in HR analytics — different
  feature space entirely.
- But HR analytics has REAL gender (not a name-based proxy) and a REAL
  binary label (`target` = 1: looking for a job change). It's a perfect
  validation companion: train a parallel binary classifier on real labels
  + real demographics, run the same fairness audit infrastructure on it.
- Findings here are interpretable as "this is how a similar drop-off model
  behaves on real (not synthetic) labels with real (not proxied) groups."

WHAT this script does:
- Loads HR analytics aug_train.csv.
- Holds out a 20% test split.
- Trains XGBoost to predict `target` from non-sensitive features.
  Sensitive feature `gender` is EXCLUDED from training (the model never
  sees it).
- On the test set, runs the same audit_predictions() machinery used in
  audit_fairness.py — demographic parity, equal opportunity, disparate
  impact — but on REAL gender groups.
- Reports findings and compares to the synthetic-data audit's findings.

VIVA: "Why exclude gender from features?"
- Standard practice in algorithmic fairness — disparate-impact metrics
  measure indirect bias through correlated features. If gender were in the
  feature set, the model could discriminate directly (illegal under most
  hiring regulations). The audit shows what bias arises *anyway* through
  correlated proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recruit.config import settings
from recruit.data.loaders import load_hr_analytics
from recruit.fairness.audit_dropoff import audit_predictions, format_fairness_table


# Features the model is allowed to use. `gender` is explicitly excluded.
_NUMERIC_COLS = ["city_development_index", "training_hours"]
_CATEGORICAL_COLS = [
    "city", "relevent_experience", "enrolled_university",
    "education_level", "major_discipline", "experience",
    "company_size", "company_type", "last_new_job",
]
_FEATURE_COLS = _NUMERIC_COLS + _CATEGORICAL_COLS
_LABEL_COL = "target"
_SENSITIVE_COL = "gender"


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs with a sentinel string for categoricals; coerce numerics."""
    out = df.copy()
    for c in _CATEGORICAL_COLS:
        out[c] = out[c].fillna("unknown").astype(str)
    for c in _NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(out[c].median())
    return out


def main() -> None:
    print("Loading HR analytics aug_train.csv...")
    df = load_hr_analytics(
        settings.data_path / "raw" / "hr_analytics" / "aug_train.csv"
    )
    print(f"  total rows: {len(df)}")
    print(f"  positive rate (target=1): {df[_LABEL_COL].mean():.3f}")
    print(f"  gender distribution:\n{df[_SENSITIVE_COL].value_counts(dropna=False).to_string()}")

    # Keep only rows where gender is one of {Male, Female, Other} so the
    # audit groups are well-defined. The "unknown gender" subset (~24%) is
    # dropped from the AUDIT but still kept in training so the model has
    # the full distribution.
    df = _preprocess(df)

    # Train/test split — stratify on the label so train/test class balance
    # matches.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[_LABEL_COL],
    )
    print(f"\nTrain: {len(train_df)}   Test: {len(test_df)}")

    # ── Fit XGBoost on non-sensitive features only ─────────────────────
    print("\nFitting XGBoost (gender intentionally EXCLUDED from features)...")
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    X_train = pre.fit_transform(train_df[_FEATURE_COLS])
    X_test = pre.transform(test_df[_FEATURE_COLS])
    y_train = train_df[_LABEL_COL].values
    y_test = test_df[_LABEL_COL].values

    from xgboost import XGBClassifier
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos))

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    p_test = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    f1 = float(f1_score(y_test, (p_test >= 0.5).astype(int)))
    print(f"\nClassifier performance on test:")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  F1 @ 0.5: {f1:.3f}")

    # ── Fairness audit on REAL gender groups ──────────────────────────
    print("\n" + "=" * 72)
    print("PART 1 — REAL-DATA FAIRNESS AUDIT  ·  HR analytics  ·  real gender")
    print("=" * 72)

    # Drop "unknown" rows from the audit only; the model still saw them at
    # train time. We're auditing whether the model treats observable gender
    # groups equitably.
    audit_mask = test_df[_SENSITIVE_COL].isin(["Male", "Female", "Other"]).values
    n_audited = int(audit_mask.sum())
    n_skipped = int(len(test_df) - n_audited)
    print(f"\nAuditing {n_audited} test rows ({n_skipped} skipped — missing gender)")

    summary = audit_predictions(
        attribute="gender",
        y_true=y_test[audit_mask],
        p_predicted=p_test[audit_mask],
        group_by_row=test_df.loc[audit_mask, _SENSITIVE_COL].values,
        threshold=0.5,
        min_group_n=30,
    )
    print()
    print(format_fairness_table(summary))

    # ── Comparison context ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PART 2 — COMPARISON TO SYNTHETIC-DATA AUDIT")
    print("=" * 72)
    print(
        "\nSynthetic audit (audit_fairness.py, country proxy on HF résumés):\n"
        "  demographic parity diff:  0.117\n"
        "  equal opportunity diff:   0.140\n"
        "  disparate impact ratio:   0.812  (PASS 4/5 rule)\n"
        "  attribute: country (proxy)\n"
    )
    print(
        f"This audit (HR analytics, real gender):\n"
        f"  demographic parity diff:  {summary.demographic_parity_diff:.3f}\n"
        f"  equal opportunity diff:   {summary.equal_opportunity_diff:.3f}\n"
        f"  disparate impact ratio:   {summary.disparate_impact_ratio:.3f}  "
        f"({'PASS' if summary.passes_4_5_rule() else 'FAIL'} 4/5 rule)\n"
        f"  attribute: gender (real)\n"
    )
    print(
        "Both audits use the SAME infrastructure (audit_predictions) — apples\n"
        "to apples for the metric values. The interesting comparison is\n"
        "whether the magnitudes are similar (suggesting our synthetic-data\n"
        "fairness picture is roughly representative) or very different\n"
        "(suggesting synthetic results don't transfer)."
    )

    # ── Persist a JSON report so the UI can display these numbers ─────
    # The UI reads reports/fairness_hr.json to render the fairness panel.
    # Regenerate this file whenever the audit is re-run (`make audit-hr`).
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "reports" / "fairness_hr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "attribute": summary.attribute,
        "threshold": summary.threshold,
        "n_audited": n_audited,
        "n_skipped": n_skipped,
        "demographic_parity_diff": summary.demographic_parity_diff,
        "equal_opportunity_diff": summary.equal_opportunity_diff,
        "disparate_impact_ratio": summary.disparate_impact_ratio,
        "passes_4_5_rule": summary.passes_4_5_rule(),
        "groups": [
            {
                "group": g.group,
                "n": g.n,
                "base_rate": g.base_rate,
                "selection_rate": g.selection_rate,
                "tpr": g.tpr,
                "fpr": g.fpr,
            }
            for g in summary.groups
        ],
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote fairness report → {out_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
