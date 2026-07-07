"""Train logistic baseline + XGBoost on the v0 drop-off dataset.

Run:  uv run python scripts/train_dropoff.py

Reports ROC-AUC, PR-AUC, F1@0.5, F1*, threshold*, Brier for both models on
val and test splits. Saves the XGBoost model to models/dropoff_v0.joblib.

WHY a baseline:
- If logistic regression gets AUC 0.95 too, XGBoost adds no value here and
  we should say so. Reporting both keeps us honest.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from recruit.config import settings
from recruit.dropoff.schemas import LABEL_COLUMN
from recruit.dropoff.train import (
    confusion_at,
    evaluate,
    fit_logistic,
    fit_xgboost,
    save_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0",
                    help="Dataset variant suffix (must match generate_dropoff_data.py).")
    args = ap.parse_args()

    proc_dir = settings.data_path / "processed"
    df_path = proc_dir / f"dropoff_{args.variant}.parquet"
    splits_path_v = proc_dir / f"dropoff_splits_{args.variant}.json"
    splits_path_legacy = proc_dir / "dropoff_splits.json"
    splits_path = splits_path_v if splits_path_v.exists() else splits_path_legacy

    if not df_path.exists():
        raise SystemExit(
            f"{df_path} not found — run scripts/generate_dropoff_data.py first."
        )

    print(f"Loading {df_path.name}...")
    df = pd.read_parquet(df_path)
    splits = json.loads(splits_path.read_text())
    train_df = df[df["sample_id"].isin(splits["train"])].reset_index(drop=True)
    val_df = df[df["sample_id"].isin(splits["val"])].reset_index(drop=True)
    test_df = df[df["sample_id"].isin(splits["test"])].reset_index(drop=True)
    print(f"  train: {len(train_df)}   val: {len(val_df)}   test: {len(test_df)}")
    print(f"  pos-rate train/val/test: "
          f"{train_df[LABEL_COLUMN].mean():.3f} / "
          f"{val_df[LABEL_COLUMN].mean():.3f} / "
          f"{test_df[LABEL_COLUMN].mean():.3f}")

    print("\nFitting logistic regression baseline...")
    log_model = fit_logistic(train_df)

    print("Fitting XGBoost...")
    xgb_model = fit_xgboost(train_df, val_df)

    print("\n" + "=" * 72)
    print(f"DROP-OFF MODEL EVALUATION  ·  variant: {args.variant}")
    print("=" * 72)
    print()
    reports = [
        evaluate("logistic", "val",  log_model, val_df),
        evaluate("logistic", "test", log_model, test_df),
        evaluate("xgboost",  "val",  xgb_model, val_df),
        evaluate("xgboost",  "test", xgb_model, test_df),
    ]
    for r in reports:
        print(r.as_row())

    # Confusion matrix at F1*-optimal threshold (test set, XGBoost).
    from recruit.dropoff.schemas import FEATURE_COLUMNS
    xgb_test_report = reports[-1]
    p_test = xgb_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    cm = confusion_at(
        threshold=xgb_test_report.threshold_optimal,
        y_true=test_df[LABEL_COLUMN].values,
        p=p_test,
    )
    print(f"\nConfusion matrix at τ={cm['threshold']:.2f} (XGBoost, test):")
    print(f"  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    print(f"  precision={cm['precision']:.3f}  recall={cm['recall']:.3f}  f1={cm['f1']:.3f}")

    # Save the XGBoost model for downstream use (Streamlit / SHAP / fusion ranker).
    out_path = settings.models_path / f"dropoff_{args.variant}.joblib"
    save_model(xgb_model, out_path)
    print(f"\nSaved XGBoost model to {out_path}")


if __name__ == "__main__":
    main()
