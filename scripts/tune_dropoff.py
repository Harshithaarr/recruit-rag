"""Optuna hyperparameter search + isotonic calibration for the drop-off model.

Run:  uv run python scripts/tune_dropoff.py [--trials 30]

What this does:
1. Runs an Optuna TPE search (default 30 trials) over XGBoost hyperparameters
   to maximise validation ROC-AUC.
2. Refits XGBoost with the best params.
3. Calibrates the predicted probabilities via isotonic regression on the
   validation set (CalibratedClassifierCV with cv='prefit').
4. Reports test-set metrics for: default model · tuned (uncalibrated) ·
   tuned + calibrated.
5. Saves both the tuned and the calibrated pipelines.

WHY also report uncalibrated tuned numbers:
- The Brier score and calibration plot of the uncalibrated model are what
  motivate the calibration step in the thesis. Comparing both makes the
  contribution explicit.

VIVA: "How much did Optuna improve AUC?"
- Whatever this script prints. No tuning of the prompt to make it sound good.
- The honest answer might be 'modest' — at our N=7000 train rows the
  irreducible-noise ceiling caps how much AUC can rise. Calibration's
  contribution shows up in Brier and calibration-error metrics.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd

from recruit.config import settings
from recruit.dropoff.schemas import FEATURE_COLUMNS, LABEL_COLUMN
from recruit.dropoff.train import (
    calibrate_pipeline,
    evaluate,
    fit_xgboost,
    save_model,
    tune_xgboost,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0",
                    help="Dataset variant suffix (must match generate_dropoff_data.py).")
    ap.add_argument("--trials", type=int, default=30, help="Optuna trial budget")
    args = ap.parse_args()

    proc_dir = settings.data_path / "processed"
    df_path = proc_dir / f"dropoff_{args.variant}.parquet"
    splits_path_v = proc_dir / f"dropoff_splits_{args.variant}.json"
    splits_path_legacy = proc_dir / "dropoff_splits.json"
    splits_path = splits_path_v if splits_path_v.exists() else splits_path_legacy
    df = pd.read_parquet(df_path)
    splits = json.loads(splits_path.read_text())
    train_df = df[df["sample_id"].isin(splits["train"])].reset_index(drop=True)
    val_df = df[df["sample_id"].isin(splits["val"])].reset_index(drop=True)
    test_df = df[df["sample_id"].isin(splits["test"])].reset_index(drop=True)
    print(f"Loaded train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ── 1. Default-hyperparameter XGBoost (the comparator) ──────────────
    print("\nFitting default XGBoost (baseline)...")
    default_model = fit_xgboost(train_df, val_df)

    # ── 2. Optuna search ────────────────────────────────────────────────
    print(f"\nRunning Optuna search ({args.trials} trials, TPE)...")
    t0 = time.time()
    best_params = tune_xgboost(train_df, val_df, n_trials=args.trials)
    t_search = time.time() - t0
    print(f"  best params (val AUC objective):")
    for k, v in best_params.items():
        print(f"    {k} = {v}")
    print(f"  search wall time: {t_search:.1f}s")

    # ── 3. Refit XGBoost with best params ───────────────────────────────
    print("\nFitting tuned XGBoost...")
    tuned_model = fit_xgboost(train_df, val_df, **best_params)

    # ── 4. Isotonic calibration on the tuned model ──────────────────────
    print("Calibrating with isotonic regression on validation set...")
    calibrated_model = calibrate_pipeline(tuned_model, val_df)

    # ── 5. Evaluate all three on val + test ─────────────────────────────
    print("\n" + "=" * 72)
    print(f"DROP-OFF MODEL COMPARISON  ·  variant: {args.variant}  ·  Optuna + isotonic")
    print("=" * 72)
    print()
    reports = [
        evaluate("default-xgb",   "val",  default_model, val_df),
        evaluate("default-xgb",   "test", default_model, test_df),
        evaluate("tuned-xgb",     "val",  tuned_model, val_df),
        evaluate("tuned-xgb",     "test", tuned_model, test_df),
        evaluate("tuned+calib",   "val",  calibrated_model, val_df),
        evaluate("tuned+calib",   "test", calibrated_model, test_df),
    ]
    for r in reports:
        print(r.as_row())

    # ── 6. Save both pipelines ──────────────────────────────────────────
    tuned_path = settings.models_path / f"dropoff_{args.variant}_tuned.joblib"
    cal_path = settings.models_path / f"dropoff_{args.variant}_calibrated.joblib"
    save_model(tuned_model, tuned_path)
    save_model(calibrated_model, cal_path)
    print(f"\nSaved tuned model      -> {tuned_path}")
    print(f"Saved calibrated model -> {cal_path}")

    # ── 7. Persist best params for the methodology chapter ──────────────
    params_path = settings.models_path / f"dropoff_{args.variant}_best_params.json"
    params_path.write_text(json.dumps({
        "best_params": best_params,
        "n_trials": args.trials,
        "search_wall_time_secs": round(t_search, 1),
        "weights_variant": args.variant,
    }, indent=2))
    print(f"Saved best params      -> {params_path}")


if __name__ == "__main__":
    main()
