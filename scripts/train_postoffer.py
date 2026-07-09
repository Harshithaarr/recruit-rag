"""CLI: train + tune + calibrate the post-offer decline classifier.

Usage:
    # Full run — Optuna tune (20 trials) + calibrate:
    uv run python scripts/train_postoffer.py

    # Fast dev run — skip tuning, use defaults:
    uv run python scripts/train_postoffer.py --no-tune

Outputs:
    models/postoffer_v1.joblib             — uncalibrated XGBoost
    models/postoffer_v1_calibrated.joblib  — with isotonic calibration
    models/postoffer_v1_best_params.json   — Optuna best trial (if tuned)
    models/postoffer_v1_metrics.json       — evaluation metrics per split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recruit.config import settings
from recruit.postoffer.train import (
    calibrate_pipeline,
    evaluate,
    fit_xgboost,
    make_splits,
    save_model,
    tune_xgboost,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_path / "processed" / "postoffer_v1.parquet",
        help="Input parquet dataset (from `make generate-postoffer-data`).",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=settings.models_path / "postoffer_v1.joblib",
    )
    parser.add_argument(
        "--out-calibrated",
        type=Path,
        default=settings.models_path / "postoffer_v1_calibrated.joblib",
    )
    parser.add_argument(
        "--out-params",
        type=Path,
        default=settings.models_path / "postoffer_v1_best_params.json",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=settings.models_path / "postoffer_v1_metrics.json",
    )
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing dataset {args.data}. Run `make generate-postoffer-data` first."
        )

    print(f"Loading {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} rows · decline rate {df['declined_offer'].mean():.3f}")

    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print(f"  splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ── Tune (or use defaults) ────────────────────────────────────────
    if args.no_tune:
        print("\nSkipping tuning — using default hyperparameters.")
        best_params: dict = {}
    else:
        print(f"\nOptuna TPE search — {args.n_trials} trials...")
        best_params = tune_xgboost(
            train_df, val_df, n_trials=args.n_trials, seed=args.seed
        )
        print("Best trial parameters:")
        for k, v in best_params.items():
            print(f"  {k:<20s} = {v}")
        args.out_params.parent.mkdir(parents=True, exist_ok=True)
        args.out_params.write_text(json.dumps(best_params, indent=2))
        print(f"  → wrote {args.out_params}")

    # ── Fit final XGBoost with tuned params ───────────────────────────
    print("\nTraining XGBoost with tuned parameters...")
    model = fit_xgboost(train_df, val_df, seed=args.seed, **best_params)

    # ── Calibrate on validation ───────────────────────────────────────
    print("Fitting isotonic calibration on validation set...")
    calibrated = calibrate_pipeline(model, val_df)

    # ── Evaluate both, on all splits ──────────────────────────────────
    print("\nEvaluation:")
    reports = []
    for name, m in [("xgboost", model), ("xgboost+isotonic", calibrated)]:
        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            r = evaluate(name, split_name, m, split_df)
            reports.append(r)
            print(r.as_row())

    # ── Persist ───────────────────────────────────────────────────────
    save_model(model, args.out_model)
    save_model(calibrated, args.out_calibrated)
    print(f"\nSaved:  {args.out_model}")
    print(f"        {args.out_calibrated}")

    metrics_dict = [
        {
            "name": r.name,
            "split": r.split,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "brier": r.brier,
            "f1_at_0_5": r.f1_at_0_5,
            "n": r.n,
        }
        for r in reports
    ]
    args.out_metrics.write_text(json.dumps(metrics_dict, indent=2))
    print(f"        {args.out_metrics}")


if __name__ == "__main__":
    main()
