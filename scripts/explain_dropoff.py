"""SHAP global importance + sample local explanations.

Run:  uv run python scripts/explain_dropoff.py

What this does:
- Loads the trained XGBoost pipeline (models/dropoff_v0.joblib).
- Computes global SHAP importance over the test set; saves beeswarm + bar PNGs
  to reports/figures/.
- Prints 3 per-prediction local explanations in textualised form — exactly
  the format the RAG explainer prompt will consume (the SHAP→RAG bridge).
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from recruit.config import settings
from recruit.dropoff.explain import DropoffExplainer
from recruit.dropoff.schemas import LABEL_COLUMN


def main() -> None:
    proc_dir = settings.data_path / "processed"
    df = pd.read_parquet(proc_dir / "dropoff_v0.parquet")
    splits = json.loads((proc_dir / "dropoff_splits.json").read_text())
    test_df = df[df["sample_id"].isin(splits["test"])].reset_index(drop=True)
    print(f"Loaded test set: {len(test_df)} rows")

    model_path = settings.models_path / "dropoff_v0.joblib"
    pipeline = joblib.load(model_path)
    print(f"Loaded model: {model_path.name}")

    explainer = DropoffExplainer(pipeline)

    # --- Global importance ---
    print("\nGlobal SHAP importance (top-12):")
    imp = explainer.global_importance(test_df, max_samples=500)
    print(imp.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig_dir = settings.project_root / "reports" / "figures"
    paths = explainer.save_global_plots(test_df, fig_dir, max_samples=500)
    print(f"\nSaved global plots:")
    for k, p in paths.items():
        print(f"  {k:>10s}: {p}")

    # --- Local explanations — one high-risk, one moderate, one low-risk ---
    print("\n" + "=" * 70)
    print("LOCAL EXPLANATIONS  ·  SHAP→RAG bridge format")
    print("=" * 70)

    # Pick three samples spanning the predicted-probability range.
    from recruit.dropoff.schemas import FEATURE_COLUMNS

    probs = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_df = test_df.copy()
    test_df["_p"] = probs

    high_idx = test_df["_p"].idxmax()
    low_idx = test_df["_p"].idxmin()
    mid_idx = (test_df["_p"] - 0.5).abs().idxmin()

    for tag, idx in [("HIGH-RISK", high_idx), ("MODERATE", mid_idx), ("LOW-RISK", low_idx)]:
        row = test_df.loc[idx]
        explanation = explainer.explain_row(
            row,
            sample_id=str(row["sample_id"]),
            top_k=5,
        )
        print(f"\n--- {tag} — sample {row['sample_id']} (true label: {row[LABEL_COLUMN]}) ---")
        print(explanation.textualize(top_k=3))


if __name__ == "__main__":
    main()
