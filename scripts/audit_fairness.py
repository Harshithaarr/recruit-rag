"""End-to-end fairness audit — retrieval channels + drop-off classifier.

Run:  OMP_NUM_THREADS=1 uv run python scripts/audit_fairness.py

What this produces:
1. Retrieval audit  — Skew@K and NDKL for the three channels (Dense, BM25,
   Hybrid-2) on gender and country proxies, averaged across HF queries
   that have at least one labelled positive.
2. Drop-off audit   — demographic parity, equal opportunity, disparate
   impact on the test split, by inferred gender and country.
3. Coverage report  — how many résumés the proxies could infer at all.

A Pareto curve (fairness ↔ accuracy across threshold sweeps) is added in
a future scope cut.
"""

from __future__ import annotations

import argparse
import json
import statistics

import joblib
import pandas as pd

from recruit.config import settings
from recruit.data.loaders import load_hf_fit_dataset, load_kaggle_resumes
from recruit.dropoff.schemas import FEATURE_COLUMNS, LABEL_COLUMN
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.fairness.audit_dropoff import audit_predictions, format_fairness_table
from recruit.fairness.audit_retrieval import format_skew_table, skew_at_k
from recruit.fairness.proxies import infer_country_from_text, infer_gender_from_text
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion


def _retrieval_audit() -> None:
    print("=" * 72)
    print("PART 1 — RETRIEVAL FAIRNESS AUDIT  (per channel, Skew@K + NDKL)")
    print("=" * 72)

    resumes, jobs, qrels = load_hf_fit_dataset(split="test")
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="hf_fit_test_resumes")
    dense_idx = DenseIndex(resume_vecs)
    bm25_idx = BM25Index(resume_texts)

    # Build per-résumé proxies + report coverage.
    gender = [infer_gender_from_text(r.text) or "unknown" for r in resumes]
    country = [infer_country_from_text(r.text) for r in resumes]
    n_inf_gender = sum(1 for g in gender if g != "unknown")
    n_inf_country = sum(1 for c in country if c != "other")
    print(f"\nProxy coverage on {len(resumes)} résumés:")
    print(f"  gender:  {n_inf_gender} inferred ({100 * n_inf_gender / len(resumes):.1f}%)")
    print(f"  country: {n_inf_country} inferred ({100 * n_inf_country / len(resumes):.1f}%)")

    # Only audit queries with at least one labelled positive — others have
    # no signal to compare against the corpus distribution.
    query_jobs = []
    for j in jobs:
        labels = qrels.get(j.job_id, {})
        if any(v > 0 for v in labels.values()):
            query_jobs.append(j)
    print(f"\nAuditing {len(query_jobs)} queries (those with ≥1 relevant labelled résumé).")

    K = 10
    channel_to_queries: dict[str, list[dict]] = {"dense": [], "bm25": [], "hybrid2": []}

    for job in query_jobs:
        qv = encoder.encode([job.description])[0]
        dense_hits = dense_idx.search(qv, k=50)
        bm25_hits = bm25_idx.search(job.description, k=50)
        dense_indices = [h.index for h in dense_hits]
        bm25_indices = [h.index for h in bm25_hits]
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
        hybrid_indices = [c.resume_idx for c in fused]

        channel_to_queries["dense"].append({"indices": dense_indices})
        channel_to_queries["bm25"].append({"indices": bm25_indices})
        channel_to_queries["hybrid2"].append({"indices": hybrid_indices})

    # Aggregate across queries:
    #   - top-K share: mean of per-query top-K shares
    #   - skew:        log(mean_topk_share / corpus_share)  -- NOT mean of log-ratios
    #   - NDKL:        mean of per-query NDKL  (linear scale, mean-safe)
    import math

    aggregated: list = []
    for channel, queries in channel_to_queries.items():
        for attr_name, attr_values in [("gender", gender), ("country", country)]:
            ndkl_vals: list[float] = []
            corpus_dist = None
            topk_share_accum: dict[str, list[float]] = {}

            for q in queries:
                result = skew_at_k(
                    channel=channel,
                    attribute=attr_name,
                    retrieved_indices=q["indices"],
                    group_by_index=attr_values,
                    k=K,
                )
                corpus_dist = result.corpus_distribution
                ndkl_vals.append(result.ndkl)
                for g in result.corpus_distribution.keys():
                    topk_share_accum.setdefault(g, []).append(
                        result.topk_distribution.get(g, 0.0)
                    )

            avg_topk = {g: statistics.fmean(v) for g, v in topk_share_accum.items()}
            # Skew from aggregated shares (not from averaged per-query skews).
            eps = 1e-9
            avg_skew = {
                g: math.log(max(avg_topk.get(g, 0.0), eps)
                            / max((corpus_dist or {}).get(g, 0.0), eps))
                for g in (corpus_dist or {}).keys()
            }
            avg_ndkl = statistics.fmean(ndkl_vals) if ndkl_vals else 0.0

            from recruit.fairness.audit_retrieval import SkewResult
            aggregated.append(SkewResult(
                channel=channel,
                attribute=attr_name,
                k=K,
                corpus_distribution=corpus_dist or {},
                topk_distribution=avg_topk,
                skew_per_group=avg_skew,
                ndkl=avg_ndkl,
            ))

    print(f"\nMean Skew@{K} / NDKL across queries:")
    print(format_skew_table(aggregated))
    print(
        "\nInterpretation: skew = log(top-K share / corpus share). "
        "0 ≈ proportional, |skew|>0.5 worth investigating. "
        "NDKL = rank-weighted divergence (lower is better)."
    )


def _dropoff_audit(variant: str = "v0") -> None:
    print("\n\n" + "=" * 72)
    print(f"PART 2 — DROP-OFF CLASSIFIER FAIRNESS AUDIT  (variant={variant})")
    print("=" * 72)

    proc_dir = settings.data_path / "processed"
    df = pd.read_parquet(proc_dir / f"dropoff_{variant}.parquet")
    splits_path_v = proc_dir / f"dropoff_splits_{variant}.json"
    splits_path_legacy = proc_dir / "dropoff_splits.json"
    splits_path = splits_path_v if splits_path_v.exists() else splits_path_legacy
    splits = json.loads(splits_path.read_text())
    test_df = df[df["sample_id"].isin(splits["test"])].reset_index(drop=True)

    # Re-derive proxies on the candidates referenced in the test split.
    # Pull text from both HF and Kaggle corpora so any resume_id resolves.
    hf_resumes, _, _ = load_hf_fit_dataset(split="test")
    kaggle_resumes = load_kaggle_resumes(
        settings.data_path / "raw" / "resumes_kaggle" / "Resume" / "Resume.csv"
    )
    resume_text_by_id = {r.resume_id: r.text for r in (hf_resumes + kaggle_resumes)}
    test_df = test_df.copy()
    test_df["gender"] = test_df["resume_id"].map(
        lambda rid: infer_gender_from_text(resume_text_by_id.get(rid, "")) or "unknown"
    )
    test_df["country"] = test_df["resume_id"].map(
        lambda rid: infer_country_from_text(resume_text_by_id.get(rid, ""))
    )

    # Predict with the calibrated tuned model when available; fall back to
    # the variant's default. Calibrated models only exist for v0 at present.
    cal_path = settings.models_path / f"dropoff_{variant}_calibrated.joblib"
    base_path = settings.models_path / f"dropoff_{variant}.joblib"
    model_path = cal_path if cal_path.exists() else base_path
    print(f"\nUsing model: {model_path.name}")
    model = joblib.load(model_path)
    p = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    y = test_df[LABEL_COLUMN].values

    print("\n--- Gender audit (proxy: name-based) ---")
    s_gender = audit_predictions(
        attribute="gender",
        y_true=y,
        p_predicted=p,
        group_by_row=test_df["gender"].values,
        threshold=0.5,
        min_group_n=30,
    )
    print(format_fairness_table(s_gender))

    print("\n--- Country audit (proxy: location text) ---")
    s_country = audit_predictions(
        attribute="country",
        y_true=y,
        p_predicted=p,
        group_by_row=test_df["country"].values,
        threshold=0.5,
        min_group_n=30,
    )
    print(format_fairness_table(s_country))

    # ── Persist JSON report so the UI can render this ─────────────────
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "reports" / "fairness_synthetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialise(summary):
        return {
            "attribute": summary.attribute,
            "threshold": summary.threshold,
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

    report = {
        "variant": variant,
        "model_used": model_path.name,
        "gender_proxy": _serialise(s_gender),
        "country_proxy": _serialise(s_country),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote fairness report → {out_path.relative_to(project_root)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0",
                    help="Drop-off model variant to audit (v0, v1, …).")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="Skip the retrieval fairness section.")
    args = ap.parse_args()

    if not args.skip_retrieval:
        _retrieval_audit()
    _dropoff_audit(variant=args.variant)


if __name__ == "__main__":
    main()
