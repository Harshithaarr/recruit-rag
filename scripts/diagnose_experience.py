"""Diagnose whether the experience layer actually does what we asked it to.

The question is NOT 'does it produce good benchmark numbers' (that we know).
The question is: does the trajectory extractor correctly surface the
'strong-but-keyword-poor candidate with deep career history' — the original
use-case motivating the channel?

This script:
1. Inspects the distribution of trajectory features across the HF corpus —
   does the extractor produce signal, or is everything 'unknown'?
2. Picks a senior-engineering JD and contrasts what each channel surfaces.
3. Picks a Tier-1-company candidate and checks whether the experience channel
   actually ranks them higher than skill-fit channels do.
"""

from __future__ import annotations

from collections import Counter

from recruit.data.loaders import load_hf_fit_dataset
from recruit.data.schemas import Job
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.experience import (
    ExperienceIndex,
    Seniority,
    extract_job_criteria,
    extract_resume_trajectory,
)
from recruit.retrieval.faiss_index import DenseIndex


def main() -> None:
    resumes, jobs, qrels = load_hf_fit_dataset(split="test")

    print("=" * 70)
    print("1. TRAJECTORY FEATURE DISTRIBUTION ON HF CORPUS")
    print("=" * 70)
    trajs = [extract_resume_trajectory(r) for r in resumes]

    seniority_counts = Counter(t.seniority.name for t in trajs)
    domain_counts = Counter(t.domain for t in trajs)
    tier_counts = Counter(t.company_tier for t in trajs)
    yoe_known = sum(1 for t in trajs if t.years_experience is not None)
    yoe_values = [t.years_experience for t in trajs if t.years_experience is not None]

    print(f"\nCorpus size: {len(trajs)} résumés\n")

    print("Seniority distribution:")
    for level in Seniority:
        n = seniority_counts.get(level.name, 0)
        print(f"  {level.name:>10s} : {n:>4d}  ({100 * n / len(trajs):5.1f}%)")

    print("\nDomain distribution:")
    for dom, n in domain_counts.most_common():
        print(f"  {dom:>10s} : {n:>4d}  ({100 * n / len(trajs):5.1f}%)")

    print("\nCompany Tier-1 detected:")
    for tier, n in sorted(tier_counts.items()):
        label = "Tier-1" if tier == 1 else "other"
        print(f"  {label:>10s} : {n:>4d}  ({100 * n / len(trajs):5.1f}%)")

    print(f"\nYOE detection:  {yoe_known}/{len(trajs)} résumés ({100 * yoe_known / len(trajs):.1f}%)")
    if yoe_values:
        print(f"  min  : {min(yoe_values):.1f}")
        print(f"  median: {sorted(yoe_values)[len(yoe_values) // 2]:.1f}")
        print(f"  max  : {max(yoe_values):.1f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. WHO DOES EACH CHANNEL SURFACE — case study")
    print("=" * 70)

    # Pick a JD that asks for senior + a specific domain.
    # Try to find one with multiple labelled positives and a clear seniority ask.
    target_jd = None
    for j in jobs:
        crit = extract_job_criteria(j)
        if crit.target_seniority.value >= Seniority.SENIOR.value:
            if j.job_id in qrels and len(qrels[j.job_id]) >= 5:
                target_jd = j
                break
    if target_jd is None:
        target_jd = jobs[0]

    crit = extract_job_criteria(target_jd)
    print(f"\nJD: {target_jd.job_id}")
    print(f"   Detected criteria — min_yoe={crit.min_yoe}, "
          f"seniority={crit.target_seniority.name}, domain={crit.target_domain}")
    print(f"   First 200 chars: {target_jd.description[:200]}...")

    # Encode corpus
    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(encoder, resume_texts, label="hf_fit_test_resumes")

    dense_idx = DenseIndex(resume_vecs)
    bm25_idx = BM25Index(resume_texts)
    exp_idx = ExperienceIndex(resumes)

    qv = encoder.encode([target_jd.description])[0]
    dense_top = dense_idx.search(qv, k=5)
    bm25_top = bm25_idx.search(target_jd.description, k=5)
    exp_top = exp_idx.search(target_jd, k=5)

    def describe(idx: int) -> str:
        t = trajs[idx]
        text_preview = resumes[idx].text[:80].replace("\n", " ")
        label = qrels.get(target_jd.job_id, {}).get(resumes[idx].resume_id, "—")
        grade_map = {2.0: "GOOD", 1.0: "POTENTIAL", 0.0: "NO_FIT"}
        gtxt = grade_map.get(label, "unlabelled") if label != "—" else "unlabelled"
        return (
            f"  idx={idx:>3d}  yoe={t.years_experience}  "
            f"sen={t.seniority.name:<8s}  dom={t.domain:<8s}  tier={t.company_tier}  "
            f"label={gtxt:<10s}\n      text: {text_preview}..."
        )

    print("\nDense top-5:")
    for h in dense_top:
        print(describe(h.index))
    print("\nBM25 top-5:")
    for h in bm25_top:
        print(describe(h.index))
    print("\nExperience top-5:")
    for h in exp_top:
        print(describe(h.index))

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DOES EXPERIENCE SURFACE TIER-1 / HIGH-YOE CANDIDATES?")
    print("=" * 70)

    # Find résumés with high YOE AND Tier-1 — the original use-case prototype.
    high_career = [
        (i, t) for i, t in enumerate(trajs)
        if t.years_experience is not None and t.years_experience >= 8 and t.company_tier == 1
    ]
    print(f"\nCandidates with YOE ≥ 8 AND Tier-1 in text: {len(high_career)}")
    for i, t in high_career[:5]:
        rank_in_exp = next(
            (r + 1 for r, h in enumerate(exp_idx.search(target_jd, k=len(resumes)))
             if h.index == i),
            None,
        )
        rank_in_dense = next(
            (r + 1 for r, h in enumerate(dense_idx.search(qv, k=len(resumes)))
             if h.index == i),
            None,
        )
        print(
            f"  idx={i:>3d}  yoe={t.years_experience:>4.1f}  "
            f"sen={t.seniority.name:<8s}  dom={t.domain:<8s}  "
            f"  rank_exp={rank_in_exp:>3d}  rank_dense={rank_in_dense:>3d}"
        )


if __name__ == "__main__":
    main()
