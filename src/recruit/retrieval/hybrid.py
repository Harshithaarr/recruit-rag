"""Hybrid ranking: combine dense + sparse + apply structured filters.

WHY hybrid?
- Dense (SBERT) captures meaning across paraphrases. BM25 captures exact terms.
- They make different mistakes, so combining them strictly dominates either
  alone on most datasets (this is the well-known "lexical + semantic" finding
  in IR — see SPLADE, ColBERTv2 discussions).

WHY Reciprocal Rank Fusion (RRF) instead of weighted-sum?
- Dense scores are cosine in [-1, 1]; BM25 scores are unbounded. Combining
  them on raw scores requires per-system normalisation that's brittle.
- RRF is rank-based — it only uses each system's *ordering*, not the scores.
  This makes it parameter-light, robust, and easy to defend in a viva.
- Formula:  RRF_score(d) = Σ_systems 1 / (k + rank_system(d))
  where k is a smoothing constant (60 is the standard from Cormack 2009).
- RRF beats weighted-sum on every TREC track it's been tested on at default k.

WHY structured filters AFTER ranking (post-filter)?
- Filtering before ranking would force us to maintain per-filter indexes.
  Post-filtering keeps the retrieval logic simple and lets the recruiter see
  the same candidate ranked across multiple JDs without re-indexing.
- For larger corpora you'd want a pre-filter — out of scope here.

VIVA: "Why these three filter dimensions?" Location, years of experience,
required certifications — these are the structured fields named explicitly
in your outline §1.
"""

from __future__ import annotations

from dataclasses import dataclass

from recruit.data.schemas import Job, Resume
from recruit.retrieval.bm25 import SearchHit as BM25Hit
from recruit.retrieval.faiss_index import SearchHit as DenseHit


@dataclass(frozen=True)
class RankedCandidate:
    """One item in the final ranked candidate list."""

    resume_idx: int
    rank: int  # 1-based
    rrf_score: float
    dense_score: float | None  # cosine similarity, if dense participated
    bm25_score: float | None  # BM25 raw score, if BM25 participated


def reciprocal_rank_fusion(
    dense_hits: list[DenseHit] | None,
    bm25_hits: list[BM25Hit] | None,
    *,
    k_smooth: int = 60,
) -> list[RankedCandidate]:
    """Combine dense and BM25 ranked lists into one RRF-fused list.

    Either list can be None / empty (then we degenerate to the other alone).
    """
    score_table: dict[int, dict[str, float]] = {}

    def _register(idx: int) -> dict[str, float]:
        if idx not in score_table:
            score_table[idx] = {"rrf": 0.0, "dense": None, "bm25": None}
        return score_table[idx]

    if dense_hits:
        for rank, hit in enumerate(dense_hits, start=1):
            entry = _register(hit.index)
            entry["rrf"] += 1.0 / (k_smooth + rank)
            entry["dense"] = hit.score

    if bm25_hits:
        for rank, hit in enumerate(bm25_hits, start=1):
            entry = _register(hit.index)
            entry["rrf"] += 1.0 / (k_smooth + rank)
            entry["bm25"] = hit.score

    # Sort by RRF descending
    ordered = sorted(score_table.items(), key=lambda kv: -kv[1]["rrf"])
    return [
        RankedCandidate(
            resume_idx=idx,
            rank=rank,
            rrf_score=entry["rrf"],
            dense_score=entry["dense"],
            bm25_score=entry["bm25"],
        )
        for rank, (idx, entry) in enumerate(ordered, start=1)
    ]


def rrf_fuse_indices(
    rankings: list[list[int]],
    *,
    k_smooth: int = 60,
    weights: list[float] | None = None,
) -> list[int]:
    """N-channel RRF fusion over plain index lists.

    Each `rankings[i]` is a list of corpus indices in retrieved order (best
    first). Returns one fused index list in descending RRF score.

    If `weights` is given, channel i's contribution is multiplied by weights[i].
    Default (weights=None) is uniform — equivalent to Cormack 2009 RRF.

    WHY weighted RRF exists:
    - Equal-weight RRF assumes all channels are equally informative. When one
      channel is much noisier (e.g. a rules-based extractor on raw text), it
      drags down the stronger channels. Weighted RRF lets a noisy channel
      still *promote* a candidate without dominating the ranking.
    - The weights are a hyperparameter swept during evaluation.

    WHY this companion to `reciprocal_rank_fusion`:
    - The two-channel function above preserves per-channel scores for the
      UI. For evaluation with N≥3 channels we only need the fused ordering,
      and an N-ary signature is cleaner than nesting two-channel calls.
    - Empty channel lists are tolerated and contribute nothing — useful for
      ablations where one channel is intentionally disabled.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} != rankings length {len(rankings)}"
        )

    scores: dict[int, float] = {}
    for i, ranking in enumerate(rankings):
        if not ranking:
            continue
        w = weights[i] if weights is not None else 1.0
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (k_smooth + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def rerank_by_trajectory(
    candidate_indices: list[int],
    trajectory_scores: list[float],
    *,
    beta: float = 0.7,
) -> list[int]:
    """Re-rank a candidate pool by combining their fused-rank position with
    a per-candidate trajectory score.

    final_score(i) = beta * (1 / rank_in_pool(i))  +  (1 - beta) * trajectory_scores[i]

    `candidate_indices` is the fused pool (top-N by Dense + BM25). The
    `trajectory_scores[i]` is the experience-channel score for the résumé
    at corpus index `candidate_indices[i]`. Both inputs must be aligned.

    WHY beta = 0.7 by default:
    - Skill-fit retrieval (Dense + BM25) is the dominant signal in the
      labelled benchmark. Trajectory adds at the margin.
    - 0.7 keeps the original fused ordering as the backbone with trajectory
      perturbing positions of nearby-ranked candidates.
    - Final value is a hyperparameter swept on validation NDCG@10.

    WHY 1/rank rather than the raw RRF score:
    - The two terms must be on similar scales. 1/rank is in [1/N, 1.0] for a
      pool of N. Trajectory scores are already in [0, 1]. RRF scores are in
      [0, ~0.05] for k_smooth=60 — would underweight skill-fit if combined
      directly.
    """
    if len(candidate_indices) != len(trajectory_scores):
        raise ValueError(
            f"candidate_indices ({len(candidate_indices)}) and "
            f"trajectory_scores ({len(trajectory_scores)}) length mismatch"
        )
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    scored: list[tuple[int, float]] = []
    for pos, (idx, traj) in enumerate(
        zip(candidate_indices, trajectory_scores), start=1
    ):
        score = beta * (1.0 / pos) + (1.0 - beta) * traj
        scored.append((idx, score))

    scored.sort(key=lambda kv: -kv[1])
    return [idx for idx, _ in scored]


def passes_structured_filters(resume: Resume, job: Job) -> bool:
    """Hard filter: drop résumés that fail any structured requirement.

    Currently checks:
    - Minimum years of experience (resume.years_experience >= job.min_years_experience)
    - Location: if both are set, must match (case-insensitive substring)
    - Required certifications: every cert required by the job must appear in
      the résumé's skills list (case-insensitive).

    Missing (None) fields are treated as "unknown — let it pass" rather than
    "fail". Trade-off: more recall, less precision. Documented choice.
    """
    if (
        job.min_years_experience is not None
        and resume.years_experience is not None
        and resume.years_experience < job.min_years_experience
    ):
        return False

    if job.location and resume.location:
        if job.location.strip().lower() not in resume.location.strip().lower():
            if resume.location.strip().lower() not in job.location.strip().lower():
                return False

    if job.required_certifications:
        resume_skills_lower = {s.strip().lower() for s in resume.skills}
        for cert in job.required_certifications:
            if cert.strip().lower() not in resume_skills_lower:
                return False

    return True


def filter_candidates(
    candidates: list[RankedCandidate],
    resumes: list[Resume],
    job: Job,
) -> list[RankedCandidate]:
    """Apply structured filters and re-number the remaining ranks."""
    kept = [c for c in candidates if passes_structured_filters(resumes[c.resume_idx], job)]
    return [
        RankedCandidate(
            resume_idx=c.resume_idx,
            rank=new_rank,
            rrf_score=c.rrf_score,
            dense_score=c.dense_score,
            bm25_score=c.bm25_score,
        )
        for new_rank, c in enumerate(kept, start=1)
    ]
