"""Per-channel fairness audit for the retrieval stage.

Metrics:
- Skew@K — log-ratio of group share in the top-K to group share in the corpus.
  Skew@K = 0 means proportional representation. > 0 means over-represented.
- Normalised Discounted KL divergence (NDKL) — rank-weighted divergence of
  the top-K group distribution from the corpus group distribution.

WHY both:
- Skew@K is one number per group per K — interpretable, presentable.
- NDKL is one number per query — comparable across queries.

VIVA: "Why per channel, not just on the final ranked list?"
- The final list is a function of the channels. Auditing each channel
  separately tells us *where* the bias enters the pipeline — actionable
  information, not just a verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SkewResult:
    """Skew@K and NDKL for one query × one channel × one group attribute."""

    channel: str
    attribute: str  # 'gender' | 'country' | ...
    k: int
    corpus_distribution: dict[str, float]
    topk_distribution: dict[str, float]
    skew_per_group: dict[str, float]  # log(topk_share / corpus_share)
    ndkl: float                       # normalised discounted KL


def _distribution(values: Sequence[str]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {k: c / n for k, c in counts.items()}


def _skew(topk_share: float, corpus_share: float, eps: float = 1e-9) -> float:
    """Skew = log(topk_share / corpus_share). Clamped to avoid log(0)."""
    return math.log(max(topk_share, eps) / max(corpus_share, eps))


def _kl_div(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(p || q) over the union of keys."""
    total = 0.0
    for k in p.keys() | q.keys():
        p_k = p.get(k, 0.0)
        q_k = q.get(k, eps)
        if p_k <= 0:
            continue
        total += p_k * math.log(p_k / max(q_k, eps))
    return total


def skew_at_k(
    *,
    channel: str,
    attribute: str,
    retrieved_indices: Sequence[int],
    group_by_index: Sequence[str],
    k: int,
) -> SkewResult:
    """Compute Skew@K + NDKL for one (channel × attribute × k)."""
    if k <= 0 or not retrieved_indices:
        raise ValueError("k must be > 0 and retrieved_indices non-empty")

    corpus_groups = list(group_by_index)
    corpus_dist = _distribution(corpus_groups)

    topk_groups = [group_by_index[i] for i in retrieved_indices[:k]]
    topk_dist = _distribution(topk_groups)

    skew_per_group = {
        g: _skew(topk_dist.get(g, 0.0), corpus_dist.get(g, 0.0))
        for g in corpus_dist.keys() | topk_dist.keys()
    }

    # NDKL — KL(P_topk || P_corpus) discounted by rank-position weights.
    # Simplified: weight each position by 1/log2(rank+1) and accumulate
    # the prefix-distribution KL contribution.
    ndkl = 0.0
    z = 0.0
    for rank in range(1, min(k, len(retrieved_indices)) + 1):
        w = 1.0 / math.log2(rank + 1)
        z += w
        prefix = [group_by_index[i] for i in retrieved_indices[:rank]]
        p = _distribution(prefix)
        ndkl += w * _kl_div(p, corpus_dist)
    ndkl /= max(z, 1e-9)

    return SkewResult(
        channel=channel,
        attribute=attribute,
        k=k,
        corpus_distribution=corpus_dist,
        topk_distribution=topk_dist,
        skew_per_group=skew_per_group,
        ndkl=ndkl,
    )


def format_skew_table(results: list[SkewResult]) -> str:
    """Render multiple SkewResults as a text table."""
    lines = [
        f"  {'channel':<22s} {'attribute':<10s} {'K':>3s} "
        f"{'group':<10s} {'corpus':>8s} {'top-K':>8s} {'skew':>8s} {'NDKL':>8s}"
    ]
    lines.append("  " + "-" * 80)
    for r in results:
        for group in r.skew_per_group.keys():
            corpus_share = r.corpus_distribution.get(group, 0.0)
            topk_share = r.topk_distribution.get(group, 0.0)
            skew = r.skew_per_group[group]
            lines.append(
                f"  {r.channel:<22s} {r.attribute:<10s} {r.k:>3d} "
                f"{group:<10s} {corpus_share:>8.3f} {topk_share:>8.3f} "
                f"{skew:>+8.2f} {r.ndkl:>8.3f}"
            )
    return "\n".join(lines)
