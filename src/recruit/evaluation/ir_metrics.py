"""Information-retrieval metrics — pure functions over rank lists.

WHY pure functions:
- These are the *measurement* tools. They must not depend on anything
  domain-specific (résumés, jobs). They take rank lists and relevance maps
  and return numbers. That separation lets the same harness evaluate any
  retriever — current matcher, future experience-channel, drop-off-aware
  hybrid — without changes here.

WHY graded relevance (not binary):
- Two candidates can both be "relevant" but at different strengths. A perfect
  fit and a stretch fit should not contribute the same NDCG weight.
- Binary relevance forces a synthetic threshold and throws away signal.
- Convention used here:  2 = highly relevant,  1 = relevant,  0 = not relevant.

VIVA: "Why these four metrics?"
- Precision@K  — fraction of top-K that are relevant (what the recruiter sees).
- Recall@K    — fraction of all relevant items the top-K covered (did we miss any?).
- NDCG@K      — rewards getting the *most* relevant items at the *top* (rank-aware).
- MRR         — how high up the FIRST relevant item appears (one-shot quality).
- Together they cover precision, recall, ranking quality, and first-hit quality.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def precision_at_k(retrieved: list[int], relevance: Mapping[int, float], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant (relevance > 0).

    Example: retrieved=[5, 2, 7, 1], relevance={5:2, 1:1}, k=4 → 2/4 = 0.5
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for idx in top_k if relevance.get(idx, 0.0) > 0)
    return hits / k


def recall_at_k(retrieved: list[int], relevance: Mapping[int, float], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k.

    Example: retrieved=[5, 2, 7, 1], relevance={5:2, 1:1, 9:2}, k=4 → 2/3
    """
    if k <= 0:
        raise ValueError("k must be positive")
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0:
        # Defensive: a query with no relevant items has undefined recall.
        # Return 0.0 and let the caller decide whether to drop the query.
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for idx in top_k if relevance.get(idx, 0.0) > 0)
    return hits / total_relevant


def dcg_at_k(retrieved: list[int], relevance: Mapping[int, float], k: int) -> float:
    """Discounted Cumulative Gain at K.

    DCG@K = Σ_{i=1..k}  (2^rel_i - 1) / log2(i + 1)

    The (2^rel - 1) form is the standard "exponential gain" used by TREC.
    Plain rel_i would underweight strongly-relevant items.
    """
    score = 0.0
    for i, idx in enumerate(retrieved[:k], start=1):
        rel = relevance.get(idx, 0.0)
        if rel <= 0:
            continue
        score += (2.0**rel - 1.0) / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[int], relevance: Mapping[int, float], k: int) -> float:
    """Normalised DCG@K. NDCG ∈ [0, 1].

    NDCG = DCG(retrieved) / DCG(ideal ordering)

    The ideal ordering sorts items by their relevance descending.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = dcg_at_k(retrieved, relevance, k)
    # Ideal ranking: take the top-k by relevance descending.
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    ideal = 0.0
    for i, rel in enumerate(ideal_order, start=1):
        if rel <= 0:
            continue
        ideal += (2.0**rel - 1.0) / math.log2(i + 1)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def reciprocal_rank(retrieved: list[int], relevance: Mapping[int, float]) -> float:
    """Reciprocal rank of the FIRST relevant item; 0 if none found.

    MRR (mean reciprocal rank) over a query set is the mean of this.
    """
    for i, idx in enumerate(retrieved, start=1):
        if relevance.get(idx, 0.0) > 0:
            return 1.0 / i
    return 0.0


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; returns 0.0 on empty input (defensive)."""
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
