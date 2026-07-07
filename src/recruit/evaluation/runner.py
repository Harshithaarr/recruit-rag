"""Evaluation harness — run any retriever over a query set, report metrics.

WHY this lives apart from the metrics module:
- ir_metrics.py is pure math. This module is orchestration: it loads qrels,
  drives the retriever, and assembles per-K result tables.
- That separation means you can swap retrievers (dense / BM25 / hybrid /
  hybrid + experience channel) without touching the metric definitions.

WHY a callable interface, not a class hierarchy:
- A retriever is just `f(query_text, k) -> list[int]`. Anything with that shape
  plugs in — including future learned rerankers. No inheritance ceremony.

VIVA: "How do you know your harness is right?"
- The metric functions have docstring examples (see ir_metrics.py) you can hand-verify.
- The smoke-test in scripts/eval_retrieval.py exercises all four metrics on
  a tiny labelled set where the expected ordering is obvious.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from recruit.evaluation.ir_metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# A retriever takes a query string and a cap k, returns a ranked list of
# corpus-internal indices (0-based positions into the résumé list).
Retriever = Callable[[str, int], list[int]]


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query.

    query_id: stable identifier (e.g. job_id).
    query_text: the text passed to the retriever (e.g. JD description).
    relevance: map from corpus-internal index → graded relevance (0, 1, 2).
    """

    query_id: str
    query_text: str
    relevance: Mapping[int, float]


@dataclass(frozen=True)
class QueryResult:
    """Per-query metric values at one K."""

    query_id: str
    k: int
    precision: float
    recall: float
    ndcg: float
    rr: float


@dataclass(frozen=True)
class AggregateResult:
    """Mean metrics over a query set at one K."""

    k: int
    mean_precision: float
    mean_recall: float
    mean_ndcg: float
    mrr: float
    n_queries: int


def evaluate_retriever(
    queries: list[EvalQuery],
    retriever: Retriever,
    ks: list[int],
    *,
    max_k_for_retrieval: int | None = None,
) -> tuple[list[QueryResult], list[AggregateResult]]:
    """Run one retriever over a query set, return per-query and aggregate metrics.

    The retriever is called once per query, asked for `max_k_for_retrieval` (or
    max(ks)) hits, and we slice for each K. This avoids re-running retrieval
    per K — most retrievers are deterministic and the top-K-prefix property
    holds.
    """
    if not queries:
        raise ValueError("queries must not be empty")
    if not ks:
        raise ValueError("ks must not be empty")

    fetch_k = max_k_for_retrieval or max(ks)

    per_query: list[QueryResult] = []
    for query in queries:
        retrieved = retriever(query.query_text, fetch_k)
        for k in ks:
            per_query.append(
                QueryResult(
                    query_id=query.query_id,
                    k=k,
                    precision=precision_at_k(retrieved, query.relevance, k),
                    recall=recall_at_k(retrieved, query.relevance, k),
                    ndcg=ndcg_at_k(retrieved, query.relevance, k),
                    # RR doesn't depend on K — but reported here for completeness.
                    rr=reciprocal_rank(retrieved, query.relevance),
                )
            )

    aggregates: list[AggregateResult] = []
    for k in ks:
        rows = [r for r in per_query if r.k == k]
        aggregates.append(
            AggregateResult(
                k=k,
                mean_precision=mean(r.precision for r in rows),
                mean_recall=mean(r.recall for r in rows),
                mean_ndcg=mean(r.ndcg for r in rows),
                mrr=mean(r.rr for r in rows),
                n_queries=len(rows),
            )
        )
    return per_query, aggregates


def load_qrels(
    qrels_path: Path,
    *,
    resume_id_to_index: Mapping[str, int],
    job_id_to_text: Mapping[str, str],
) -> list[EvalQuery]:
    """Load a JSON qrels file into EvalQuery objects.

    Expected JSON shape:
        {
          "J002": {"R002": 2, "R006": 2, "R007": 1},
          "J003": {"R003": 2, "R005": 1},
          ...
        }

    Skips queries whose job_id is not in job_id_to_text (i.e. job missing from
    the loaded corpus) and skips relevance entries pointing at résumés not in
    resume_id_to_index. Both cases are logged via the return-value count; the
    caller can compare len(returned) vs len(json keys) to detect mismatch.
    """
    with qrels_path.open() as f:
        raw = json.load(f)

    out: list[EvalQuery] = []
    for job_id, resume_grades in raw.items():
        if job_id not in job_id_to_text:
            continue
        relevance: dict[int, float] = {}
        for resume_id, grade in resume_grades.items():
            if resume_id not in resume_id_to_index:
                continue
            relevance[resume_id_to_index[resume_id]] = float(grade)
        if not relevance:
            # No usable labels for this query — skip rather than report zeros
            # on a query the matcher had no chance with.
            continue
        out.append(
            EvalQuery(
                query_id=job_id,
                query_text=job_id_to_text[job_id],
                relevance=relevance,
            )
        )
    return out


def queries_from_qrels(
    qrels: Mapping[str, Mapping[str, float]],
    *,
    resume_id_to_index: Mapping[str, int],
    job_id_to_text: Mapping[str, str],
    min_relevant_per_query: int = 1,
) -> list[EvalQuery]:
    """Build EvalQuery objects from an in-memory qrels dict.

    Sibling of `load_qrels` for when qrels are produced programmatically
    (e.g. from a HF dataset) rather than read from a JSON file.

    Drops queries with fewer than `min_relevant_per_query` items at grade > 0 —
    a query with zero relevant items can't measure precision or recall.
    """
    out: list[EvalQuery] = []
    for job_id, resume_grades in qrels.items():
        if job_id not in job_id_to_text:
            continue
        relevance: dict[int, float] = {}
        for resume_id, grade in resume_grades.items():
            if resume_id not in resume_id_to_index:
                continue
            relevance[resume_id_to_index[resume_id]] = float(grade)
        n_relevant = sum(1 for v in relevance.values() if v > 0)
        if n_relevant < min_relevant_per_query:
            continue
        out.append(
            EvalQuery(
                query_id=job_id,
                query_text=job_id_to_text[job_id],
                relevance=relevance,
            )
        )
    return out


def format_aggregate_table(
    label: str,
    aggregates: list[AggregateResult],
) -> str:
    """Pretty-print a single retriever's aggregate metrics as a text table."""
    lines = [
        f"  {label}",
        f"  {'-' * (len(label))}",
        f"  {'K':>4}  {'P@K':>7}  {'R@K':>7}  {'NDCG@K':>8}  {'MRR':>7}  {'n':>4}",
    ]
    for agg in aggregates:
        lines.append(
            f"  {agg.k:>4}  "
            f"{agg.mean_precision:>7.3f}  "
            f"{agg.mean_recall:>7.3f}  "
            f"{agg.mean_ndcg:>8.3f}  "
            f"{agg.mrr:>7.3f}  "
            f"{agg.n_queries:>4}"
        )
    return "\n".join(lines)
