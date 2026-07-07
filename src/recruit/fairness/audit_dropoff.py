"""Fairness audit for the drop-off classifier.

Computes group-level metrics expected in a hiring-AI fairness chapter:
- Demographic parity difference  — does P(y_hat=1) differ across groups?
- Equal-opportunity difference   — does TPR differ across groups?
- Disparate-impact ratio         — ratio of selection rates across groups.

VIVA: "Why these three metrics specifically?"
- Demographic parity is the headline number examiners expect.
- Equal opportunity speaks to fairness of true-positive treatment — Hardt
  et al. 2016.
- Disparate impact is the legal/regulatory threshold (4/5 rule, EEOC).
  Reporting all three covers complementary failure modes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupReport:
    """Per-group counts + base rates for one binary-prediction audit."""

    group: str
    n: int
    base_rate: float        # share of group = 1 in TRUE labels
    selection_rate: float   # share predicted = 1
    tpr: float              # P(y_hat=1 | y=1, group)
    fpr: float              # P(y_hat=1 | y=0, group)


@dataclass(frozen=True)
class FairnessSummary:
    """Aggregated fairness metrics across all groups."""

    attribute: str
    threshold: float
    groups: list[GroupReport]
    demographic_parity_diff: float   # max selection_rate - min selection_rate
    equal_opportunity_diff: float    # max TPR - min TPR
    disparate_impact_ratio: float    # min selection_rate / max selection_rate

    def passes_4_5_rule(self) -> bool:
        """EEOC's 4/5 rule: DI ratio ≥ 0.8 is the conventional pass."""
        return self.disparate_impact_ratio >= 0.8


def _safe_rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def audit_predictions(
    *,
    attribute: str,
    y_true: Sequence[int],
    p_predicted: Sequence[float],
    group_by_row: Sequence[str],
    threshold: float = 0.5,
    min_group_n: int = 30,
) -> FairnessSummary:
    """Compute parity / equal-opportunity / disparate-impact for one attribute.

    Groups with fewer than `min_group_n` examples are dropped (statistics
    too unstable). Drop is reported via `len(summary.groups) < len(unique groups)`.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_predicted)
    g = np.asarray(group_by_row)
    if not (len(y) == len(p) == len(g)):
        raise ValueError("y_true, p_predicted, group_by_row must align in length")

    y_hat = (p >= threshold).astype(int)

    reports: list[GroupReport] = []
    for group in sorted(set(g)):
        mask = g == group
        n_g = int(mask.sum())
        if n_g < min_group_n:
            continue
        y_g = y[mask]
        yh_g = y_hat[mask]

        n_pos = int((y_g == 1).sum())
        n_neg = int((y_g == 0).sum())

        sel_rate = float(yh_g.mean())
        tpr = _safe_rate(int(((yh_g == 1) & (y_g == 1)).sum()), n_pos)
        fpr = _safe_rate(int(((yh_g == 1) & (y_g == 0)).sum()), n_neg)
        reports.append(GroupReport(
            group=str(group),
            n=n_g,
            base_rate=float(y_g.mean()),
            selection_rate=sel_rate,
            tpr=tpr,
            fpr=fpr,
        ))

    sel_rates = [r.selection_rate for r in reports]
    tprs = [r.tpr for r in reports]
    if sel_rates:
        dp_diff = max(sel_rates) - min(sel_rates)
        di_ratio = (min(sel_rates) / max(sel_rates)) if max(sel_rates) > 0 else 0.0
    else:
        dp_diff = 0.0
        di_ratio = 1.0
    eo_diff = max(tprs) - min(tprs) if tprs else 0.0

    return FairnessSummary(
        attribute=attribute,
        threshold=threshold,
        groups=reports,
        demographic_parity_diff=dp_diff,
        equal_opportunity_diff=eo_diff,
        disparate_impact_ratio=di_ratio,
    )


def format_fairness_table(summary: FairnessSummary) -> str:
    """Render a FairnessSummary as text."""
    lines = [
        f"  attribute='{summary.attribute}'   threshold={summary.threshold:.2f}",
        f"  {'group':<10s} {'n':>6s} {'base':>7s} {'sel_rate':>9s} "
        f"{'TPR':>7s} {'FPR':>7s}",
        "  " + "-" * 60,
    ]
    for g in summary.groups:
        lines.append(
            f"  {g.group:<10s} {g.n:>6d} {g.base_rate:>7.3f} "
            f"{g.selection_rate:>9.3f} {g.tpr:>7.3f} {g.fpr:>7.3f}"
        )
    lines.append("  " + "-" * 60)
    lines.append(
        f"  Demographic parity diff (max - min selection rate): "
        f"{summary.demographic_parity_diff:.3f}"
    )
    lines.append(
        f"  Equal opportunity diff (max - min TPR):             "
        f"{summary.equal_opportunity_diff:.3f}"
    )
    lines.append(
        f"  Disparate impact ratio (min / max selection rate):  "
        f"{summary.disparate_impact_ratio:.3f}  "
        f"({'PASS' if summary.passes_4_5_rule() else 'FAIL'} 4/5 rule)"
    )
    return "\n".join(lines)
