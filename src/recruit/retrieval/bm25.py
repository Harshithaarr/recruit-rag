"""BM25 sparse-retrieval baseline.

WHY BM25 is the baseline (and not a strawman):
- Enterprise ATS systems including Jobvite, Workday, and Greenhouse run BM25
  over Solr or OpenSearch in production. Your committed outline names it as
  the comparator (§1, §3).
- BM25 has decades of refinement. A naive bag-of-words baseline would not be
  a credible comparator. BM25 is the bar dense retrieval has to clear.

WHY BM25 will sometimes win against SBERT:
- On rare-skill exact-match queries ("must know COBOL", "FAA cert required")
  BM25's lexical precision beats dense retrieval's tendency to overgeneralise.

WHY BM25 will sometimes lose:
- On terminology mismatch ("software engineer" in JD vs "developer" in résumé)
  BM25 has no semantic notion. SBERT does.

The thesis evaluation chapter will quantify *when* each wins, on what
query types. That comparison is the point of having a baseline.

VIVA: "How does BM25 score documents?" Briefly:
    BM25(d, q) = Σ_{t in q}  IDF(t) · TF_normalised(t, d)
Two parameters: k1 (term-frequency saturation, 1.2–2.0 typical) and
b (length normalisation, 0.75 typical). rank_bm25's BM25Okapi uses these
defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")


def tokenize(text: str) -> list[str]:
    """Lower-case word-ish tokens. Preserves things like 'C++', '.NET', 'PostgreSQL'.

    This is intentionally simple. Production BM25 would also do stemming,
    stop-word removal, and synonym expansion — which would close some of the
    terminology-mismatch gap BM25 has against dense retrieval. We are
    *deliberately* using a vanilla tokenizer so the baseline reflects what a
    typical out-of-the-box ATS does, not a hand-tuned one.
    """
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


@dataclass(frozen=True)
class SearchHit:
    """One result from a BM25 search."""

    index: int
    score: float  # BM25 raw score, NOT bounded


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi.

    Holds the tokenised corpus and provides a top-K search.
    """

    def __init__(self, documents: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._n = len(documents)
        tokens = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokens)

    def __len__(self) -> int:
        return self._n

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        # argpartition is O(n) vs argsort O(n log n); we only need top k
        k_capped = min(k, self._n)
        top_idx = np.argpartition(-scores, k_capped - 1)[:k_capped]
        # Sort the top-k itself
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [SearchHit(index=int(i), score=float(scores[i])) for i in top_idx]
