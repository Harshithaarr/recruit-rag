"""FAISS Flat-index wrapper for dense semantic retrieval.

WHY FAISS Flat (and not IVF / HNSW)?
- Our corpus is small (< 100k records). A flat brute-force index gives exact
  results in milliseconds at this scale.
- IVF and HNSW only pay off above ~1M vectors, and they trade recall for speed.
- Reproducibility: a flat index is deterministic — no training step, no
  partitioning hyperparameters to defend in the viva.

WHY IndexFlatIP and not IndexFlatL2?
- We normalise embeddings to unit length in SBertEncoder.encode().
- Inner product on unit vectors equals cosine similarity.
- IndexFlatIP is slightly faster than IndexFlatL2 because it skips the subtract
  step in distance computation.

VIVA: "Why FAISS?" Local, free, well-documented, reproducible. Pgvector is the
production-equivalent (and what a Jobvite integration would use); the trade-off
is discussed in the limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    """One result from a dense search."""

    index: int  # position in the original corpus
    score: float  # cosine similarity in [-1, 1]


class DenseIndex:
    """Thin wrapper around faiss.IndexFlatIP.

    Stores the corpus vectors and provides a top-K search.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        # faiss is imported inside __init__ so the module is cheap to import
        # in environments where matcher deps aren't installed (e.g. UI-only).
        import faiss

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")

        self._dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)
        self._n = vectors.shape[0]

    def __len__(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchHit]:
        """Return the top-k most similar items to a single query vector.

        query: shape (dim,) or (1, dim) — assumed already L2-normalised.
        """
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._dim:
            raise ValueError(
                f"Query dim {query.shape[1]} does not match index dim {self._dim}"
            )

        k_capped = min(k, self._n)
        scores, indices = self._index.search(query, k_capped)
        # scores/indices are shape (1, k); pull out row 0
        return [
            SearchHit(index=int(idx), score=float(scr))
            for idx, scr in zip(indices[0], scores[0])
            if idx != -1  # FAISS uses -1 to mean "no result"
        ]
