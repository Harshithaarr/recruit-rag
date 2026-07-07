"""SBERT (Sentence-BERT) encoder wrapper.

WHY this file:
- Centralises model-loading so only one process loads the ~80MB MiniLM weights.
- Batches encoding so we hit GPU/CPU throughput, not per-call overhead.
- L2-normalises output vectors so cosine similarity == inner product. This
  lets FAISS use the much faster IndexFlatIP and gives identical results.

VIVA: "Why SBERT vs plain BERT?" Plain BERT's [CLS] embedding is not designed
for sentence similarity; SBERT fine-tunes BERT in a siamese network on NLI/STS
data so cosine distance becomes a meaningful semantic similarity. (Reimers &
Gurevych, EMNLP 2019.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from recruit.config import settings


class SBertEncoder:
    """Lazy-loading wrapper around a sentence-transformers model.

    The actual model loads on first encode() call so that simply importing this
    module is cheap.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.sbert_model
        self._model = None  # populated lazily

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Import inside the function so the heavy import only runs when
            # someone actually encodes something.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of strings into an (N, dim) float32 array.

        normalize=True makes inner product equal cosine similarity, which is
        what every downstream consumer (FAISS, our scoring code) expects.
        """
        self._ensure_loaded()
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


def cache_path_for(label: str) -> Path:
    """Where to write/read a cached embedding matrix.

    label is something like 'resumes' or 'jobs'. Cached as <label>.npy in
    indexes_path so we never re-embed the same corpus across runs.
    """
    settings.indexes_path.mkdir(parents=True, exist_ok=True)
    return settings.indexes_path / f"{label}.npy"


def embed_with_cache(
    encoder: SBertEncoder,
    texts: list[str],
    label: str,
    *,
    force: bool = False,
) -> np.ndarray:
    """Encode `texts`, caching to disk under `label`. Re-uses cache when present.

    WARNING: the cache key is just the label — there is no content hash. If you
    change the underlying corpus you must pass force=True (or delete the file).
    """
    path = cache_path_for(label)
    if path.exists() and not force:
        return np.load(path)
    vectors = encoder.encode(texts, show_progress_bar=True)
    np.save(path, vectors)
    return vectors
