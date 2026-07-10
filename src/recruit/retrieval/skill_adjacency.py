"""Skill-gap adjacency map — decompose skill match into matched / adjacent / missing.

End-sem extension. The mid-sem system reports skill match as binary — either
the candidate has "React" verbatim or doesn't. This misses the recruiter's
most important judgement: **transferable skills**. A candidate who has "Vue"
when the JD asks "React" isn't a full match, but they're not a rejection
either — they're 80% of the way there in the embedding space.

This module scores every required skill against every candidate skill in
the SBERT embedding space and classifies each pair as matched / adjacent /
missing based on a cosine-similarity threshold.

WHY novel:
- Turns SBERT retrieval into an interpretable per-skill map that a recruiter
  can inspect. The overall similarity score becomes a decomposition — "here's
  the specific skills where the candidate is transferable, not missing."
- Reuses your existing embedding infrastructure — no new model training.

VIVA: "Why 0.55 as the adjacency threshold?"
- Above ~0.75: usually the same skill spelt differently ("Postgres" vs
  "PostgreSQL") or very close variants ("React" vs "ReactJS").
- Between 0.55 and 0.75: genuinely transferable ("Vue" ~ "React", "Go" ~
  "Rust", "TensorFlow" ~ "PyTorch").
- Below 0.55: too far apart to claim transfer.
- Threshold is exposed as a parameter — reviewers can retune if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# Category verdict for one (required skill, candidate skill list) pair.
_CAT_MATCHED = "matched"
_CAT_ADJACENT = "adjacent"
_CAT_MISSING = "missing"


@dataclass(frozen=True)
class SkillMatch:
    """One required skill's coverage assessment for one candidate."""

    required_skill: str
    category: str  # matched / adjacent / missing
    best_candidate_skill: str | None  # the closest candidate skill (or None)
    similarity: float  # cosine similarity to best_candidate_skill (0.0 if missing)


@dataclass(frozen=True)
class SkillAdjacencyReport:
    """Full skill-adjacency map for one (candidate, JD) pair."""

    matches: list[SkillMatch]

    @property
    def n_matched(self) -> int:
        return sum(1 for m in self.matches if m.category == _CAT_MATCHED)

    @property
    def n_adjacent(self) -> int:
        return sum(1 for m in self.matches if m.category == _CAT_ADJACENT)

    @property
    def n_missing(self) -> int:
        return sum(1 for m in self.matches if m.category == _CAT_MISSING)

    @property
    def coverage_score(self) -> float:
        """Weighted skill coverage — matched=1.0, adjacent=0.6, missing=0.0.

        Result in [0, 1]. Weighting reflects the recruiter's judgement:
        adjacent skills are meaningful but not equivalent to a direct match.
        """
        if not self.matches:
            return 0.0
        total = sum(
            {_CAT_MATCHED: 1.0, _CAT_ADJACENT: 0.6, _CAT_MISSING: 0.0}[m.category]
            for m in self.matches
        )
        return total / len(self.matches)


# ─── Skill embedding cache ─────────────────────────────────────────────
# Skills are short strings — embedding is cheap. We cache per-process
# using lru_cache so repeated queries during a Streamlit session share
# the computation.

_SKILL_ENCODER = None


def _get_encoder():
    """Lazy-load the SBERT encoder — same one used for résumé retrieval."""
    global _SKILL_ENCODER
    if _SKILL_ENCODER is None:
        from recruit.embeddings.sbert import SBertEncoder
        _SKILL_ENCODER = SBertEncoder()
    return _SKILL_ENCODER


@lru_cache(maxsize=2048)
def _embed_skill(skill: str) -> tuple:
    """Return the (immutable tuple) SBERT vector for one skill string."""
    encoder = _get_encoder()
    vec = encoder.encode([skill.lower()])[0]
    return tuple(float(x) for x in vec)


def _skill_vector(skill: str) -> np.ndarray:
    return np.asarray(_embed_skill(skill), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — SBERT vectors are L2-normalised so this is safe."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Public API ────────────────────────────────────────────────────────


def build_skill_adjacency_report(
    required_skills: list[str],
    candidate_skills: list[str],
    *,
    match_threshold: float = 0.85,
    adjacent_threshold: float = 0.55,
) -> SkillAdjacencyReport:
    """Classify every required skill as matched / adjacent / missing.

    For each required skill:
      1. Compute cosine similarity to every candidate skill.
      2. Find the best (candidate skill, similarity) pair.
      3. Categorise:
         · sim >= match_threshold    → matched
         · sim >= adjacent_threshold → adjacent
         · otherwise                 → missing

    Also detects verbatim string matches (case-insensitive) as a shortcut
    without needing to embed — most skill lists have direct overlaps and
    we save the embedding cost.
    """
    if not required_skills:
        return SkillAdjacencyReport(matches=[])

    cand_lower = {s.lower(): s for s in candidate_skills}

    # Pre-embed candidate skills once (used inside the loop).
    cand_vectors = {s: _skill_vector(s) for s in candidate_skills}

    matches: list[SkillMatch] = []
    for req in required_skills:
        req_lower = req.lower()

        # Shortcut: verbatim (or case-insensitive) match.
        if req_lower in cand_lower:
            matches.append(SkillMatch(
                required_skill=req,
                category=_CAT_MATCHED,
                best_candidate_skill=cand_lower[req_lower],
                similarity=1.0,
            ))
            continue

        if not candidate_skills:
            matches.append(SkillMatch(
                required_skill=req,
                category=_CAT_MISSING,
                best_candidate_skill=None,
                similarity=0.0,
            ))
            continue

        req_vec = _skill_vector(req)
        best_skill = None
        best_sim = 0.0
        for cand_skill, cand_vec in cand_vectors.items():
            sim = _cosine(req_vec, cand_vec)
            if sim > best_sim:
                best_sim = sim
                best_skill = cand_skill

        if best_sim >= match_threshold:
            category = _CAT_MATCHED
        elif best_sim >= adjacent_threshold:
            category = _CAT_ADJACENT
        else:
            category = _CAT_MISSING

        matches.append(SkillMatch(
            required_skill=req,
            category=category,
            best_candidate_skill=best_skill,
            similarity=best_sim,
        ))

    return SkillAdjacencyReport(matches=matches)
