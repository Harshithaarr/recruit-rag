"""Templated candidate explanation — Phase D POC.

WHY templated (not LLM):
- For the POC, this composes the exact same JSON shape an LLM would later
  produce. Swapping to Ollama / llama3.1-8b is a one-line change at the
  call site; the schema and downstream consumers don't change.
- A templated baseline is also defensible in the thesis as the *deterministic
  comparator* for an LLM explainer (RAGAS faithfulness comparison: does
  the LLM stay grounded, or does it add unsourced claims?).

VIVA: "Isn't a template just string formatting?"
- Yes — and that's the point. The contract is the *output schema* and the
  *evidence sources*. Whether the prose is composed by templates or by an
  LLM is implementation detail. Both must cite the same evidence sources
  (matched skills, missing skills, SHAP drivers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from recruit.data.schemas import Job, Resume
from recruit.dropoff.predict import DropoffPrediction
from recruit.skills import extract_skills


@dataclass(frozen=True)
class CandidateExplanation:
    """One end-to-end explanation packaged for the UI and the RAG ablation.

    The JSON shape here is identical to what the LLM explainer will produce
    in Phase D v2 — same fields, same types. The thesis ablation can then
    compare "templated explainer" vs "LLM explainer" on faithfulness and
    answer-relevance without changing any downstream consumers.
    """

    resume_id: str
    job_id: str

    matched_skills: list[str]
    missing_skills: list[str]
    skill_overlap_pct: int

    overall_fit: str          # one-sentence summary
    dropoff_risk: str         # "low" | "moderate" | "high"
    dropoff_drivers: str      # textualised SHAP — the SHAP→RAG bridge
    recommendation: str       # "interview" | "consider" | "likely_mismatch"
    recommendation_note: str  # one-sentence justification

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """Human-readable form for CLI demos and Streamlit st.markdown."""
        lines = [
            f"Candidate {self.resume_id}  ·  Job {self.job_id}",
            "",
            f"Matched skills  ({len(self.matched_skills)}): "
            + (", ".join(self.matched_skills) or "—"),
            f"Missing skills  ({len(self.missing_skills)}): "
            + (", ".join(self.missing_skills) or "—"),
            f"Skill overlap : {self.skill_overlap_pct}%",
            "",
            f"Overall fit   : {self.overall_fit}",
            "",
            "Drop-off risk : " + self.dropoff_risk.upper(),
            self.dropoff_drivers,
            "",
            f"Recommendation: {self.recommendation.upper().replace('_', ' ')}",
            f"  → {self.recommendation_note}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Composition rules — the actual "template"
# ─────────────────────────────────────────────────────────────────────────


def _overall_fit_sentence(
    matched: list[str],
    missing: list[str],
    overlap: float,
) -> str:
    n_req = len(matched) + len(missing)
    if n_req == 0:
        return "Job lists no required skills; fit assessment relies on semantic match alone."
    if overlap >= 0.75:
        return (
            f"Strong fit — covers {len(matched)} of {n_req} required skills"
            + (f" ({', '.join(matched[:4])}{'…' if len(matched) > 4 else ''})." if matched else ".")
        )
    if overlap >= 0.50:
        gap_phrase = (
            f"; primary gaps on {', '.join(missing[:3])}"
            if missing else ""
        )
        return (
            f"Partial fit — has {len(matched)} of {n_req} required skills"
            + gap_phrase + "."
        )
    if overlap >= 0.25:
        return (
            f"Weak skill alignment — only {len(matched)} of {n_req} required skills present"
            + (f" (missing {', '.join(missing[:3])}{'…' if len(missing) > 3 else ''})." if missing else ".")
        )
    return (
        f"Skill mismatch — {len(matched)} of {n_req} required skills present; "
        "consider only if other signals are strong."
    )


def _recommendation(overlap: float, dropoff_band: str) -> tuple[str, str]:
    """Return (recommendation_key, justification_sentence).

    Combines skill_overlap (retrieval signal) with drop-off risk band
    (engagement signal) — the fusion novelty made concrete.
    """
    if overlap >= 0.50 and dropoff_band == "low":
        return (
            "interview",
            "High skill alignment and low predicted drop-off risk — strong action candidate.",
        )
    if overlap >= 0.50 and dropoff_band == "moderate":
        return (
            "consider",
            "Skill match is good; moderate drop-off risk suggests outreach earlier rather than later.",
        )
    if overlap >= 0.50 and dropoff_band == "high":
        return (
            "consider",
            "Skills line up but candidate shows high drop-off risk — prioritise immediate contact.",
        )
    if overlap >= 0.25 and dropoff_band == "low":
        return (
            "consider",
            "Partial skill overlap; candidate appears engaged so worth a brief screen.",
        )
    return (
        "likely_mismatch",
        "Low skill alignment with the role; recommend deprioritising unless context says otherwise.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


def build_explanation(
    *,
    resume: Resume,
    job: Job,
    prediction: DropoffPrediction,
) -> CandidateExplanation:
    """Compose one CandidateExplanation from a (Resume, Job, DropoffPrediction).

    Evidence sources are explicit — same set the LLM explainer will use:
      - matched / missing skills derived from the shared skill vocabulary
      - SHAP drivers from the drop-off prediction (the SHAP→RAG bridge)
    """
    # Skill overlap — uses the same vocabulary as the data generator and the
    # retrieval ablation, for consistency.
    if resume.skills:
        cand_skills = {s.lower().strip() for s in resume.skills}
    else:
        cand_skills = extract_skills(resume.text)

    if job.required_skills:
        req_skills = {s.lower().strip() for s in job.required_skills}
    else:
        req_skills = extract_skills(job.description)

    matched = sorted(cand_skills & req_skills)
    missing = sorted(req_skills - cand_skills)
    overlap = len(matched) / len(req_skills) if req_skills else 0.0

    overall_fit = _overall_fit_sentence(matched, missing, overlap)
    rec_key, rec_note = _recommendation(overlap, prediction.risk_band)

    return CandidateExplanation(
        resume_id=resume.resume_id,
        job_id=job.job_id,
        matched_skills=matched,
        missing_skills=missing,
        skill_overlap_pct=int(round(100 * overlap)),
        overall_fit=overall_fit,
        dropoff_risk=prediction.risk_band,
        dropoff_drivers=prediction.shap_context_text(top_k=3),
        recommendation=rec_key,
        recommendation_note=rec_note,
    )
