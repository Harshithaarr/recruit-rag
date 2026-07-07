"""LLM-based RAG candidate explanation — Phase D v2.

WHY this module:
- Produces the same `CandidateExplanation` JSON shape as the templated
  explainer, but the prose is composed by a local LLM grounded in retrieved
  evidence (résumé sections, JD requirements, SHAP-derived risk text).
- This is the actual *generation* half of "retrieval-augmented generation",
  the dissertation title's promise.

VIVA: "What stops the LLM from hallucinating?"
- The prompt is constrained to retrieved evidence and forced into structured
  JSON output. Faithfulness is measured against the same evidence sources
  with RAGAS in the evaluation chapter.
- The output schema is identical to the templated explainer's — so a
  faithfulness-failed LLM output is detectable (extra unsourced skills,
  contradictory recommendations).

VIVA: "Why Ollama and not the Anthropic / OpenAI API?"
- Plan §13 calls for a local LLM for reproducibility. Llama 3.1 8B is the
  default. The thesis must run without paid API keys; API-based runs are
  noted in limitations as future work.

USAGE (requires Ollama running locally):
    ollama serve              # in a separate terminal
    ollama pull llama3.1:8b-instruct
    # then use:
    from recruit.explain.llm_rag import build_llm_explanation
"""

from __future__ import annotations

import json
import re

from recruit.config import settings
from recruit.data.schemas import Job, Resume
from recruit.dropoff.predict import DropoffPrediction
from recruit.explain.templated import (
    CandidateExplanation,
    build_explanation as build_templated_explanation,
)
from recruit.skills import extract_skills


# ─────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a recruitment-assistant explainer. Given a job description, a candidate résumé, the matched and missing skills sets, and a drop-off risk analysis grounded in SHAP feature attributions, you produce ONE concise rationale strictly in the JSON schema below.

You must:
1. Cite only evidence given to you. Do NOT invent skills, employers, or claims not present in the inputs.
2. Use the matched/missing skill lists verbatim.
3. Reflect the drop-off risk band and primary drivers in your reasoning.
4. Output ONLY the JSON object, no preface, no markdown fences."""


_PROMPT_TEMPLATE = """JOB DESCRIPTION
{jd_text}

CANDIDATE RÉSUMÉ
{resume_text}

EVIDENCE (computed)
- matched_skills (verbatim):  {matched_skills}
- missing_skills (verbatim):  {missing_skills}
- skill_overlap_pct: {overlap_pct}%
- drop-off risk band: {risk_band}
- drop-off drivers (SHAP, log-odds):
{drivers}

OUTPUT SCHEMA (return JSON only)
{{
  "matched_skills":      [string, ...],     // copy verbatim from EVIDENCE
  "missing_skills":      [string, ...],     // copy verbatim from EVIDENCE
  "overall_fit":         string,            // 1-2 sentences, grounded
  "dropoff_risk":        "low" | "moderate" | "high",
  "dropoff_drivers":     string,            // copy verbatim from EVIDENCE
  "recommendation":      "interview" | "consider" | "likely_mismatch",
  "recommendation_note": string             // 1 sentence justifying recommendation
}}
"""


# Truncation lengths — keep prompts under the 8B model's effective context.
_MAX_RESUME_CHARS = 1500
_MAX_JD_CHARS = 1200


def _build_prompt(
    *,
    resume: Resume,
    job: Job,
    matched: list[str],
    missing: list[str],
    overlap_pct: int,
    risk_band: str,
    drivers_text: str,
) -> str:
    return _PROMPT_TEMPLATE.format(
        jd_text=job.description[:_MAX_JD_CHARS],
        resume_text=resume.text[:_MAX_RESUME_CHARS],
        matched_skills=", ".join(matched) if matched else "(none)",
        missing_skills=", ".join(missing) if missing else "(none)",
        overlap_pct=overlap_pct,
        risk_band=risk_band,
        drivers=drivers_text,
    )


# ─────────────────────────────────────────────────────────────────────────
# Ollama invocation + JSON parsing
# ─────────────────────────────────────────────────────────────────────────


def _call_ollama(
    prompt: str,
    *,
    model: str = "",
    temperature: float = 0.2,
) -> str:
    """Call the local Ollama server's /api/chat endpoint. Returns raw text."""
    import ollama

    model = model or settings.ollama_model
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
        format="json",   # ask Ollama for JSON-shaped output
    )
    return response["message"]["content"]


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_json(raw: str) -> dict | None:
    """Best-effort parse — Ollama's format='json' usually returns clean JSON
    but defensively strip ```json ... ``` wrappers if any model adds them.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Public entry point — matches the templated explainer's signature
# ─────────────────────────────────────────────────────────────────────────


def build_llm_explanation(
    *,
    resume: Resume,
    job: Job,
    prediction: DropoffPrediction,
    model: str = "",
    fallback_on_error: bool = True,
) -> CandidateExplanation:
    """LLM-grounded explanation with optional fallback to the templated version.

    The output is the same CandidateExplanation dataclass produced by the
    templated explainer — downstream consumers don't need to know which
    explainer produced it.

    `fallback_on_error=True` returns the templated explanation if the LLM
    call fails or the JSON is malformed. Recommended for demo robustness.
    """
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
    overlap_pct = (
        int(round(100 * len(matched) / len(req_skills))) if req_skills else 0
    )
    drivers_text = prediction.shap_context_text(top_k=3)

    prompt = _build_prompt(
        resume=resume,
        job=job,
        matched=matched,
        missing=missing,
        overlap_pct=overlap_pct,
        risk_band=prediction.risk_band,
        drivers_text=drivers_text,
    )

    try:
        raw = _call_ollama(prompt, model=model)
        parsed = _parse_json(raw)
        if parsed is None:
            raise ValueError("LLM returned non-JSON output")
    except Exception as e:  # noqa: BLE001
        if not fallback_on_error:
            raise
        # Fallback: hand back the deterministic templated explanation, but
        # marked so the UI can show "LLM unavailable — used template".
        templated = build_templated_explanation(
            resume=resume,
            job=job,
            prediction=prediction,
        )
        return _as_fallback(templated, reason=str(e))

    # Validate / coerce the parsed dict back into our schema.
    return _coerce_to_explanation(
        parsed,
        resume_id=resume.resume_id,
        job_id=job.job_id,
        matched=matched,
        missing=missing,
        overlap_pct=overlap_pct,
        drivers_text=drivers_text,
        prediction=prediction,
    )


def _as_fallback(exp: CandidateExplanation, *, reason: str) -> CandidateExplanation:
    return CandidateExplanation(
        resume_id=exp.resume_id,
        job_id=exp.job_id,
        matched_skills=exp.matched_skills,
        missing_skills=exp.missing_skills,
        skill_overlap_pct=exp.skill_overlap_pct,
        overall_fit=exp.overall_fit + f"  [templated fallback: {reason}]",
        dropoff_risk=exp.dropoff_risk,
        dropoff_drivers=exp.dropoff_drivers,
        recommendation=exp.recommendation,
        recommendation_note=exp.recommendation_note,
    )


_VALID_RISK = {"low", "moderate", "high"}
_VALID_REC = {"interview", "consider", "likely_mismatch"}


def _coerce_to_explanation(
    parsed: dict,
    *,
    resume_id: str,
    job_id: str,
    matched: list[str],
    missing: list[str],
    overlap_pct: int,
    drivers_text: str,
    prediction: DropoffPrediction,
) -> CandidateExplanation:
    """Validate the LLM's JSON; fall back to evidence-grounded defaults on
    any field the model got wrong. The matched/missing skill lists are
    forced to the computed ground truth (the LLM is not allowed to invent
    skills)."""
    risk = str(parsed.get("dropoff_risk", "")).lower()
    if risk not in _VALID_RISK:
        risk = prediction.risk_band

    rec = str(parsed.get("recommendation", "")).lower().strip()
    if rec not in _VALID_REC:
        rec = "consider"  # safe default

    overall = str(parsed.get("overall_fit", "")).strip()
    if not overall:
        overall = f"{overlap_pct}% skill overlap with the role."

    rec_note = str(parsed.get("recommendation_note", "")).strip()
    if not rec_note:
        rec_note = "Recommendation derived from LLM output."

    return CandidateExplanation(
        resume_id=resume_id,
        job_id=job_id,
        # Force evidence-grounded skill lists (LLM may not invent skills).
        matched_skills=matched,
        missing_skills=missing,
        skill_overlap_pct=overlap_pct,
        overall_fit=overall,
        dropoff_risk=risk,
        # Drivers text always comes from SHAP, never the LLM — protects
        # against the LLM rewriting / softening the risk analysis.
        dropoff_drivers=drivers_text,
        recommendation=rec,
        recommendation_note=rec_note,
    )
