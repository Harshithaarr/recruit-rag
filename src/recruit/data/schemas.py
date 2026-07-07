"""Pydantic models for the core domain entities.

WHY use Pydantic schemas instead of raw dicts/DataFrames everywhere:
- A schema is a *contract*. Every other module knows what fields exist and their types.
- Validation catches bad data at the boundary (loader), not deep inside the matcher.
- In your thesis, you can refer to "the Resume schema" as a precise object, not a vibe.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Resume(BaseModel):
    """A single candidate résumé record."""

    resume_id: str
    candidate_name: Optional[str] = None
    text: str = Field(..., description="Full résumé text (the field SBERT will embed)")
    skills: list[str] = Field(default_factory=list, description="Extracted/declared skills")
    years_experience: Optional[float] = None
    location: Optional[str] = None
    education: Optional[str] = None
    target_role: Optional[str] = None

    model_config = {"frozen": True}  # immutable once constructed — safer for parallel processing


class Job(BaseModel):
    """A single job posting."""

    job_id: str
    title: str
    company: Optional[str] = None
    description: str = Field(..., description="Full JD text (the field SBERT will embed)")
    required_skills: list[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None
    location: Optional[str] = None
    required_certifications: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class Match(BaseModel):
    """A single candidate-job match emitted by the retrieval + ranking stage.

    This is the object the RAG explainer and the Streamlit UI will consume.
    """

    resume_id: str
    job_id: str
    semantic_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity")
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rank: int
    dropoff_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    rationale: Optional[str] = None
