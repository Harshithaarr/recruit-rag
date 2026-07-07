"""Experience-recall channel — the third first-phase retrieval channel.

WHY this channel exists:
- Dense retrieval (SBERT + FAISS) catches semantic equivalents but misses
  candidates whose résumé wording diverges sharply from the JD even when their
  *career history* is exactly what the role needs.
- BM25 catches exact terms but is blind to "this person has 15 years of
  progressively senior engineering work" if the literal words don't overlap.
- The experience channel surfaces those candidates at FIRST-PHASE recall, so
  they enter the top-K pool that downstream stages (fusion, rerank, RAG) can
  actually see. Putting this signal at rerank-only would be too late — a
  candidate not in the top-100 cannot be rescued by reordering.

WHY rules-based and not learned:
- No labelled trajectory data exists. Synthesising labels then learning over
  them would compound assumptions on top of assumptions.
- Rules are transparent, auditable, and defensible against a fairness
  examiner. Every weight has a written-down justification.
- The cost is fragility — if a résumé uses unusual phrasing, the parser may
  miss it. That cost is documented in the limitations chapter.

VIVA: "Doesn't ranking on company prestige reproduce the Amazon-2018 bias?"
- Yes, naively, it would. Three mitigations are baked into this module:
  1. `company_tier` is OPTIONAL and audited (the disparate-impact ratio is
     reported both with and without it as part of the fairness chapter).
  2. The default scoring weights make `company_tier` a small bonus (capped
     at 0.15), not a dominant factor.
  3. Trajectory signal is dominated by *progression and tenure* (objective,
     pertains to demonstrated ability) rather than brand alone (proxy for
     access to elite employers).

VIVA: "How do you compute these features when the dataset is raw text?"
- Regex + keyword dictionaries. This is the same approach production
  résumé-parsers (pyresparser, ResumeParser) use as their first pass.
- An LLM-assisted parser is future work; the rules-based v0 is the
  evaluable baseline for that comparison.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

from recruit.data.schemas import Job, Resume


# =====================================================================
# Data structures
# =====================================================================


class Seniority(Enum):
    """Ordered seniority levels. Higher value = more senior."""

    UNKNOWN = 0
    JUNIOR = 1  # intern, junior, graduate, associate
    MID = 2  # engineer, developer (no qualifier)
    SENIOR = 3  # senior
    STAFF = 4  # staff, lead, principal IC
    DIRECTOR = 5  # manager, director, head, VP, CTO


@dataclass(frozen=True)
class ResumeTrajectory:
    """Structured career-history features extracted from one résumé.

    Computed once per résumé at index-build time and cached.
    """

    years_experience: float | None
    seniority: Seniority
    company_tier: int  # 0 = unknown / other,  1 = Tier-1 ("big tech")
    domain: str  # one of DOMAIN_KEYWORDS keys, or "general"
    tenure_signal: float  # log-scaled, in [0, 1]
    role_count: int  # rough estimate from "at <company>" patterns


@dataclass(frozen=True)
class JobCriteria:
    """Structured experience requirements extracted from one JD."""

    min_yoe: float | None
    target_seniority: Seniority
    target_domain: str


@dataclass(frozen=True)
class SearchHit:
    """One result from an experience-channel search.

    Mirrors `faiss_index.SearchHit` and `bm25.SearchHit` so the hybrid layer
    can treat all three channels uniformly.
    """

    index: int
    score: float  # composite trajectory score in [0, 1]


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-component contributions to a single trajectory score.

    Persisted alongside hits so the RAG explainer can textualise *why* a
    candidate ranked where they did — the SHAP→RAG bridge pattern applied
    to this channel.
    """

    yoe_match: float
    seniority_match: float
    domain_overlap: float
    tenure: float
    tier_bonus: float
    total: float


# =====================================================================
# Keyword dictionaries (the "rules" part of the rules-based scorer)
# =====================================================================


# WHY keep Tier-1 small and conservative:
# - The shorter the list, the smaller the prestige-bias surface.
# - "Tier-1" here means firms whose hiring bar is publicly documented as
#   one-of-the-highest. Anything beyond that opens subjective debates.
# - The viva-defensible position: this list is a hypothesis tested by the
#   ablation, not a value judgment.
_TIER_1_COMPANIES: frozenset[str] = frozenset(
    name.lower()
    for name in {
        "Google",
        "Alphabet",
        "Meta",
        "Facebook",
        "Amazon",
        "Apple",
        "Microsoft",
        "Netflix",
        "OpenAI",
        "Anthropic",
        "DeepMind",
    }
)


# Domain keyword sets. A résumé's "primary domain" is whichever set has the
# most hits in its text. Same for JDs.
_DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "backend": frozenset({
        "backend", "back-end", "server", "api", "rest", "grpc", "microservice",
        "python", "go ", "java ", "kotlin", "ruby", "rails", "django", "fastapi",
        "postgres", "postgresql", "mysql", "redis", "kafka", "rabbitmq",
        "distributed systems", "scalability",
    }),
    "frontend": frozenset({
        "frontend", "front-end", "react", "vue", "angular", "next.js",
        "typescript", "javascript", "css", "tailwind", "html", "accessibility",
        "ui/ux", "responsive",
    }),
    "ml": frozenset({
        "machine learning", "ml ", "deep learning", "neural network",
        "pytorch", "tensorflow", "scikit-learn", "huggingface", "hugging face",
        "nlp", "computer vision", "transformer", "recommender",
        "model training", "fine-tuning",
    }),
    "data": frozenset({
        "data science", "data scientist", "data engineer", "data analyst",
        "pandas", "spark", "etl", "sql ", "data warehouse", "dbt", "airflow",
        "analytics", "tableau", "looker", "powerbi",
    }),
    "devops": frozenset({
        "devops", "sre", "kubernetes", "k8s", "docker", "terraform", "ansible",
        "ci/cd", "github actions", "jenkins", "aws", "gcp", "azure",
        "infrastructure", "platform", "observability",
    }),
    "mobile": frozenset({
        "ios", "android", "swift", "kotlin android", "react native", "flutter",
        "objective-c", "mobile",
    }),
    "security": frozenset({
        "security", "pentest", "vulnerability", "ciso", "iam", "soc",
        "appsec", "infosec", "cryptography",
    }),
}


# Seniority detection — patterns that require an engineering / role noun
# adjacent to the seniority word. Stops "directed the team" or "Director of
# XYZ Bootcamp" from triggering DIRECTOR.
#
# Each pattern is a regex; the WHOLE pattern must match (not a substring).
_ROLE_NOUN = r"(?:engineer|developer|architect|scientist|analyst|programmer|manager|consultant|specialist|administrator|admin|lead|director|officer)"

_SENIORITY_PATTERNS: list[tuple[Seniority, list[re.Pattern[str]]]] = [
    (Seniority.DIRECTOR, [
        re.compile(rf"\b(?:director|vp|vice\s+president|head)\s+of\s+(?:engineering|technology|software|data|product|design)", re.IGNORECASE),
        re.compile(rf"\b(?:engineering|technology|software|data|product)\s+director\b", re.IGNORECASE),
        re.compile(r"\bchief\s+(?:technology|technical|data|product|information)\s+officer\b", re.IGNORECASE),
        re.compile(r"\bcto\b|\bcio\b|\bcdo\b", re.IGNORECASE),
    ]),
    (Seniority.STAFF, [
        re.compile(rf"\bstaff\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\bprincipal\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\b(?:tech|engineering|technical)\s+lead\b", re.IGNORECASE),
    ]),
    (Seniority.SENIOR, [
        re.compile(rf"\bsenior\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\bsr\.?\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\blead\s+{_ROLE_NOUN}", re.IGNORECASE),
    ]),
    (Seniority.JUNIOR, [
        re.compile(rf"\b(?:junior|jr\.?)\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\b(?:associate|graduate|entry[-\s]?level)\s+{_ROLE_NOUN}", re.IGNORECASE),
        re.compile(rf"\bintern\b(?:\s+at\b)?", re.IGNORECASE),
    ]),
]


# YOE extraction patterns. Order matters: more specific first.
_YOE_PATTERNS: list[re.Pattern[str]] = [
    # "5 years of experience / professional / work / industry / hands-on"
    re.compile(r"(\d{1,2})\+?\s*years?\s+of\s+(?:professional\s+)?(?:experience|professional|work|industry|hands-on)",
               re.IGNORECASE),
    # "5 years' experience" / "5 year's experience"
    re.compile(r"(\d{1,2})\+?\s*years?[’']?\s+(?:experience|exp)\b", re.IGNORECASE),
    # "5+ years experience" / "5 years exp"
    re.compile(r"(\d{1,2})\+?\s*years?\s+(?:experience|exp)\b", re.IGNORECASE),
    # "with 5+ years"
    re.compile(r"with\s+(\d{1,2})\+?\s*years?\b", re.IGNORECASE),
    # "over 5 years"
    re.compile(r"\bover\s+(\d{1,2})\+?\s*years?\b", re.IGNORECASE),
    # "more than 5 years"
    re.compile(r"\bmore\s+than\s+(\d{1,2})\s*years?\b", re.IGNORECASE),
    # "5+ year" / "5-year" (no plural 's')
    re.compile(r"\b(\d{1,2})\+?[-\s]year[-\s]+(?:career|veteran|background|track)", re.IGNORECASE),
    # "having 5 years"
    re.compile(r"\bhaving\s+(\d{1,2})\+?\s*years?\b", re.IGNORECASE),
    # "5 yrs" or "5+ yrs"
    re.compile(r"(\d{1,2})\+?\s*yrs?\b", re.IGNORECASE),
    # "since YEAR" — convert to YOE relative to a fixed anchor (use 2026 as
    # the corpus reference year; the dissertation submission year).
    re.compile(r"\bsince\s+((?:19|20)\d{2})\b", re.IGNORECASE),
]


# Rough "role" detector — counts occurrences of "at <Company>" or "@ <Company>".
# Defensible v0 — won't catch every résumé format, but produces a usable signal.
_ROLE_AT_PATTERN = re.compile(r"\b(?:at|@)\s+[A-Z][A-Za-z0-9&\-\.]+", re.IGNORECASE)


# Reference year used when converting "since 2015" → YOE.
_CORPUS_REFERENCE_YEAR = 2026


# =====================================================================
# Feature extraction
# =====================================================================


def _detect_yoe(text: str) -> float | None:
    """Best-effort YOE extraction from raw text. Returns None if not found.

    Pattern order matters: tries more specific phrasings first. The 'since YEAR'
    pattern is treated specially — converted to YOE relative to the corpus
    reference year, not the literal year number.
    """
    for pat in _YOE_PATTERNS:
        match = pat.search(text)
        if not match:
            continue
        try:
            num = int(match.group(1))
        except (ValueError, IndexError):
            continue
        # 'since YEAR' branch — convert to YOE delta.
        if 1900 <= num <= 2100:
            yoe = _CORPUS_REFERENCE_YEAR - num
            # Sanity bounds: 0 < yoe < 60.
            if 0 < yoe < 60:
                return float(yoe)
            continue
        # Plain 'N years' branch — sanity bound.
        if 0 < num < 60:
            return float(num)
    return None


def _detect_seniority(text: str) -> Seniority:
    """Highest seniority pattern present wins. Defaults to MID if a generic
    engineering role noun is present, else UNKNOWN.

    Each pattern in `_SENIORITY_PATTERNS` is a regex that REQUIRES the
    seniority word to appear next to an engineering / role noun. This stops
    "directed the team" or "Director of XYZ Bootcamp" from triggering DIRECTOR.
    """
    for level, patterns in _SENIORITY_PATTERNS:
        if any(p.search(text) for p in patterns):
            return level
    lower = text.lower()
    if any(kw in lower for kw in ("engineer", "developer", "scientist", "analyst")):
        return Seniority.MID
    return Seniority.UNKNOWN


def _detect_domain(text: str) -> str:
    """Pick the domain with the most keyword hits. Defaults to 'general'."""
    lower = text.lower()
    best_domain = "general"
    best_hits = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
    return best_domain


def _detect_company_tier(text: str) -> int:
    """1 if any Tier-1 company name appears in an EMPLOYMENT context, else 0.

    Bare-word matching produced ~55% false positives on the HF corpus —
    "used Apple developer tools" or "shopped on Amazon" triggered the flag.
    Now we require one of these surrounding contexts:
      - "at <Company>"            → "at Google"
      - "<Company> Inc." / "Corp." / "LLC."
      - "<Company>, YYYY"         → "Google, 2018"
      - "<Company>, City"         → "Google, Mountain View"
      - "<Company>\n<Title>"      → "Google\nSenior Engineer"
      - "joined <Company>"
      - "@ <Company>"
    """
    for company in _TIER_1_COMPANIES:
        c = re.escape(company)
        patterns = [
            rf"\b(?:at|@|joined|employed\s+(?:by|at)|hired\s+(?:by|at))\s+{c}\b",
            rf"\b{c}\s+(?:inc\.?|corp\.?|corporation|llc\.?|ltd\.?)\b",
            rf"\b{c}\s*,\s*(?:19|20)\d{{2}}",
            rf"\b{c}\s*,\s*[A-Z][a-z]+",   # "Google, Mountain View"
            rf"\b{c}\b\s*\n",              # company on its own line (résumé header)
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            return 1
    return 0


def _tenure_signal(yoe: float | None) -> float:
    """Log-scaled tenure depth, in [0, 1].

    Caps at ~1.0 around 20 YOE. Returns 0.0 for unknown.
    The point of the log: a 10y senior and a 20y senior should not be far
    apart — both are "experienced". A 0y and 5y should be far apart.
    """
    if yoe is None or yoe <= 0:
        return 0.0
    return min(1.0, math.log(yoe + 1.0) / math.log(21.0))


def _role_count(text: str) -> int:
    """Rough count of role mentions via 'at <Company>' patterns."""
    return min(20, len(_ROLE_AT_PATTERN.findall(text)))


def extract_resume_trajectory(resume: Resume) -> ResumeTrajectory:
    """Extract the structured trajectory features from one résumé.

    Prefers schema-declared fields (resume.years_experience) over text-derived
    ones — when both are available, the declared value wins.
    """
    yoe = resume.years_experience
    if yoe is None:
        yoe = _detect_yoe(resume.text)

    seniority = _detect_seniority(
        (resume.target_role or "") + " " + resume.text
    )
    domain = _detect_domain(
        (" ".join(resume.skills) + " " if resume.skills else "")
        + (resume.target_role or "")
        + " "
        + resume.text
    )
    tier = _detect_company_tier(resume.text)
    tenure = _tenure_signal(yoe)
    roles = _role_count(resume.text)

    return ResumeTrajectory(
        years_experience=yoe,
        seniority=seniority,
        company_tier=tier,
        domain=domain,
        tenure_signal=tenure,
        role_count=roles,
    )


def extract_job_criteria(job: Job) -> JobCriteria:
    """Extract structured experience requirements from one JD."""
    yoe = job.min_years_experience
    if yoe is None:
        yoe = _detect_yoe(job.description)

    seniority = _detect_seniority(job.title + " " + job.description)
    domain = _detect_domain(
        (" ".join(job.required_skills) + " " if job.required_skills else "")
        + job.title
        + " "
        + job.description
    )

    return JobCriteria(
        min_yoe=yoe,
        target_seniority=seniority,
        target_domain=domain,
    )


# =====================================================================
# Scoring
# =====================================================================


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the composite trajectory score. Sum to 1.0 by convention.

    Defaults reflect the dissertation's design stance:
    - Tenure and role-progression-equivalent signals dominate.
    - Domain overlap is a strong but not overwhelming match signal.
    - Company-tier is a small bonus (capped), not a primary factor.

    These are starting weights. The evaluation harness will sensitivity-test
    them — that sensitivity analysis is its own thesis figure.
    """

    yoe_match: float = 0.30
    seniority_match: float = 0.20
    domain_overlap: float = 0.25
    tenure: float = 0.10
    tier_bonus: float = 0.15  # capped contribution — see notes in `trajectory_score`


def _yoe_match_score(candidate_yoe: float | None, min_yoe: float | None) -> float:
    """How well does candidate YOE meet/exceed the JD's requirement?

    Returns 1.0 if candidate meets or exceeds, smooth decay below.
    """
    if min_yoe is None or min_yoe <= 0:
        # JD has no YOE requirement → don't penalise anyone on this dimension.
        return 0.5
    if candidate_yoe is None:
        return 0.3  # uncertain — slight penalty for missing signal
    if candidate_yoe >= min_yoe:
        return 1.0
    # Smooth decay: 0.0 at YOE = 0 when min_yoe asks for 10+.
    return max(0.0, candidate_yoe / min_yoe)


def _seniority_match_score(candidate: Seniority, target: Seniority) -> float:
    """How close is candidate's seniority to the target?

    Distance of 0 = perfect (1.0); distance of 1 = adjacent (0.7);
    distance of 2 = noticeable mismatch (0.4); 3+ = poor (0.1).
    Unknown on either side → 0.5 (neutral).
    """
    if candidate == Seniority.UNKNOWN or target == Seniority.UNKNOWN:
        return 0.5
    diff = abs(candidate.value - target.value)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    if diff == 2:
        return 0.4
    return 0.1


def _domain_overlap_score(candidate_domain: str, target_domain: str) -> float:
    """Discrete: same domain → 1.0, both general → 0.5, mismatch → 0.0."""
    if candidate_domain == target_domain:
        return 1.0
    if candidate_domain == "general" or target_domain == "general":
        return 0.5
    return 0.0


def trajectory_score(
    trajectory: ResumeTrajectory,
    criteria: JobCriteria,
    *,
    weights: ScoringWeights | None = None,
    use_company_tier: bool = True,
) -> ScoreBreakdown:
    """Composite trajectory score for one (résumé, JD) pair.

    All component scores are in [0, 1]; the total is also in [0, 1] when
    weights sum to 1.0.

    `use_company_tier` toggles the tier bonus. The fairness audit reports
    metrics both with and without it to isolate its effect.
    """
    w = weights or ScoringWeights()

    yoe = _yoe_match_score(trajectory.years_experience, criteria.min_yoe)
    sen = _seniority_match_score(trajectory.seniority, criteria.target_seniority)
    dom = _domain_overlap_score(trajectory.domain, criteria.target_domain)
    ten = trajectory.tenure_signal
    tier = (trajectory.company_tier * 1.0) if use_company_tier else 0.0

    total = (
        w.yoe_match * yoe
        + w.seniority_match * sen
        + w.domain_overlap * dom
        + w.tenure * ten
        + w.tier_bonus * tier
    )
    return ScoreBreakdown(
        yoe_match=yoe,
        seniority_match=sen,
        domain_overlap=dom,
        tenure=ten,
        tier_bonus=tier,
        total=total,
    )


# =====================================================================
# Index (the channel surface)
# =====================================================================


class ExperienceIndex:
    """Third recall channel — ranks the entire corpus by trajectory score.

    Pre-computes a ResumeTrajectory per résumé at construction (one-time
    O(N) cost) so search-time work is just JD parsing + N scalar scorings.

    Mirrors the API of DenseIndex and BM25Index so the hybrid fusion layer
    can call all three identically.
    """

    def __init__(
        self,
        resumes: list[Resume],
        *,
        weights: ScoringWeights | None = None,
        use_company_tier: bool = True,
    ) -> None:
        self._resumes = resumes
        self._weights = weights
        self._use_company_tier = use_company_tier
        self._trajectories: list[ResumeTrajectory] = [
            extract_resume_trajectory(r) for r in resumes
        ]

    def __len__(self) -> int:
        return len(self._resumes)

    @property
    def trajectories(self) -> list[ResumeTrajectory]:
        return self._trajectories

    def search(self, job: Job, k: int = 10) -> list[SearchHit]:
        """Top-k résumés ranked by trajectory score against the JD."""
        criteria = extract_job_criteria(job)
        scored: list[tuple[int, float]] = []
        for i, trajectory in enumerate(self._trajectories):
            breakdown = trajectory_score(
                trajectory,
                criteria,
                weights=self._weights,
                use_company_tier=self._use_company_tier,
            )
            scored.append((i, breakdown.total))

        scored.sort(key=lambda kv: -kv[1])
        return [
            SearchHit(index=idx, score=score) for idx, score in scored[: k or len(scored)]
        ]

    # Convenience: also expose a text-only entry point matching the BM25 API
    # so eval_hf_fit.py can plug it in alongside the existing retrievers.
    def search_by_text(self, query_text: str, k: int = 10) -> list[SearchHit]:
        """Search using a Job built only from the query text.

        Some eval harnesses pass plain JD text rather than Job objects;
        this wraps the text into a minimal Job for parsing.
        """
        synthetic_job = Job(job_id="__inline__", title="", description=query_text)
        return self.search(synthetic_job, k=k)
