"""Lightweight skill-set extraction from raw text.

WHY a shared module (not just inline in the data generator):
- The drop-off feature extractor, the templated explainer, and the matcher
  ablations all need a consistent notion of "what skills appear in this
  text". Defining it once means every downstream comparison uses the same
  vocabulary.

VIVA: "Why a fixed vocabulary rather than NER / O*NET tagging?"
- O*NET-based skill tagging is on the future-work list (LLM-assisted
  extraction). The fixed vocabulary is the rules-based baseline.
- Examiner-defensible because the vocabulary is documented, auditable, and
  identical across components.
"""

from __future__ import annotations


# Compact tech vocabulary. Not exhaustive — calibrated to give a non-trivial
# skill_overlap distribution on the HF résumé / JD corpus.
SKILL_VOCAB: list[str] = [
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "ruby", "kotlin", "swift", "scala", "php", "sql",
    "react", "vue", "angular", "next.js", "node.js", "django", "flask",
    "fastapi", "spring", "rails",
    "postgres", "postgresql", "mysql", "mongodb", "redis", "kafka",
    "elasticsearch", "snowflake", "bigquery",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions",
    "pytorch", "tensorflow", "scikit-learn", "huggingface", "transformers",
    "pandas", "numpy", "spark", "airflow", "dbt",
    "rest", "grpc", "graphql", "microservices", "ci/cd", "agile", "scrum",
    "tableau", "looker", "powerbi",
]


def extract_skills(text: str, vocab: list[str] = SKILL_VOCAB) -> set[str]:
    """Lower-case substring match against the compact tech vocabulary.

    Returns a *set* of lower-cased skill names (deduplicated, order-independent).
    """
    low = text.lower()
    return {s for s in vocab if s in low}
