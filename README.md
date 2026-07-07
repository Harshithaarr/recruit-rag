# recruit-rag

AI-Based Recruitment Assistant — dissertation project, BITS Pilani M.Tech DSE.

**Title:** Semantic Candidate Matching and Application Drop-Off Prediction Using Retrieval-Augmented Generation
**Student:** Harshitha R (2024DA04025)
**Host:** Jobvite India

## Components

1. **Semantic matcher** — SBERT + FAISS + hybrid ranking with structured filters; BM25 baseline.
2. **Drop-off predictor** — XGBoost on synthetic session-behavioural data; SHAP for interpretation.
3. **RAG explainer** — Local LLM produces a 2–3 sentence rationale per matched candidate-job pair.
4. **Streamlit UI** — Unified recruiter-facing workflow.

## Quickstart

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync deps (base — adds matcher / dropoff / rag / ui as needed)
uv sync
uv sync --extra matcher --extra dropoff --extra rag --extra ui

# Run the data-layer demo
uv run python scripts/demo_data_layer.py
```

## Layout

```
src/recruit/        # library code
  config.py
  data/             # schemas, loaders, cleaners
  embeddings/       # SBERT wrapper
  retrieval/        # FAISS, BM25, hybrid
  dropoff/          # XGBoost + SHAP
  explain/          # RAG pipeline
  evaluation/       # IR & ML metrics, fairness
  ui/               # Streamlit app
data/
  sample/           # tiny committed fixtures
  raw/              # gitignored
  processed/        # gitignored
scripts/            # runnable demos and pipelines
docs/concepts/      # concept-paired notes (these become thesis chapters)
```

## Concept notes

Each slice has a paired concept note in [docs/concepts/](docs/concepts/). These are the source material for the thesis methodology chapter.

See [../viva/](../viva/) for outline-viva preparation.
