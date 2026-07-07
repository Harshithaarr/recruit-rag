# recruit-rag

**AI-Based Recruitment Assistant — Semantic Candidate Matching and Application Drop-Off Prediction Using Retrieval-Augmented Generation**

M.Tech Dissertation · BITS Pilani · Data Science and Engineering
Student: Harshitha R (2024DA04025)
Supervisor: Gokul Gopi
Host organisation: Jobvite India (Employ Inc.)

Mid-sem report: [`reports/midsem_report.md`](reports/midsem_report.md) (submitted 21 June 2026)

---

## What this project does

An integrated recruiter workflow that combines three techniques normally used in isolation:

1. **Semantic candidate matching** — a shortlist for a job description is produced by combining
   dense retrieval (SBERT + FAISS) and lexical retrieval (BM25) via Reciprocal Rank Fusion.
2. **Drop-off prediction** — a calibrated XGBoost model estimates the probability a candidate
   will abandon the application process, using résumé-submission signals from ATS telemetry.
3. **Grounded natural-language rationale** — a local LLM (Ollama) rewrites the model's SHAP
   evidence into a plain-English sentence. The prompt is constrained to rephrase evidence only —
   the LLM cannot invent skills or claims not present in the evidence.

A **fusion ranker** combines the match score and the drop-off probability into one ranking:

    final = 0.60 · semantic + 0.40 · (1 − P_dropoff)

with an optional third term for the **employment-history channel** (evaluator feedback):

    final = 0.60 · semantic + 0.40 · (1 − P_dropoff) + 0.20 · employment-history

A recruiter-facing Streamlit web app exposes all of this as a three-step workflow:
**Search JD → Shortlist → Live watch**.

---

## Quickstart

### 1. Install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project + all optional groups (matcher, dropoff, rag, ui)
uv sync --all-extras
```

### 2. (Optional) Configure environment

Copy `.env.example` to `.env` and adjust if you need to change model names or paths.
Default values work out of the box.

### 3. Download the public datasets

```bash
make data          # uses the kaggle CLI — needs a Kaggle account + API token
```

If you already have the raw CSVs somewhere, drop them into `data/raw/` matching the paths in the
Makefile — `make data` will detect them.

### 4. Set up the local LLM (for the RAG explainer)

The natural-language rationale runs through **Ollama** locally — no API keys required.

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model (~5 GB; only needed once)
ollama pull qwen2.5:7b       # default in .env.example
# — or —
ollama pull llama3.1:8b      # if you'd rather use Llama

# Start the server (leave running in a separate terminal)
ollama serve
```

Change `OLLAMA_MODEL` in `.env` if you prefer a different model.

### 5. Rebuild everything from raw data

```bash
make reproduce     # ~5 min on warm cache; ~15 min cold
```

This runs the full pipeline — generates synthetic data, trains the drop-off model, tunes it with
Optuna, runs the retrieval evaluation, runs the fairness audit, and writes SHAP figures.

### 6. Launch the demo UI

```bash
make ui            # opens on http://localhost:8501
```

---

## Reproducibility

Every artefact in this project is regenerable from raw data via `make` targets. See `make help`
for the full list. Key targets:

| Target | What it does |
|--------|-------------|
| `make install`          | Install Python deps |
| `make data`             | Download public datasets from Kaggle |
| `make generate-data`    | Build the synthetic drop-off training set |
| `make train`            | Train logistic + XGBoost baselines |
| `make tune`             | Optuna hyper-parameter search + isotonic calibration |
| `make eval-retrieval`   | Retrieval benchmark on the labelled HF corpus |
| `make eval-combined`    | Retrieval eval on the combined HF + Kaggle corpus |
| `make explain`          | Global + local SHAP figures |
| `make audit`            | Fairness audit on the synthetic drop-off data |
| `make audit-hr`         | Fairness audit on real HR Analytics demographics |
| `make diagnose-exp`     | Employment-history channel diagnostic study |
| `make generate-postoffer-data` | Synthesize the post-offer decline PoC dataset |
| `make demo`             | CLI end-to-end pipeline demo |
| `make ui`               | Launch the Streamlit UI |
| `make reproduce`        | One-command rebuild of the whole pipeline |

---

## Datasets

| Dataset | Origin | Role |
|---------|--------|------|
| `resume-job-description-fit` | Hugging Face Hub | 477 labelled résumés + 71 labelled JDs — the retrieval benchmark |
| `snehaanbhawal/resume-dataset` | Kaggle | 2,483 additional résumés — corpus realism |
| `arshkon/linkedin-job-postings` | Kaggle | Sampled 71 JDs — out-of-distribution retrieval diagnostic |
| `arashnic/hr-analytics-job-change-of-data-scientists` | Kaggle | ~19,000 rows with real demographic proxies — fairness audit |

Total retrieval corpus: **2,960 résumés** (477 labelled + 2,483 unlabelled).
Labelled scoring uses only the Hugging Face benchmark.

**Note on drop-off labels:** the mid-application and post-offer drop-off datasets are synthetic.
Public labelled data for these events does not exist. The report positions both as
proof-of-concepts with the framework validated end-to-end; production deployment would require
partnership with an ATS vendor for real telemetry.

---

## Repository layout

```
recruit-rag/
├── Makefile                  # 20 reproducibility targets
├── pyproject.toml            # dependencies (uv-managed)
├── .env.example              # editable environment template
│
├── src/recruit/              # library code
│   ├── config.py             # central Settings (model names, paths, seeds)
│   ├── skills.py             # shared skill extractor
│   ├── data/                 # loaders + Pydantic domain schemas
│   ├── embeddings/           # SBERT wrapper with disk cache
│   ├── retrieval/            # BM25 + FAISS + RRF hybrid + employment-history
│   ├── dropoff/              # schemas + simulation + train + predict + SHAP
│   ├── postoffer/            # post-offer decline PoC (end-sem extension)
│   ├── explain/              # templated composer + LLM-RAG composer
│   ├── fairness/             # Fairlearn audits (dropoff + retrieval)
│   ├── evaluation/           # nDCG / Recall@K / MRR + ablation runner
│   └── ui/                   # Streamlit 3-step recruiter web app
│
├── scripts/                  # CLI entry points (one per make target)
│
├── data/
│   ├── raw/                  # gitignored — populated by `make data`
│   ├── processed/            # gitignored — generated artefacts
│   ├── sample/               # tiny committed fixtures for smoke tests
│   └── eval/                 # sample qrels
│
├── docs/concepts/            # architectural / methodology notes
│   ├── 01_data_layer.md
│   ├── 02_matcher.md
│   ├── 03_experience_channel.md
│   └── 04_dropoff_design.md
│
├── models/                   # gitignored — trained .joblib artefacts
├── indexes/                  # gitignored — SBERT / FAISS caches
│
└── reports/
    ├── midsem_report.md      # mid-sem submission (source)
    ├── midsem_report.docx    # mid-sem submission (Word)
    ├── figures/              # SHAP plots referenced in the report
    └── logs/                 # captured pipeline / diagnostic outputs
```

---

## Concept notes

Architectural rationale and methodology detail live in [`docs/concepts/`](docs/concepts/). Each
file is a chapter-sized note that feeds into the dissertation report:

- [`01_data_layer.md`](docs/concepts/01_data_layer.md) — dataset selection, schemas, loader design
- [`02_matcher.md`](docs/concepts/02_matcher.md) — retrieval-channel design and fusion
- [`03_experience_channel.md`](docs/concepts/03_experience_channel.md) — employment-history channel design
- [`04_dropoff_design.md`](docs/concepts/04_dropoff_design.md) — drop-off feature engineering

---

## Reviewer note

This repository was pushed publicly following the mid-sem viva. The mid-sem submission is fully
captured in `reports/midsem_report.md`. End-sem changes address the four feedback points raised
at the viva:

1. **Employment-history alongside keyword search** — [`src/recruit/retrieval/experience.py`](src/recruit/retrieval/experience.py) + an optional UI toggle
2. **Code accessible to reviewer** — this repository
3. **Working POC demo** — Streamlit UI at `make ui`
4. **Beyond-Kaggle résumé sourcing** — 477 labelled résumés from Hugging Face

Additional end-sem work is tracked in the commit history:

- **Post-offer drop-off PoC** — new `src/recruit/postoffer/` module extending the framework
- **Résumé-submission reframing** — drop-off model repositioned around modern candidate flows
- **Recent-experience weighting** — trajectory extractor updated to prioritise last 3–5 years
- **Fairness metrics surfaced** — outputs made prominent in the end-sem materials
