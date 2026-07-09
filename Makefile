# Reproducibility targets for the recruitment-assistant dissertation.
#
# Run `make help` to see the available targets.
# Most targets are idempotent: skip work if outputs already exist.

# ─── Configuration ──────────────────────────────────────────────────────
PY            := OMP_NUM_THREADS=1 uv run python
DATA          := data
RAW           := $(DATA)/raw
PROCESSED     := $(DATA)/processed
MODELS        := models
INDEXES       := indexes
REPORTS       := reports
FIGURES       := $(REPORTS)/figures

# Kaggle dataset slugs (matches scripts/* loaders).
KAGGLE_RESUMES   := snehaanbhawal/resume-dataset
KAGGLE_LINKEDIN  := arshkon/linkedin-job-postings
KAGGLE_HR        := arashnic/hr-analytics-job-change-of-data-scientists

# Drop-off variant for the *default* training/audit run.
VARIANT := v1

# ─── Helpful guidance ───────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Recruitment-Assistant — reproducibility targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install Python deps (uv sync --all-extras)"
	@echo "  make data            Download all Kaggle datasets to data/raw/"
	@echo ""
	@echo "Build models:"
	@echo "  make generate-data           Build the synthetic drop-off dataset (VARIANT=$(VARIANT))"
	@echo "  make train                   Train logistic + XGBoost (VARIANT=$(VARIANT))"
	@echo "  make tune                    Optuna search + isotonic calibration (VARIANT=$(VARIANT))"
	@echo "  make generate-postoffer-data Post-offer decline PoC dataset (end-sem extension)"
	@echo "  make train-postoffer         Train post-offer XGBoost with Optuna tuning"
	@echo "  make train-postoffer-fast    Post-offer training, no tuning (fast dev iteration)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-retrieval  IR ablation on HF labelled benchmark"
	@echo "  make eval-combined   IR ablation on HF + Kaggle combined corpus"
	@echo "  make eval-linkedin   Out-of-distribution diagnostic on LinkedIn queries"
	@echo "  make explain         SHAP global + 3 local explanations + figures"
	@echo "  make diagnose-exp    Trajectory extractor noise diagnostic"
	@echo ""
	@echo "Fairness:"
	@echo "  make audit           Synthetic-data fairness audit (variant=$(VARIANT))"
	@echo "  make audit-hr        Real-demographic fairness audit on HR analytics"
	@echo ""
	@echo "Demo:"
	@echo "  make demo            CLI end-to-end pipeline demo"
	@echo "  make ui              Unified Streamlit dashboard on localhost:8501"
	@echo "                       (recruiter sourcing + live in-session watch"
	@echo "                        panel — 'Watch this candidate apply live' on"
	@echo "                        each candidate card opens the live form below)"
	@echo ""
	@echo "All-in-one:"
	@echo "  make all             install → data → train → tune → audit → eval"
	@echo "  make reproduce       Full reproducibility run (~5 min on warm cache)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove generated parquet / models / figures"
	@echo "  make distclean       Also remove SBERT embedding caches and raw data"

# ─── Setup ──────────────────────────────────────────────────────────────
.PHONY: install
install:
	uv sync --all-extras

# Download targets are file-existence guarded so `make data` is idempotent.
$(RAW)/resumes_kaggle/Resume/Resume.csv:
	@mkdir -p $(RAW)/resumes_kaggle
	kaggle datasets download -d $(KAGGLE_RESUMES) -p $(RAW)/resumes_kaggle --unzip

$(RAW)/linkedin_jobs/postings.csv:
	@mkdir -p $(RAW)/linkedin_jobs
	kaggle datasets download -d $(KAGGLE_LINKEDIN) -p $(RAW)/linkedin_jobs --unzip

$(RAW)/hr_analytics/aug_train.csv:
	@mkdir -p $(RAW)/hr_analytics
	kaggle datasets download -d $(KAGGLE_HR) -p $(RAW)/hr_analytics --unzip

.PHONY: data
data: $(RAW)/resumes_kaggle/Resume/Resume.csv \
      $(RAW)/linkedin_jobs/postings.csv \
      $(RAW)/hr_analytics/aug_train.csv
	@echo "All Kaggle datasets present."

# ─── Synthetic data + model artefacts ──────────────────────────────────
$(PROCESSED)/dropoff_$(VARIANT).parquet: $(RAW)/resumes_kaggle/Resume/Resume.csv \
                                          $(RAW)/linkedin_jobs/postings.csv
	$(PY) scripts/generate_dropoff_data.py \
		--variant $(VARIANT) --resumes combined --jds linkedin --target-rate 0.25

.PHONY: generate-data
generate-data: $(PROCESSED)/dropoff_$(VARIANT).parquet

# ─── Post-offer PoC (end-sem extension responding to mid-sem feedback) ─
$(PROCESSED)/postoffer_v1.parquet:
	$(PY) scripts/generate_postoffer_data.py --n-samples 3000 --target-rate 0.18

.PHONY: generate-postoffer-data
generate-postoffer-data: $(PROCESSED)/postoffer_v1.parquet

$(MODELS)/postoffer_v1.joblib: $(PROCESSED)/postoffer_v1.parquet
	$(PY) scripts/train_postoffer.py

.PHONY: train-postoffer
train-postoffer: $(MODELS)/postoffer_v1.joblib

# Fast dev variant — no Optuna tune, uses defaults. Useful for iteration.
.PHONY: train-postoffer-fast
train-postoffer-fast: $(PROCESSED)/postoffer_v1.parquet
	$(PY) scripts/train_postoffer.py --no-tune

$(MODELS)/dropoff_$(VARIANT).joblib: $(PROCESSED)/dropoff_$(VARIANT).parquet
	$(PY) scripts/train_dropoff.py --variant $(VARIANT)

.PHONY: train
train: $(MODELS)/dropoff_$(VARIANT).joblib

$(MODELS)/dropoff_$(VARIANT)_calibrated.joblib: $(MODELS)/dropoff_$(VARIANT).joblib
	$(PY) scripts/tune_dropoff.py --variant $(VARIANT) --trials 30

.PHONY: tune
tune: $(MODELS)/dropoff_$(VARIANT)_calibrated.joblib

# ─── Evaluation ────────────────────────────────────────────────────────
.PHONY: eval-retrieval
eval-retrieval:
	$(PY) scripts/eval_hf_fit.py

.PHONY: eval-combined
eval-combined:
	$(PY) scripts/eval_combined_corpus.py

.PHONY: eval-linkedin
eval-linkedin:
	$(PY) scripts/eval_linkedin_queries.py

.PHONY: explain
explain: $(MODELS)/dropoff_$(VARIANT).joblib
	$(PY) scripts/explain_dropoff.py

.PHONY: diagnose-exp
diagnose-exp:
	$(PY) scripts/diagnose_experience.py

# ─── Fairness ──────────────────────────────────────────────────────────
.PHONY: audit
audit: $(MODELS)/dropoff_$(VARIANT).joblib
	$(PY) scripts/audit_fairness.py --variant $(VARIANT)

.PHONY: audit-hr
audit-hr: $(RAW)/hr_analytics/aug_train.csv
	$(PY) scripts/audit_fairness_hr.py

# ─── Demo ──────────────────────────────────────────────────────────────
.PHONY: demo
demo: $(MODELS)/dropoff_$(VARIANT).joblib
	$(PY) scripts/demo_pipeline.py --job 24 --top-k 5

.PHONY: ui
ui: $(MODELS)/dropoff_$(VARIANT).joblib
	$(PY) -m streamlit run src/recruit/ui/streamlit_app.py \
		--server.port 8501 --browser.gatherUsageStats false

# ─── Composite targets ─────────────────────────────────────────────────
.PHONY: all
all: install data generate-data train tune audit eval-retrieval

# `make reproduce` is the examiner-facing one-shot: assumes deps installed,
# data downloaded, and runs the full pipeline + headline evaluations.
.PHONY: reproduce
reproduce: generate-data train tune audit eval-retrieval eval-combined explain
	@echo ""
	@echo "Reproducibility run complete. Artefacts:"
	@echo "  models/      — trained classifiers"
	@echo "  data/processed/ — synthetic dataset + splits"
	@echo "  reports/figures/ — SHAP figures"
	@echo "  Logs above show all evaluation tables."

# ─── Cleanup ───────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(PROCESSED) $(MODELS) $(FIGURES)
	@echo "Removed generated parquet / models / figures."

.PHONY: distclean
distclean: clean
	rm -rf $(RAW) $(INDEXES)
	@echo "Also removed raw Kaggle data and SBERT embedding caches."
