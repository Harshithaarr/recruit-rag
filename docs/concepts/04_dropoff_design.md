# Drop-Off Prediction — Label Spec, Feature Schema, Methodology

**Status:** Design proposal · review before coding
**Phase:** C (Implementation Plan §4 steps 11–14)
**Position in thesis:** Methodology chapter (drop-off section)

This document fixes the contracts before any code is written. Everything in
Phase C depends on the label definition and feature schema below.

---

## 1. Label definition

**Drop-off** = a candidate began an application but did not submit within 7
days of their first interaction with the application form.

Precise contract:

| Outcome  | Condition                                                                 |
|----------|---------------------------------------------------------------------------|
| **1 — drop-off**  | `session_started == True` AND `submitted == False` AND `days_since_first_touch ≥ 7` |
| **0 — completed** | `session_started == True` AND `submitted == True` AND `submit_within_days ≤ 7` |
| **excluded**      | `session_started == False`  (visit-only sessions, not real attempts)                  |
| **excluded**      | flagged as bot / spam by basic heuristics (no field interaction, sub-1s session, etc.) |

**Why 7 days specifically:** industry reporting commonly uses 7-day windows
for application-cohort completion rate. Long enough that genuine completers
returning the next day are not mis-classified as drop-off; short enough that
the recruiter intervention signal is still actionable.

**Why exclude visit-only sessions:** they are pre-intent traffic. Including
them inflates the drop-off rate with users who never had an application
intent to abandon. The thesis target is *abandonment of an in-progress
application*, not "did not start one."

**Why this is a binary classifier, not survival analysis:** the recruiter
intervention has to fire within an active session. Time-to-event modelling
would be richer but the action is binary (intervene now, or not).

---

## 2. Why synthetic labels — methodology

Public datasets do not include true application-level drop-off labels. The
plan (§13) flags three honest options:

1. Repurpose adjacent labels (e.g. HR Analytics "looking for change") as a
   proxy — defensible but loose.
2. Use IBM HR Attrition — measures employee attrition, not applicant
   drop-off, so the target is wrong.
3. **Synthesise labels under a transparent simulation rule.**

This work takes path (3). The rule is documented in §3 below. The thesis
methodology chapter frames the experiment as:

> *"In the absence of public ground-truth on application-level drop-off,
> we generate a labelled dataset under a transparent simulation rule. The
> experiment demonstrates that a tree-based classifier can recover the
> structure implied by that rule, and provides infrastructure for evaluation
> on real ATS data should it become available. Real-world predictive
> performance is validated by sensitivity analysis on the simulation rule
> rather than by absolute metric values."*

Defensibility hinges on three things:

- The rule must be **non-trivial** — a single deterministic threshold would
  let any model achieve AUC=1.0, proving nothing.
- The rule must include **noise** — so calibration and probability
  estimates are meaningful, not trivially saturated.
- The rule must be **sensitivity-tested** — at least three variations
  (different weights, different noise levels) reported, to show results are
  robust to the specific simulation parameterisation.

---

## 3. Label-generating rule

Drop-off probability is a logistic function of weighted features plus a
noise term:

```
logit(P_dropoff) =
    w_0
    + w_1  · skill_gap                       # ∈ [0, 1]
    + w_2  · max(0, min_yoe − candidate_yoe) # under-qualified penalty
    − w_3  · max(0, candidate_yoe − min_yoe) # over-qualified buffer (caps at 5)
    + w_4  · 𝟙(posting_age_days > 14)
    + w_5  · (n_required_fields / 10)
    − w_6  · 𝟙(has_remote_option)
    + w_7  · 𝟙(is_mobile_session)
    + w_8  · (time_on_page_secs / 600)       # >10 min => signal of struggle
    + w_9  · (n_fields_skipped / n_required_fields)
    + ε,    ε ~ Normal(0, σ²)

P_dropoff = sigmoid( logit(P_dropoff) )
label    ~ Bernoulli( P_dropoff )
```

**Default weights (`v0` simulation):**

```
w_0 = -0.4   (intercept — base log-odds, calibrates rate to ~55%)
w_1 = +2.0   (skill_gap)              — dominant signal
w_2 = +1.2   (under-qualified)        — strong
w_3 = +0.3   (over-qualified buffer)  — mild
w_4 = +0.8   (posting age > 14d)      — moderate
w_5 = +0.5   (n_required_fields/10)   — moderate
w_6 = +0.7   (remote available)       — moderate
w_7 = +0.6   (mobile session)         — moderate
w_8 = +0.9   (long time-on-page)      — moderate-strong
w_9 = +1.5   (% fields skipped)       — strong
σ   = 0.4    (noise term)
```

**Three sensitivity variants reported** (`v1`, `v2`, `v3`): weights scaled
±25%, noise σ ∈ {0.2, 0.6}. The thesis reports model performance across
all four parameterisations to show the result is not an artefact of any
specific weight choice.

**Calibrated drop-off rate:** ~55% under `v0`, matching the >60% industry
report cited in the outline §3. Validated post-generation by computing the
empirical positive class share.

---

## 4. Feature schema

Four feature families per the plan §7. The model at prediction time sees
the same features that generated the label (the simulation contract).

### 4.1 Match features  (candidate ↔ job alignment)

| Column            | Type    | Computation                                                  |
|-------------------|---------|--------------------------------------------------------------|
| `sem_similarity`  | float   | Cosine similarity from the SBERT encoder (the value already produced by the matcher) |
| `bm25_score`      | float   | BM25 score from the lexical retriever                        |
| `skill_overlap`   | float ∈ [0,1] | `|skills ∩ required| / |required|` (set overlap)        |
| `skill_gap`       | float ∈ [0,1] | `1 − skill_overlap`  (mirror for interpretability)      |
| `yoe_gap`         | float   | `candidate.years_experience − job.min_years_experience`      |
| `under_qualified` | bool    | `yoe_gap < 0`                                                |
| `education_match` | bool    | True iff candidate's highest degree ≥ job's required degree (when declared) |

### 4.2 Candidate features

| Column                  | Type    | Source                              |
|-------------------------|---------|-------------------------------------|
| `cand_yoe`              | float   | `resume.years_experience`           |
| `cand_n_past_roles`     | int     | Trajectory extractor (`role_count`) |
| `cand_avg_tenure_yrs`   | float   | `yoe / max(1, n_past_roles)`        |
| `cand_seniority_level`  | int 0–5 | Trajectory extractor                |
| `cand_domain`           | enum    | Trajectory extractor (one-hot at training) |
| `cand_location_country` | str     | Parsed from `resume.location` (one-hot top-N + "other") |

### 4.3 Job features

| Column               | Type    | Source                                         |
|----------------------|---------|------------------------------------------------|
| `job_posting_age_days` | int   | `now − job.posted_at`  (simulated in synthesis) |
| `job_n_required_fields` | int  | Length of the application form (simulated)     |
| `job_has_remote_option` | bool | Parsed from JD text                            |
| `job_seniority_level` | int 0–5 | Trajectory extractor on JD                    |
| `job_n_applicants_so_far` | int | Simulated under a Poisson process            |
| `job_salary_band`     | int 1–4 | Simulated 4-tier band                         |

### 4.4 Interaction (session) features

| Column                 | Type  | Source                                              |
|------------------------|-------|-----------------------------------------------------|
| `session_time_on_page_secs` | int  | Simulated session duration                     |
| `session_hour_of_day`  | int 0–23 | Simulated, with realistic distribution          |
| `session_is_weekend`   | bool  | Simulated                                           |
| `session_device_mobile` | bool | Simulated (mobile vs desktop share)                 |
| `session_n_fields_completed` | int | Simulated                                       |
| `session_n_fields_skipped`   | int | Simulated                                       |
| `session_field_completion_rate` | float ∈ [0,1] | `completed / required`                  |
| `session_navigation_back_count` | int | Simulated count of back-navigations            |

**Excluded for fairness:** name, photo, gender, age, race. Documented as
explicit omissions in the fairness chapter.

---

## 5. Dataset generation

- **N rows:** 10,000 simulated (candidate, job, session) triples.
- **Construction:** sample `(resume_i, job_j)` pairs from the existing
  resumes corpus (HF fit + later Kaggle). For each pair, sample synthetic
  session features under realistic distributions, compute the label per
  §3.
- **Stratification:** stratify the sample by job-seniority bucket to
  prevent the model overfitting one seniority slice.
- **Splits:**
  - Train: 70%  (seeded random)
  - Validation: 15%
  - Test: 15%
  - Persisted in `data/processed/dropoff_splits.json` by row index.

---

## 6. Training methodology

**Baseline (required):** logistic regression with same features. Reported
alongside XGBoost so the thesis can claim the tree-based model adds value,
not just performs.

**XGBoost configuration:**

- `objective = binary:logistic`
- `eval_metric = ['auc', 'logloss']`
- `scale_pos_weight = (negatives / positives)` for class balance
- Early stopping on validation AUC with `early_stopping_rounds = 50`
- Optuna search over `max_depth`, `learning_rate`, `n_estimators`,
  `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`,
  `reg_lambda` — 50 trials, TPE sampler.
- Random seed fixed at `settings.seed`.

**Calibration:** isotonic regression on validation predictions
(`CalibratedClassifierCV` with `cv='prefit'`).

**Threshold selection:** F1-optimal threshold computed on validation; also
report performance at threshold = 0.5 for comparison.

**Sensitivity analysis:** rerun the entire pipeline on `v1`, `v2`, `v3`
simulations and report metric distribution.

---

## 7. Evaluation metrics

Per outline §4 and plan §9:

| Metric        | Required target | Source                          |
|---------------|-----------------|---------------------------------|
| ROC-AUC       | ≥ 0.80          | outline §4 objective 4          |
| F1            | ≥ 0.70          | outline §4 objective 4          |
| PR-AUC        | report          | better than AUC for imbalanced  |
| Brier score   | report          | calibration quality             |
| Calibration plot | report       | visual                          |
| Confusion matrix at chosen threshold | report | for context |

All reported with bootstrap confidence intervals (1000 resamples).

---

## 8. SHAP interpretability

Per plan §8:

- `shap.TreeExplainer(model)` on the trained XGBoost.
- Global beeswarm + feature-importance bar on test set.
- Per-candidate waterfall in the Streamlit UI.
- **The SHAP→RAG bridge** — top-3 SHAP features per prediction are
  textualised and fed into the RAG explainer prompt as structured context.
  This is the cross-component contribution flagged in the outline novelty
  section.
- SHAP values cached per (candidate, job) pair — computation is the
  expensive step.

---

## 9. Open questions for review before coding

1. **Drop-off label window** — confirm 7 days. Reasonable alternatives are
   3 days (matches mobile-funnel reporting) or 14 days (matches LinkedIn).
2. **Simulation N** — 10,000 rows. Smaller (5k) is faster to iterate;
   larger (50k) gives tighter confidence intervals.
3. **Sensitivity variants** — 4 parameterisations as proposed, or fewer
   (just `v0` for v1 of the thesis, add variants later)?
4. **Fairness columns** — confirm name/photo/gender/age/race are excluded
   from the feature set. The fairness chapter audits *proxies* (e.g.
   location, education) for disparate impact.
5. **Real-data validation pointer** — should the methodology section
   include language for a later "if production data becomes available"
   validation pass, or leave that strictly to Future Work?

---

## 10. Implementation file layout

```
src/recruit/dropoff/
  __init__.py
  schemas.py           # Pydantic models: Session, DropoffSample, etc.
  simulation.py        # label-generating rule + dataset constructor
  features.py          # feature extractors (4 families)
  train.py             # baseline + XGBoost + Optuna + calibration
  predict.py           # serving-time prediction
  shap_utils.py        # global + local explainers, cached
scripts/
  generate_dropoff_data.py  # one-shot dataset builder
  train_dropoff.py          # baseline + XGBoost end-to-end
  eval_dropoff.py           # metric tables + sensitivity sweep
data/processed/
  dropoff_v0.parquet  (and v1/v2/v3)
  dropoff_splits.json
```

Once the five open questions above are answered, the implementation order
is: simulation → features → train → calibrate → SHAP → predict.
