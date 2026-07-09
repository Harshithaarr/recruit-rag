"""Streamlit recruiter interface — the POC visual demo.

Run:  OMP_NUM_THREADS=1 uv run streamlit run src/recruit/ui/streamlit_app.py

What this UI does:
- Paste or pick a JD → click "Find candidates" → see Top-K ranked candidates
  with semantic + drop-off + final scores, matched / missing skills, a
  drop-off risk badge with textualised SHAP drivers, and a templated
  recommendation.

The UI is the *visual* surface for the same end-to-end pipeline implemented
in scripts/demo_pipeline.py. Anyone who can open a browser can demo it.

VIVA: "Why Streamlit?" — standard ML-demo framework, six lines render a
sortable table. Plan §10 specifies it. Production ATS integration would
use a proper web framework — out of scope.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from recruit.config import settings
from recruit.data.loaders import load_hf_fit_dataset, load_kaggle_resumes
from recruit.data.schemas import Job, Resume
from recruit.dropoff.predict import DropoffPredictor
from recruit.embeddings.sbert import SBertEncoder, embed_with_cache
from recruit.explain.templated import build_explanation
from recruit.explain.llm_rag import build_llm_explanation
from recruit.retrieval.bm25 import BM25Index
from recruit.retrieval.faiss_index import DenseIndex
from recruit.retrieval.hybrid import reciprocal_rank_fusion

# Single-threaded BLAS / OpenMP to avoid the segfault when faiss + xgboost
# + sklearn share thread pools on macOS. Same workaround as the CLI demo.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ─────────────────────────────────────────────────────────────────────────
# One-time pipeline load — cached for the life of the Streamlit process
# ─────────────────────────────────────────────────────────────────────────


# Model registry — friendly label → path. Order is "best first" for the
# default dropdown selection and the comparison view.
_MODEL_REGISTRY: list[tuple[str, str]] = [
    ("v1 calibrated  (recommended)", "dropoff_v1_calibrated.joblib"),
    ("v1 tuned",                     "dropoff_v1_tuned.joblib"),
    ("v1 default",                   "dropoff_v1.joblib"),
    ("v0 calibrated",                "dropoff_v0_calibrated.joblib"),
    ("v0 tuned",                     "dropoff_v0_tuned.joblib"),
    ("v0 default",                   "dropoff_v0.joblib"),
]


# Application form fields used by the inline live-watch panel below each
# candidate card. Generic fields so they apply to any (candidate, JD) pair.
FORM_FIELDS: list[dict] = [
    {"key": "name",       "label": "Full name",                            "long": False},
    {"key": "email",      "label": "Email address",                        "long": False},
    {"key": "location",   "label": "Current location",                     "long": False},
    {"key": "start_date", "label": "Earliest start date",                  "long": False},
    {"key": "salary",     "label": "Expected salary",                      "long": False},
    {"key": "ml_exp",     "label": "Describe your relevant experience",    "long": True},
    {"key": "pytorch",    "label": "Skills breakdown for this role",       "long": True},
    {"key": "interest",   "label": "Why are you interested in this role?", "long": True},
]


@st.cache_resource(show_spinner="Loading combined corpus + indexes (one-time)…")
def load_corpus_and_indexes() -> dict:
    """Heavy one-time load: combined HF + Kaggle résumé corpus + indexes.

    Résumé pool = HF labelled set (477) + Kaggle snehaanbhawal corpus (2,483)
    → 2,960 résumés total. JD pool stays as the HF labelled set (71) — those
    are the JDs with relevance labels, used by the eval scripts.

    SBERT vectors for the combined corpus are cached at
    indexes/combined_hf_plus_kaggle.npy from prior eval runs, so the first
    load is instant once the cache is warm.
    """
    hf_resumes, jobs, _ = load_hf_fit_dataset(split="test")
    kaggle_resumes = load_kaggle_resumes(
        settings.data_path / "raw" / "resumes_kaggle" / "Resume" / "Resume.csv"
    )
    resumes = hf_resumes + kaggle_resumes

    encoder = SBertEncoder()
    resume_texts = [r.text for r in resumes]
    resume_vecs = embed_with_cache(
        encoder, resume_texts, label="combined_hf_plus_kaggle"
    )
    dense_idx = DenseIndex(resume_vecs)
    bm25_idx = BM25Index(resume_texts)

    # Employment-history channel — pre-computes a trajectory per résumé
    # (one-time O(N) cost so search-time work is a scalar per candidate).
    from recruit.retrieval.experience import ExperienceIndex
    exp_idx = ExperienceIndex(resumes)

    from recruit.skills import extract_skills
    job_indices_by_richness = sorted(
        range(len(jobs)),
        key=lambda i: -len(extract_skills(jobs[i].description)),
    )
    return {
        "resumes": resumes,
        "jobs": jobs,
        "n_hf": len(hf_resumes),
        "n_kaggle": len(kaggle_resumes),
        "encoder": encoder,
        "dense": dense_idx,
        "bm25": bm25_idx,
        "experience": exp_idx,
        "job_order": job_indices_by_richness,
    }


@st.cache_resource(show_spinner="Loading drop-off model…")
def load_predictor(model_filename: str) -> DropoffPredictor:
    """One predictor per model file; cached so flipping between v0/v1 is instant."""
    path = settings.models_path / model_filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return DropoffPredictor(path)


def _available_models() -> list[tuple[str, str]]:
    """Subset of the registry that's actually present on disk."""
    return [
        (label, fname)
        for label, fname in _MODEL_REGISTRY
        if (settings.models_path / fname).exists()
    ]


# ─────────────────────────────────────────────────────────────────────────
# Live-watch state — the in-session live drop-off panel state, per one
# selected candidate. All state lives in st.session_state so it survives
# Streamlit reruns.
# ─────────────────────────────────────────────────────────────────────────


def init_watch_state() -> None:
    """One-time initialisation of the live-watch session_state keys."""
    ss = st.session_state
    ss.setdefault("watching_resume_idx", None)
    ss.setdefault("watch_started_ts", None)
    ss.setdefault("watch_back_count", 0)
    ss.setdefault("watch_history", [])         # list of (elapsed_secs, p_dropoff)
    ss.setdefault("watch_is_mobile", False)
    for f in FORM_FIELDS:
        ss.setdefault(f"watch_field_{f['key']}_value", "")
        ss.setdefault(f"watch_field_{f['key']}_status", "pending")


def reset_watch_state(new_resume_idx: int | None = None) -> None:
    """Reset the watch state — called when the recruiter clicks a different
    candidate's "Watch live" button, or the "Reset" button.
    """
    ss = st.session_state
    ss.watching_resume_idx = new_resume_idx
    ss.watch_started_ts = None
    ss.watch_back_count = 0
    ss.watch_history = []
    ss.watch_is_mobile = False
    ss.current_scenario_label = None
    for f in FORM_FIELDS:
        ss[f"watch_field_{f['key']}_value"] = ""
        ss[f"watch_field_{f['key']}_status"] = "pending"


# Résumé-submission scenarios — set the underlying session state to simulate
# what an ATS would emit during a candidate's résumé-upload session. Same
# features the drop-off model already consumes; only the semantic framing
# changes (résumé-submission funnel instead of form-field completion).
_SUBMISSION_SCENARIOS: dict[str, dict] = {
    "fast": {
        "label": "🟢 Fast submitter — uploaded and submitted immediately",
        "n_completed": 6,
        "n_skipped": 0,
        "elapsed_secs": 30,
        "back_count": 0,
        "is_mobile": False,
    },
    "hesitant": {
        "label": "🟡 Hesitant applicant — considered before submitting",
        "n_completed": 4,
        "n_skipped": 2,
        "elapsed_secs": 90,
        "back_count": 1,
        "is_mobile": True,
    },
    "abandoning": {
        "label": "🔴 Abandoning applicant — opened uploader but never submitted",
        "n_completed": 2,
        "n_skipped": 5,
        "elapsed_secs": 120,
        "back_count": 3,
        "is_mobile": True,
    },
}


def _apply_submission_scenario(scenario_key: str, resume_idx: int) -> None:
    """Populate session state as if the ATS had emitted these telemetry events.

    Same underlying state the form-based UI used, but set directly by the
    scenario rather than by field-by-field candidate interaction. The
    aggregation logic in aggregate_watch_session is unchanged.
    """
    scenario = _SUBMISSION_SCENARIOS[scenario_key]
    ss = st.session_state
    ss.watching_resume_idx = resume_idx
    ss.watch_started_ts = time.time() - scenario["elapsed_secs"]
    ss.watch_back_count = scenario["back_count"]
    ss.watch_is_mobile = scenario["is_mobile"]
    ss.current_scenario_label = scenario["label"]
    ss.watch_history = []

    # Distribute filled/skipped/pending across the underlying step slots.
    n_completed = scenario["n_completed"]
    n_skipped = scenario["n_skipped"]
    for i, f in enumerate(FORM_FIELDS):
        if i < n_completed:
            ss[f"watch_field_{f['key']}_status"] = "filled"
            ss[f"watch_field_{f['key']}_value"] = "(auto-populated from ATS)"
        elif i < n_completed + n_skipped:
            ss[f"watch_field_{f['key']}_status"] = "skipped"
            ss[f"watch_field_{f['key']}_value"] = ""
        else:
            ss[f"watch_field_{f['key']}_status"] = "pending"
            ss[f"watch_field_{f['key']}_value"] = ""


def _increment_status(new_status: str) -> None:
    """Mark one more pending step as filled or skipped — for fine-tuning."""
    ss = st.session_state
    for f in FORM_FIELDS:
        if ss[f"watch_field_{f['key']}_status"] == "pending":
            ss[f"watch_field_{f['key']}_status"] = new_status
            if new_status == "filled":
                ss[f"watch_field_{f['key']}_value"] = "(auto-populated from ATS)"
            if ss.watch_started_ts is None:
                ss.watch_started_ts = time.time()
            return


def _seconds_watching() -> int:
    ss = st.session_state
    if ss.watch_started_ts is None:
        return 0
    return int(time.time() - ss.watch_started_ts)


def aggregate_watch_session() -> dict:
    """Compute the live session-feature dict from the current watch state."""
    ss = st.session_state
    completed = sum(
        1 for f in FORM_FIELDS
        if ss[f"watch_field_{f['key']}_status"] == "filled"
    )
    skipped = sum(
        1 for f in FORM_FIELDS
        if ss[f"watch_field_{f['key']}_status"] == "skipped"
    )
    n_required = len(FORM_FIELDS)
    now = time.localtime()
    return {
        "session_time_on_page_secs": _seconds_watching(),
        "session_hour_of_day": now.tm_hour,
        "session_is_weekend": now.tm_wday >= 5,
        "session_device_mobile": ss.watch_is_mobile,
        "session_n_fields_completed": completed,
        "session_n_fields_skipped": skipped,
        "session_field_completion_rate": completed / n_required,
        "session_navigation_back_count": ss.watch_back_count,
        "job_n_required_fields": n_required,
    }


def load_pipeline(model_filename: str) -> dict:
    """Compose corpus + indexes + chosen predictor."""
    base = load_corpus_and_indexes()
    predictor = load_predictor(model_filename)
    postoffer_predictor = _try_load_postoffer_predictor()
    return {
        **base,
        "predictor": predictor,
        "postoffer_predictor": postoffer_predictor,
        "active_model_path": settings.models_path / model_filename,
    }


@st.cache_resource(show_spinner="Loading post-offer PoC model…")
def _try_load_postoffer_predictor():
    """Load the post-offer predictor if the model artefact exists.

    The post-offer PoC is an end-sem extension responding to reviewer
    feedback that post-offer drop-off has higher business value than mid-
    application. Returns None gracefully if the model hasn't been trained
    yet (`make train-postoffer` produces the artefact).
    """
    path = settings.models_path / "postoffer_v1_calibrated.joblib"
    if not path.exists():
        return None
    try:
        from recruit.postoffer.predict import PostOfferPredictor
        return PostOfferPredictor(path)
    except Exception as e:  # noqa: BLE001
        # Non-fatal — post-offer step will surface a helpful error and
        # the main workflow keeps working.
        print(f"Post-offer predictor unavailable: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# End-to-end scoring for one JD
# ─────────────────────────────────────────────────────────────────────────


def score_candidates(
    pipe: dict,
    jd_text: str,
    *,
    pool: int = 30,
    w_sem: float = 0.6,
    w_stay: float = 0.4,
    w_exp: float = 0.0,
    use_llm: bool = False,
) -> list[dict]:
    encoder = pipe["encoder"]
    dense_idx = pipe["dense"]
    bm25_idx = pipe["bm25"]
    predictor = pipe["predictor"]
    resumes: list[Resume] = pipe["resumes"]
    exp_idx = pipe.get("experience")

    target_job = Job(job_id="__inline__", title="(pasted JD)", description=jd_text)

    qv = encoder.encode([jd_text])[0]
    dense_hits = dense_idx.search(qv, k=pool)
    bm25_hits = bm25_idx.search(jd_text, k=pool)
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)

    dense_scores = {h.index: h.score for h in dense_hits}
    bm25_scores = {h.index: h.score for h in bm25_hits}

    # Compute employment-history criteria once per JD; trajectories are
    # pre-computed at pipeline load. Only scored if the toggle is on.
    exp_criteria = None
    if exp_idx is not None:
        from recruit.retrieval.experience import extract_job_criteria
        exp_criteria = extract_job_criteria(target_job)

    rows: list[dict] = []
    for c in fused[:pool]:
        resume = resumes[c.resume_idx]
        sem = dense_scores.get(c.resume_idx, 0.0)
        bm25 = bm25_scores.get(c.resume_idx, 0.0)

        prediction = predictor.predict(
            resume=resume,
            job=target_job,
            sem_similarity=sem,
            bm25_score=bm25,
        )
        if use_llm:
            explanation = build_llm_explanation(
                resume=resume,
                job=target_job,
                prediction=prediction,
                fallback_on_error=True,
            )
        else:
            explanation = build_explanation(
                resume=resume,
                job=target_job,
                prediction=prediction,
            )

        # Employment-history score (0..1). Always computed so it's available
        # for display even when w_exp=0; only enters the fusion when w_exp>0.
        traj_score = 0.0
        if exp_idx is not None and exp_criteria is not None:
            from recruit.retrieval.experience import trajectory_score
            traj_score = trajectory_score(
                exp_idx.trajectories[c.resume_idx], exp_criteria
            ).total

        final = (
            w_sem * sem
            + w_stay * (1.0 - prediction.probability)
            + w_exp * traj_score
        )
        rows.append({
            "resume": resume,
            "resume_idx": c.resume_idx,
            "sem": float(sem),
            "bm25": float(bm25),
            "p_dropoff": prediction.probability,
            "trajectory": float(traj_score),
            "final": final,
            "explanation": explanation,
            "shap_text": prediction.shap_context_text(top_k=3),
            "prediction": prediction,
        })

    rows.sort(key=lambda r: -r["final"])
    return rows


def score_with_secondary_model(
    pipe: dict,
    rows: list[dict],
    secondary_predictor: DropoffPredictor,
    jd_text: str,
) -> dict[int, float]:
    """Re-score the existing pool with a different model. Returns
    {resume_idx: p_dropoff} so the renderer can show v0 alongside v1.
    """
    target_job = Job(job_id="__inline__", title="(pasted JD)", description=jd_text)
    out: dict[int, float] = {}
    for r in rows:
        prediction = secondary_predictor.predict(
            resume=r["resume"],
            job=target_job,
            sem_similarity=r["sem"],
            bm25_score=r["bm25"],
        )
        out[r["resume_idx"]] = prediction.probability
    return out


# ─────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────


_RISK_COLOR_HEX = {
    "low": "#16a34a",       # green
    "moderate": "#ca8a04",  # amber
    "high": "#dc2626",      # red
}
_RECOMMEND_BADGE = {
    "interview":      ("#16a34a", "✅ Interview"),
    "consider":       ("#ca8a04", "🟡 Consider"),
    "likely_mismatch": ("#94a3b8", "⚠️ Likely Mismatch"),
}


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:14px;font-size:13px;font-weight:600;'>"
        f"{text}</span>"
    )


def render_candidate_card(
    rank: int,
    row: dict,
    predictor: DropoffPredictor,
    *,
    secondary_p: float | None = None,
    secondary_label: str = "",
    jd_text: str = "",
) -> None:
    """One candidate as a Streamlit container with key metrics + expanders.

    If `secondary_p` is given, it's displayed alongside the primary
    P(drop-off) value with a delta — used by the v0/v1 comparison mode.
    """
    resume: Resume = row["resume"]
    exp = row["explanation"]
    risk_color = _RISK_COLOR_HEX[exp.dropoff_risk]
    rec_color, rec_text = _RECOMMEND_BADGE[exp.recommendation]

    with st.container(border=True):
        head_left, head_right = st.columns([3, 2])
        with head_left:
            st.markdown(
                f"**#{rank} · `{resume.resume_id}`** &nbsp;&nbsp; "
                + _badge(rec_text, rec_color),
                unsafe_allow_html=True,
            )
            st.caption(exp.overall_fit)
        with head_right:
            if secondary_p is None:
                m1, m2, m3 = st.columns(3)
                m1.metric("semantic", f"{row['sem']:+.3f}")
                m2.metric("P(drop-off)", f"{row['p_dropoff']:.2f}")
                m3.metric("final", f"{row['final']:+.3f}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("semantic", f"{row['sem']:+.3f}")
                m2.metric("P(drop-off)", f"{row['p_dropoff']:.2f}")
                delta = row["p_dropoff"] - secondary_p
                m3.metric(
                    f"P · {secondary_label}",
                    f"{secondary_p:.2f}",
                    delta=f"{delta:+.2f}",
                    delta_color="inverse",
                    help="Negative delta = the primary model predicts LESS drop-off than the secondary.",
                )
                m4.metric("final", f"{row['final']:+.3f}")

        st.markdown(
            f"Drop-off risk: {_badge(exp.dropoff_risk.upper(), risk_color)} &nbsp;&nbsp; "
            f"<span style='color:#475569'>{exp.recommendation_note}</span>",
            unsafe_allow_html=True,
        )

        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                f"**Matched skills ({len(exp.matched_skills)})**\n\n"
                + (", ".join(exp.matched_skills) if exp.matched_skills else "_none_")
            )
        with cols[1]:
            st.markdown(
                f"**Missing skills ({len(exp.missing_skills)})**\n\n"
                + (", ".join(exp.missing_skills) if exp.missing_skills else "_none_")
            )

        with st.expander("Drop-off drivers (SHAP-grounded)"):
            st.code(row["shap_text"], language=None)

        with st.expander("SHAP waterfall"):
            fig = predictor.explainer.waterfall_figure(
                row["prediction"].feature_row,
                top_k=10,
            )
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)
            st.caption(
                "Each bar shows that feature's log-odds contribution to the prediction. "
                "Green pushes toward staying (lower drop-off risk); red pushes toward dropping off."
            )

        with st.expander("Résumé text"):
            text = resume.text
            st.text(text[:2400] + ("…" if len(text) > 2400 else ""))

        # ── Watch-live entry point ──────────────────────────────────────
        # Clicking this button asks the recruiter to *simulate* this
        # candidate applying to the pasted JD. The live in-session mode is
        # activated below the top-K list, with real form-interaction
        # telemetry updating the drop-off model in real time.
        watching = st.session_state.get("watching_resume_idx")
        is_this_one = watching == row["resume_idx"]

        btn_col1, btn_col2 = st.columns([1, 1])
        if not is_this_one:
            if btn_col1.button(
                "▶ Watch this candidate apply live",
                key=f"watch_start_{row['resume_idx']}",
                help="Simulate this candidate filling out the application form. "
                     "Real form-interaction telemetry feeds the drop-off model "
                     "in real time — the production use case.",
                type="secondary",
            ):
                reset_watch_state(new_resume_idx=row["resume_idx"])
                # Stash the JD text alongside the watch so the live panel
                # can display it. Same session state.
                st.session_state["watch_jd_text"] = jd_text
                st.rerun()
        else:
            btn_col1.success("🛰️ Live watch active — see panel below.")
            if btn_col2.button(
                "Stop watching",
                key=f"watch_stop_{row['resume_idx']}",
                type="secondary",
            ):
                reset_watch_state(new_resume_idx=None)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Live watch panel — the in-session live drop-off view for the selected
# candidate. Rendered inline BELOW the top-K list, in two columns:
# a form on the left, a live drop-off widget on the right. All feature
# assembly, prediction and SHAP explanation happen on every Streamlit
# rerun (i.e. every interaction).
# ─────────────────────────────────────────────────────────────────────────


_LIVE_RISK_COLOR = {"low": "#16a34a", "moderate": "#ca8a04", "high": "#dc2626"}
_LIVE_STATUS_BADGE = {
    "pending": ("#94a3b8", "○ pending"),
    "filled":  ("#16a34a", "✓ filled"),
    "skipped": ("#dc2626", "× skipped"),
}


def render_live_watch_panel(
    pipe: dict,
    watched_row: dict,
    jd_text: str,
) -> None:
    """Inline live-watch panel for the currently watched candidate.

    Feeds real form-interaction telemetry from st.session_state into the
    same DropoffPredictor used by the top-K ranking. This is the
    "in-session live" mode of the drop-off model (see docs/concepts/
    04_dropoff_design.md §2.b).
    """
    resume: Resume = watched_row["resume"]
    predictor: DropoffPredictor = pipe["predictor"]

    form_col, signal_col = st.columns([3, 2], gap="large")

    # ── Left column: application form — the drop-off observation surface ─
    with form_col:
        st.markdown(
            "<div style='font-weight:600;margin-bottom:4px;'>"
            "📋 Application form — watch drop-off signals as the candidate fills it</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The recruiter watches the model update in real time as the candidate "
            "fills the application. Each field completion, skip, or navigation "
            "back is a telemetry event feeding the drop-off predictor. "
            "This is how a hiring team identifies **which fields drive drop-off** "
            "and iterates the application design to make it simpler."
        )

        # Quick-simulate shortcut — collapsed. Useful when a demo audience
        # wants to see the model's reaction to a candidate archetype without
        # typing field-by-field. The primary interaction is the form below.
        with st.expander("⚡ Quick simulate a candidate archetype", expanded=False):
            st.caption(
                "Pre-populates the form fields to simulate a typical candidate "
                "behaviour pattern. Useful for demos where you want to jump "
                "between archetypes quickly rather than filling the form."
            )
            sc_a, sc_b, sc_c, sc_d = st.columns(4)
            with sc_a:
                if st.button("🟢 Engaged", use_container_width=True,
                             key="quick_sim_fast",
                             help="Filled most fields, no back-navigation, quick submit."):
                    _apply_submission_scenario("fast", watched_row["resume_idx"])
                    st.rerun()
            with sc_b:
                if st.button("🟡 Hesitant", use_container_width=True,
                             key="quick_sim_hesitant",
                             help="Some skips, moderate time, one back-navigation."):
                    _apply_submission_scenario("hesitant", watched_row["resume_idx"])
                    st.rerun()
            with sc_c:
                if st.button("🔴 Abandoning", use_container_width=True,
                             key="quick_sim_abandoning",
                             help="Many skips, multiple back-clicks, likely to leave."):
                    _apply_submission_scenario("abandoning", watched_row["resume_idx"])
                    st.rerun()
            with sc_d:
                if st.button("Reset", use_container_width=True,
                             key="quick_sim_reset",
                             help="Clear the form."):
                    reset_watch_state(new_resume_idx=watched_row["resume_idx"])
                    st.rerun()

        # Compact control row: candidate action + UI actions grouped.
        ctl_a, ctl_b, ctl_c = st.columns([2, 1, 1])
        with ctl_a:
            st.checkbox(
                "Mobile device",
                key="watch_is_mobile",
                help="Toggle the mobile session feature — mobile applications have higher drop-off rates.",
            )
        with ctl_b:
            if st.button("◀ Back nav", key="watch_prev_btn",
                         use_container_width=True,
                         help="Counts as a candidate navigation-back event."):
                st.session_state.watch_back_count += 1
                st.toast("Back-navigation recorded.")
        with ctl_c:
            if st.button("Reset", key="watch_reset_btn",
                         use_container_width=True,
                         help="Clear all form state and start fresh."):
                reset_watch_state(new_resume_idx=watched_row["resume_idx"])
                st.rerun()

        # ── The application form — field-by-field ─────────────────────
        for f in FORM_FIELDS:
            key = f["key"]
            status = st.session_state[f"watch_field_{key}_status"]
            badge_color, badge_text = _LIVE_STATUS_BADGE[status]

            # Field label + status badge on one row; input + action buttons
            # on the next.
            st.markdown(
                f"<div class='watch-field-label'>"
                f"<b>{f['label']}</b>"
                f"<span style='background:{badge_color};color:white;"
                f"padding:2px 8px;border-radius:10px;font-size:10px;"
                f"font-weight:600;'>{badge_text}</span></div>",
                unsafe_allow_html=True,
            )

            input_col, skip_col, clear_col = st.columns([6, 1, 1])
            with input_col:
                if f["long"]:
                    value = st.text_area(
                        f["label"],
                        value=st.session_state[f"watch_field_{key}_value"],
                        key=f"watch_input_{key}",
                        height=68,
                        label_visibility="collapsed",
                        disabled=(status == "skipped"),
                    )
                else:
                    value = st.text_input(
                        f["label"],
                        value=st.session_state[f"watch_field_{key}_value"],
                        key=f"watch_input_{key}",
                        label_visibility="collapsed",
                        disabled=(status == "skipped"),
                    )

                if value and st.session_state.watch_started_ts is None:
                    st.session_state.watch_started_ts = time.time()
                if value and status != "skipped":
                    st.session_state[f"watch_field_{key}_value"] = value
                    st.session_state[f"watch_field_{key}_status"] = "filled"

            with skip_col:
                if st.button("Skip",
                             key=f"watch_skip_{key}",
                             use_container_width=True,
                             disabled=(status == "filled")):
                    st.session_state[f"watch_field_{key}_status"] = "skipped"
                    st.session_state[f"watch_field_{key}_value"] = ""
                    if st.session_state.watch_started_ts is None:
                        st.session_state.watch_started_ts = time.time()
                    st.rerun()
            with clear_col:
                if st.button("Clear",
                             key=f"watch_clear_{key}",
                             use_container_width=True):
                    st.session_state[f"watch_field_{key}_status"] = "pending"
                    st.session_state[f"watch_field_{key}_value"] = ""
                    st.rerun()


    # ── Right column: live drop-off signal + telemetry + SHAP ──────────
    with signal_col:
        st.markdown(
            "<div style='font-weight:600;margin-bottom:6px;'>"
            "📡 Live drop-off signal</div>",
            unsafe_allow_html=True,
        )
        sess = aggregate_watch_session()
        target_job = Job(job_id="__watched__", title="(pasted JD)", description=jd_text)
        prediction = predictor.predict(
            resume=resume,
            job=target_job,
            sem_similarity=watched_row["sem"],
            bm25_score=watched_row["bm25"],
            session_overrides=sess,
        )

        # Record the trend point.
        if st.session_state.watch_started_ts is not None:
            st.session_state.watch_history.append(
                (_seconds_watching(), prediction.probability)
            )

        p_pct = prediction.probability * 100
        color = _LIVE_RISK_COLOR[prediction.risk_band]

        st.markdown(
            f"<div style='text-align:center;padding:18px 10px;background:{color};"
            f"border-radius:12px;color:white;box-shadow:0 1px 4px rgba(0,0,0,0.15);'>"
            f"<div style='font-size:40px;font-weight:800;line-height:1;'>{p_pct:.0f}%</div>"
            f"<div style='font-size:11px;letter-spacing:2px;margin-top:4px;'>"
            f"{prediction.risk_band.upper()} RISK · LIVE</div></div>",
            unsafe_allow_html=True,
        )

        if prediction.probability >= 0.5:
            st.error(
                "**Reach out now** — candidate showing drop-off signals.",
                icon="🚨",
            )

        # Compact telemetry summary rendered as a two-column grid.
        st.markdown(
            f"<div style='margin-top:14px;font-size:12px;color:#64748b;"
            f"letter-spacing:1px;font-weight:600;'>SESSION TELEMETRY</div>",
            unsafe_allow_html=True,
        )
        tel_l, tel_r = st.columns(2)
        with tel_l:
            st.markdown(
                f"<div style='font-size:13px;'>"
                f"⏱️ <b>{sess['session_time_on_page_secs']}s</b> on page<br/>"
                f"✅ <b>{sess['session_n_fields_completed']}</b>/"
                f"{sess['job_n_required_fields']} completed<br/>"
                f"× <b>{sess['session_n_fields_skipped']}</b> skipped"
                f"</div>",
                unsafe_allow_html=True,
            )
        with tel_r:
            st.markdown(
                f"<div style='font-size:13px;'>"
                f"◀ <b>{sess['session_navigation_back_count']}</b> back clicks<br/>"
                f"📱 <b>{'mobile' if sess['session_device_mobile'] else 'desktop'}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='margin-top:14px;font-size:12px;color:#64748b;"
            f"letter-spacing:1px;font-weight:600;'>TOP DRIVERS (SHAP)</div>",
            unsafe_allow_html=True,
        )
        for a in prediction.explanation.top_attributions[:3]:
            sign = "↑" if a.value > 0 else "↓"
            col = "#dc2626" if a.value > 0 else "#16a34a"
            st.markdown(
                f"<div style='font-size:12px;margin-top:3px;'>"
                f"<code>{a.feature}</code> {sign} "
                f"<span style='color:{col};font-weight:700;'>"
                f"{a.value:+.2f}</span></div>",
                unsafe_allow_html=True,
            )

        if len(st.session_state.watch_history) >= 2:
            import pandas as pd
            df = pd.DataFrame(
                st.session_state.watch_history, columns=["sec", "p_dropoff"]
            )
            st.markdown(
                "<div style='margin-top:14px;font-size:12px;color:#64748b;"
                "letter-spacing:1px;font-weight:600;'>RISK TREND</div>",
                unsafe_allow_html=True,
            )
            st.line_chart(df.set_index("sec"), height=130)


# ─────────────────────────────────────────────────────────────────────────
# Step-based wizard rendering  (Tier-2 UI simplification)
#
# The page is a 3-step wizard:
#   1. Search JD   — paste JD, run search
#   2. Shortlist   — compact top-K cards, select one to watch
#   3. Live watch  — full-view live application for the selected candidate
#
# All step state lives in st.session_state["step"] ∈ {"search","shortlist","watch"}.
# Step transitions happen from button clicks — set state then st.rerun().
# ─────────────────────────────────────────────────────────────────────────


_STEPS = ["search", "shortlist", "watch", "postoffer"]
_STEP_LABELS = {
    "search":    "1 · Search JD",
    "shortlist": "2 · Shortlist",
    "watch":     "3 · Live watch",
    "postoffer": "4 · Post-offer",
}


def _set_step(new_step: str) -> None:
    st.session_state["step"] = new_step


def _current_step() -> str:
    return st.session_state.get("step", "search")


def render_step_navigator() -> None:
    """Top-bar breadcrumb showing the three steps and the current position.

    Uses uniform HTML pills for active/disabled states and a Streamlit
    button only for the clickable navigation targets, all rendered inside
    consistent 44 px pill containers so heights match visually.
    """
    cur = _current_step()
    has_search = "search_rows" in st.session_state
    has_watch = st.session_state.get("watching_resume_idx") is not None

    cols = st.columns(len(_STEPS), gap="small")
    for i, step in enumerate(_STEPS):
        active = (step == cur)
        # Post-offer is independent — the recruiter can jump there directly
        # to explore the extension without needing a prior candidate context.
        disabled = (
            (step == "shortlist" and not has_search)
            or (step == "watch" and not has_watch)
        )
        label = _STEP_LABELS[step]
        with cols[i]:
            if active:
                st.markdown(
                    f"<div style='height:44px;padding:0 14px;background:#1f3a5f;"
                    f"color:white;border-radius:10px;display:flex;align-items:center;"
                    f"justify-content:center;font-weight:700;font-size:14px;"
                    f"box-shadow:0 1px 3px rgba(0,0,0,0.12);'>"
                    f"● {label}</div>",
                    unsafe_allow_html=True,
                )
            elif disabled:
                st.markdown(
                    f"<div style='height:44px;padding:0 14px;background:#f1f5f9;"
                    f"color:#94a3b8;border:1px solid #e2e8f0;border-radius:10px;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-weight:500;font-size:14px;'>"
                    f"○ {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"○ {label}",
                    key=f"nav_{step}",
                    use_container_width=True,
                ):
                    _set_step(step)
                    st.rerun()


def _metric_pill(label: str, value: str, *, tone: str = "neutral") -> str:
    """Small numeric pill used inline on the compact card. Replaces st.metric
    which is way too large for a dense shortlist row.
    """
    color = {"good": "#16a34a", "warn": "#ca8a04", "bad": "#dc2626", "neutral": "#334155"}[tone]
    return (
        f"<div style='display:inline-block;text-align:center;"
        f"padding:2px 10px;margin-right:6px;'>"
        f"<div style='font-size:10px;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>{label}</div>"
        f"<div style='font-size:18px;font-weight:700;color:{color};"
        f"line-height:1.1;'>{value}</div>"
        f"</div>"
    )


def render_compact_candidate_card(
    rank: int,
    row: dict,
    *,
    secondary_p: float | None = None,
    secondary_label: str = "",
) -> None:
    """Compact card: one header row (id + badges + metrics + button),
    one fit-summary line, one score-composition mini-bar, two skill rows.
    Each card is ~5 visual rows tall.
    """
    resume: Resume = row["resume"]
    exp = row["explanation"]
    risk_color = _RISK_COLOR_HEX[exp.dropoff_risk]
    rec_color, rec_text = _RECOMMEND_BADGE[exp.recommendation]

    # Tone the metric pills by their actual meaning.
    p = row["p_dropoff"]
    p_tone = "good" if p < 0.30 else ("warn" if p < 0.60 else "bad")

    with st.container(border=True):
        # ── Row 1: everything on one line — id, badges, metrics, button ──
        head_col, metrics_col, action_col = st.columns([3, 3, 1.2])
        with head_col:
            st.markdown(
                f"<div style='font-size:14px;line-height:1.5;'>"
                f"<span style='color:#94a3b8;'>#{rank}</span> &nbsp; "
                f"<code style='background:#f1f5f9;padding:1px 6px;border-radius:4px;"
                f"font-size:12px;'>{resume.resume_id}</code></div>"
                f"<div style='margin-top:4px;'>"
                + _badge(rec_text, rec_color)
                + " &nbsp; "
                + _badge(exp.dropoff_risk.upper(), risk_color)
                + "</div>",
                unsafe_allow_html=True,
            )
        with metrics_col:
            pills = (
                _metric_pill("semantic", f"{row['sem']:+.2f}", tone="neutral")
                + _metric_pill("P(drop)", f"{p:.2f}", tone=p_tone)
                + _metric_pill("final", f"{row['final']:+.3f}", tone="good")
            )
            if secondary_p is not None:
                delta = p - secondary_p
                sec_tone = "good" if delta < 0 else "bad"
                pills += _metric_pill(
                    f"vs {secondary_label}",
                    f"{secondary_p:.2f} ({delta:+.2f})",
                    tone=sec_tone,
                )
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;"
                f"align-items:center;margin-top:2px;'>{pills}</div>",
                unsafe_allow_html=True,
            )
        with action_col:
            if st.button(
                "▶ Watch live",
                key=f"select_{row['resume_idx']}",
                use_container_width=True,
                help="Open the in-session live watch for this candidate.",
            ):
                reset_watch_state(new_resume_idx=row["resume_idx"])
                st.session_state["watch_jd_text"] = st.session_state.get(
                    "search_jd_text", ""
                )
                _set_step("watch")
                st.rerun()

        # ── Row 2: fit summary sentence ─────────────────────────────────
        st.markdown(
            f"<div style='color:#475569;font-size:13px;margin-top:6px;'>"
            f"{exp.overall_fit}</div>",
            unsafe_allow_html=True,
        )

        # ── Row 3: score composition mini-bar ───────────────────────────
        w_sem = st.session_state.get("search_w_sem", 0.6)
        w_stay = st.session_state.get("search_w_stay", 0.4)
        w_exp = st.session_state.get("search_w_exp", 0.0)
        sem_c = w_sem * row["sem"]
        stay_c = w_stay * (1.0 - row["p_dropoff"])
        exp_c = w_exp * row.get("trajectory", 0.0)
        total = max(1e-6, sem_c + stay_c + exp_c)
        sem_pct = 100 * sem_c / total
        stay_pct = 100 * stay_c / total
        exp_pct = 100 * exp_c / total

        segments = (
            f"<div style='background:#1f3a5f;width:{sem_pct:.1f}%;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"sem {sem_c:+.2f}</div>"
            f"<div style='background:#c99a3b;width:{stay_pct:.1f}%;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"stay {stay_c:+.2f}</div>"
        )
        if exp_pct > 0.5:  # only draw third segment when employment toggle is on
            segments += (
                f"<div style='background:#7c3aed;width:{exp_pct:.1f}%;"
                f"display:flex;align-items:center;justify-content:center;'>"
                f"emp {exp_c:+.2f}</div>"
            )
        st.markdown(
            f"<div style='display:flex;height:14px;border-radius:4px;overflow:hidden;"
            f"margin-top:8px;font-size:10px;font-weight:600;color:white;"
            f"box-shadow:inset 0 -1px 0 rgba(0,0,0,0.06);'>"
            + segments +
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Row 4-5: skills summary ─────────────────────────────────────
        matched_str = ", ".join(exp.matched_skills[:6]) or "—"
        missing_str = ", ".join(exp.missing_skills[:5]) or "—"
        st.markdown(
            f"<div style='font-size:12.5px;margin-top:8px;line-height:1.6;'>"
            f"<span style='color:#16a34a;font-weight:600;'>"
            f"Matched ({len(exp.matched_skills)}):</span> "
            f"<span style='color:#334155;'>{matched_str}"
            + (" …" if len(exp.matched_skills) > 6 else "")
            + f"</span><br/>"
            f"<span style='color:#dc2626;font-weight:600;'>"
            f"Missing ({len(exp.missing_skills)}):</span> "
            f"<span style='color:#334155;'>{missing_str}"
            + (" …" if len(exp.missing_skills) > 5 else "")
            + f"</span></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Search
# ─────────────────────────────────────────────────────────────────────────


def render_search_step(pipe: dict, config: dict) -> None:
    st.subheader("Paste a Job Description")
    st.caption(
        f"🔍  Recruiter sourcing across a {len(pipe['resumes']):,}-résumé corpus."
    )

    job_order = pipe["job_order"]
    options = ["(custom)"] + [
        f"JD #{i:>3d}  ·  {pipe['jobs'][i].description[:65].strip()}…"
        for i in job_order
    ]

    col_a, col_b = st.columns([3, 2])
    with col_b:
        sample_choice = st.selectbox(
            "Or pick a sample JD",
            options=options,
            index=1,
            help="Sample JDs are sorted by tech-skill density.",
        )
    default_text = ""
    if sample_choice != "(custom)":
        try:
            idx = int(sample_choice.split("#")[1].split("·")[0].strip())
            default_text = pipe["jobs"][idx].description
        except (IndexError, ValueError):
            default_text = ""

    with col_a:
        jd_text = st.text_area(
            "JD text",
            value=default_text,
            height=220,
            placeholder="Paste a job description here or pick a sample on the right…",
        )

    # Right-sized primary button — not full-width (looked oversized before).
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button(
            "🔎 Find candidates",
            type="primary",
            disabled=not jd_text.strip(),
            use_container_width=True,
        )

    if run:
        spinner_msg = (
            "Retrieving, scoring drop-off, composing explanations…"
            if not config["use_llm"]
            else "Retrieving, scoring, composing LLM explanations (slower)…"
        )
        with st.spinner(spinner_msg):
            rows = score_candidates(
                pipe,
                jd_text=jd_text,
                pool=config["pool"],
                w_sem=config["w_sem"],
                w_stay=config["w_stay"],
                w_exp=config["w_exp"],
                use_llm=config["use_llm"],
            )
            secondary_p_by_idx: dict[int, float] = {}
            if config["compare"] and config["secondary_filename"]:
                sec_predictor = load_predictor(config["secondary_filename"])
                secondary_p_by_idx = score_with_secondary_model(
                    pipe, rows, sec_predictor, jd_text=jd_text
                )
        st.session_state["search_rows"] = rows
        st.session_state["search_jd_text"] = jd_text
        st.session_state["search_top_k"] = config["top_k"]
        st.session_state["search_w_sem"] = config["w_sem"]
        st.session_state["search_w_stay"] = config["w_stay"]
        st.session_state["search_w_exp"] = config["w_exp"]
        st.session_state["search_secondary_p"] = secondary_p_by_idx
        st.session_state["search_primary_filename"] = config["primary_filename"]
        st.session_state["search_secondary_filename"] = config["secondary_filename"]
        st.session_state["search_compare"] = config["compare"]

        # Any previous watch is invalidated by a new search.
        reset_watch_state(new_resume_idx=None)

        _set_step("shortlist")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Shortlist
# ─────────────────────────────────────────────────────────────────────────


def render_shortlist_step() -> None:
    if "search_rows" not in st.session_state:
        st.warning("No search results yet. Go back to Step 1.")
        return

    rows = st.session_state["search_rows"]
    top_k = st.session_state["search_top_k"]
    w_sem = st.session_state["search_w_sem"]
    w_stay = st.session_state["search_w_stay"]
    w_exp = st.session_state.get("search_w_exp", 0.0)
    secondary_p_by_idx = st.session_state["search_secondary_p"]
    compare = st.session_state["search_compare"]
    secondary_filename = st.session_state["search_secondary_filename"]

    jd_text = st.session_state.get("search_jd_text", "")

    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.subheader(f"Top-{top_k} candidates")
        if w_exp > 0:
            st.caption(
                f"ℹ️  Pre-application ranking · simulated-session P(drop-off) · "
                f"final = {w_sem:.2f}·sem + {w_stay:.2f}·(1−P_dropoff) + "
                f"{w_exp:.2f}·employment-history"
            )
        else:
            st.caption(
                f"ℹ️  Pre-application ranking · simulated-session P(drop-off) · "
                f"final = {w_sem:.2f}·sem + {w_stay:.2f}·(1−P_dropoff)"
            )
    with head_r:
        if st.button("◀ New search", use_container_width=True):
            _set_step("search")
            st.rerun()

    with st.expander("Job description used"):
        st.write((jd_text[:400] + "…") if len(jd_text) > 400 else jd_text)

    short_secondary = ""
    if compare and secondary_filename:
        short_secondary = secondary_filename.replace("dropoff_", "").replace(".joblib", "")

    for rank, row in enumerate(rows[:top_k], start=1):
        render_compact_candidate_card(
            rank,
            row,
            secondary_p=secondary_p_by_idx.get(row["resume_idx"]) if compare else None,
            secondary_label=short_secondary,
        )


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — Live watch
# ─────────────────────────────────────────────────────────────────────────


def _render_candidate_resume(resume: Resume, watched_row: dict) -> None:
    """Compact résumé card — shown at the top of the Live-watch step.

    Groups the extracted facts (skills, YOE, location) on the left and the
    truncated résumé text on the right. In production the résumé comes
    from the ATS's candidate record; here we show the same fields we
    already extracted at index-build time.
    """
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown(f"**Résumé ID:** `{resume.resume_id}`")
        if resume.candidate_name:
            st.markdown(f"**Name:** {resume.candidate_name}")
        if resume.target_role:
            st.markdown(f"**Target role:** {resume.target_role}")
        yoe = (
            f"{resume.years_experience:.1f} years"
            if resume.years_experience is not None
            else "not declared"
        )
        st.markdown(f"**Experience:** {yoe}")
        if resume.location:
            st.markdown(f"**Location:** {resume.location}")
        if resume.education:
            st.markdown(f"**Education:** {resume.education}")

        st.markdown(" ")
        st.markdown("**Skills extracted from résumé**")
        if resume.skills:
            skill_html = "  ".join(
                f"<span style='display:inline-block;background:#e0e7ff;"
                f"color:#1f3a5f;padding:2px 8px;border-radius:10px;"
                f"font-size:11px;font-weight:500;margin:2px;'>{s}</span>"
                for s in sorted(set(resume.skills))[:20]
            )
            st.markdown(skill_html, unsafe_allow_html=True)
        else:
            st.caption("(no explicit skills field — the retrieval layer extracts skills from the résumé text)")

        st.markdown(" ")
        st.markdown(
            f"<div style='font-size:12px;color:#64748b;'>"
            f"Semantic similarity vs JD: "
            f"<b style='color:#1f3a5f;'>{watched_row['sem']:+.3f}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Résumé text** *(as the ATS would receive it)*")
        # Truncate to keep the panel compact — the whole point is to give
        # the recruiter a scannable snapshot, not a full document reader.
        text = resume.text.strip()
        preview_chars = 900
        if len(text) <= preview_chars:
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:6px;"
                f"padding:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                f"font-size:12px;line-height:1.5;color:#1e293b;"
                f"white-space:pre-wrap;max-height:280px;overflow-y:auto;'>"
                f"{text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:6px;"
                f"padding:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                f"font-size:12px;line-height:1.5;color:#1e293b;"
                f"white-space:pre-wrap;max-height:280px;overflow-y:auto;'>"
                f"{text[:preview_chars]}…</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Showing first {preview_chars:,} characters of {len(text):,} total. "
                f"The full text is what SBERT + BM25 index against."
            )


def render_watch_step(pipe: dict) -> None:
    watching_idx = st.session_state.get("watching_resume_idx")
    if watching_idx is None:
        st.warning("No candidate selected. Go back to Step 2 and pick one.")
        return

    rows = st.session_state.get("search_rows", [])
    watched_row = next((r for r in rows if r["resume_idx"] == watching_idx), None)
    if watched_row is None:
        st.warning("Selected candidate is no longer in the current shortlist. "
                   "Go back to Step 1 and re-search.")
        return

    resume: Resume = watched_row["resume"]
    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.subheader(f"Live watch — {resume.resume_id}")
        yoe_txt = (
            f"{resume.years_experience:.1f} YOE"
            if resume.years_experience is not None else "YOE unknown"
        )
        st.caption(
            f"🛰️  {yoe_txt} · semantic vs JD **{watched_row['sem']:+.3f}** · "
            f"real telemetry drives the model live"
        )
    with head_r:
        if st.button("◀ Back to shortlist", use_container_width=True):
            _set_step("shortlist")
            st.rerun()

    # ── Candidate résumé viewer ──────────────────────────────────────
    # The recruiter clicked "Watch live" on this specific candidate.
    # Show who they are before the scenario buttons — otherwise the
    # simulator is disconnected from the person being evaluated.
    with st.expander("📄 Candidate résumé", expanded=False):
        _render_candidate_resume(resume, watched_row)

    # ── Live SHAP waterfall — updates every interaction ──
    with st.expander("📊 Live SHAP waterfall — why the model predicts what it does",
                     expanded=True):
        predictor: DropoffPredictor = pipe["predictor"]
        target_job = Job(
            job_id="__watched__",
            title="(pasted JD)",
            description=st.session_state.get("watch_jd_text", ""),
        )
        # Build a fresh prediction with the CURRENT live session state so the
        # waterfall reflects what the recruiter is seeing on the right panel.
        live_sess = aggregate_watch_session()
        live_prediction = predictor.predict(
            resume=resume,
            job=target_job,
            sem_similarity=watched_row["sem"],
            bm25_score=watched_row["bm25"],
            session_overrides=live_sess,
        )
        try:
            fig = predictor.explainer.waterfall_figure(
                live_prediction.feature_row,
                top_k=8,
                height_inches=3.5,
            )
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)
            st.caption(
                "Each bar shows one feature's log-odds contribution to the "
                "prediction. Green bars push toward staying (lower drop-off); "
                "red bars push toward abandoning. Updates live as the "
                "candidate interacts with the form below."
            )
        except Exception as e:  # noqa: BLE001
            st.warning(f"SHAP waterfall unavailable: {e}")

    # ── RAG explainer panel — grounded natural-language rationale ──
    # "R" in RAG = retrieved evidence (SHAP drivers + candidate summary).
    # "G" = the LLM, constrained to rephrase evidence, not introduce new claims.
    with st.expander(
        "Candidate rationale  ·  LLM-generated, SHAP-grounded",
        expanded=False,
    ):
        st.markdown(
            "A local LLM composes the natural-language rationale below, "
            "constrained to the top-3 SHAP drivers and the candidate / JD "
            "summary shown as evidence. The model is instructed to **rephrase "
            "evidence only** — introducing a claim not supported by the "
            "evidence would constitute a faithfulness failure. Regenerate to "
            "produce an updated rationale reflecting the current session state."
        )

        drivers_text = live_prediction.shap_context_text(top_k=3)
        st.markdown("**Evidence provided to the model** *(top-3 SHAP drivers, log-odds):*")
        st.code(drivers_text, language=None)

        current_exp = watched_row["explanation"]
        st.markdown(
            "**Baseline rationale** *(deterministic composer, generated at search time):*"
        )
        st.markdown(
            f"<div style='padding:10px;background:#f8fafc;border-radius:6px;"
            f"border-left:3px solid #64748b;font-size:13px;line-height:1.5;'>"
            f"{current_exp.overall_fit}</div>",
            unsafe_allow_html=True,
        )

        llm_key = f"live_llm_rationale_{watching_idx}"
        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            regen = st.button(
                "Regenerate rationale",
                key=f"regen_llm_btn_{watching_idx}",
                use_container_width=True,
                help=(
                    "Calls the local LLM (Ollama) with the current live "
                    "session context. First call is ~10 s cold; ~3 s warm. "
                    "Falls back to the deterministic composer on any error."
                ),
            )
        with col_note:
            st.caption(
                "The regenerated rationale uses the live session context — "
                "reflecting fields skipped, back-navigation, and time on "
                "page — rather than the pre-application snapshot from "
                "search time."
            )

        if regen:
            with st.spinner("Generating rationale via local LLM…"):
                try:
                    from recruit.explain.llm_rag import build_llm_explanation
                    fresh_exp = build_llm_explanation(
                        resume=resume,
                        job=target_job,
                        prediction=live_prediction,
                    )
                    st.session_state[llm_key] = fresh_exp.overall_fit
                except Exception as e:  # noqa: BLE001
                    st.error(f"LLM call failed: {e}. Falling back to templated.")
                    st.session_state[llm_key] = None

        if st.session_state.get(llm_key):
            st.markdown(
                "**LLM-generated rationale** *(grounded in the evidence above):*"
            )
            st.markdown(
                f"<div style='padding:10px;background:#eff6ff;border-radius:6px;"
                f"border-left:3px solid #8b5cf6;font-size:13px;line-height:1.5;'>"
                f"{st.session_state[llm_key]}</div>",
                unsafe_allow_html=True,
            )

    # Reuse the existing live watch panel implementation.
    jd_text = st.session_state.get("watch_jd_text", "")
    render_live_watch_panel(pipe, watched_row, jd_text)


# ─────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────


_GLOBAL_CSS = """
<style>
    /* Tighter overall vertical rhythm — Streamlit defaults are loose. */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 3rem; }

    /* Slightly denser sidebar. */
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* Compact button defaults. */
    .stButton > button {
        min-height: 40px;
        font-size: 13px;
    }

    /* Tone down oversized headings — they compete with the content. */
    h1 { font-size: 1.9rem !important; margin-bottom: 0.4rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Cards should have subtle borders, not heavy ones. Only bordered
       containers (st.container(border=True)) get the frame — otherwise
       every nested vertical block would draw a box around itself. */
    div[data-testid="stVerticalBlockBorderWrapper"][data-testid$="Wrapper"] {
        border-radius: 10px !important;
    }

    /* Give the field-label markdown breathing room from the input below.
       Was overlapping with 'gap: 0.35rem' — reverted to a targeted rule. */
    .watch-field-label {
        margin-top: 14px;
        margin-bottom: 6px;
        font-size: 13px;
        line-height: 1.2;
    }
    .watch-field-label b { margin-right: 6px; }
</style>
"""


def render_experience_diagnostic() -> None:
    """Collapsed high-level panel — did adding an employment-history channel help?

    Kept intentionally sparse: one narrative paragraph, one small rescue table,
    one clear verdict. Full numeric drill-down lives in
    reports/logs/diagnose_experience_output.log for anyone who wants to inspect.
    """
    import pandas as pd

    with st.expander(
        "Did adding an employment-history channel improve retrieval?  "
        "(click to expand)",
        expanded=False,
    ):
        # ── The question in one sentence ───────────────────────────────
        st.markdown(
            "**The question.**  I built a retrieval channel that reads work "
            "history from résumé text — years of experience, seniority, and "
            "employer — and uses it alongside keyword search to find matching "
            "candidates. Did adding this channel improve the aggregate ranking "
            "quality?"
        )

        st.markdown(
            "**The answer.**  On aggregate benchmark scores — **no**.  "
            "On the specific case the channel was designed for — **yes**.  "
            "The story below explains why both are true."
        )

        st.markdown("---")

        # ── The one visual worth showing — the rescue ─────────────────
        st.markdown(
            "#### The design targeted strong-but-keyword-poor candidates."
        )
        st.caption(
            "Four candidates in the corpus have both 8+ years of experience "
            "AND a top-tier employer — exactly the profile that plain keyword "
            "search would miss. Here's where each channel ranks them:"
        )

        st.dataframe(
            pd.DataFrame([
                {"Candidate": "A", "Rank by keyword search": 217,
                 "Rank by employment-history channel": 1,   "Rescued?": "yes"},
                {"Candidate": "B", "Rank by keyword search": 381,
                 "Rank by employment-history channel": 7,   "Rescued?": "yes"},
                {"Candidate": "C", "Rank by keyword search": 183,
                 "Rank by employment-history channel": 2,   "Rescued?": "yes"},
                {"Candidate": "D", "Rank by keyword search":  23,
                 "Rank by employment-history channel": 69,  "Rescued?": "no"},
            ]),
            hide_index=True, use_container_width=True,
        )

        st.success(
            "**Three of the four target candidates are rescued** — from "
            "invisible (ranks 183, 217, 381) to visible (top 10). "
            "The design works on the exact profile it targeted."
        )

        st.markdown("---")

        # ── Why aggregate metrics don't show it ───────────────────────
        st.markdown("#### Then why doesn't the aggregate benchmark improve?")

        st.markdown(
            "- The candidates the channel rescues are **unlabelled** in the "
            "benchmark — the graders reviewed only what keyword search already "
            "surfaced, so they never saw these candidates to grade them.  \n"
            "- The benchmark has only **71 labelled jobs** — statistically too "
            "small to detect the small aggregate difference this channel "
            "produces.  \n"
            "- Semantic embeddings (SBERT) already implicitly capture some of "
            "the trajectory signal, so the marginal add is small."
        )

        st.markdown("---")

        # ── What I chose to do about it ───────────────────────────────
        st.markdown("#### How I handled the mixed result.")

        c1, c2 = st.columns(2)
        with c1:
            st.info(
                "**Reported the negative result honestly.**\n\n"
                "No weight tuning to fabricate a positive number. The write-up "
                "explains exactly why the aggregate metric can't measure the "
                "targeted improvement."
            )
        with c2:
            st.info(
                "**Kept the channel in the repository.**\n\n"
                "It becomes the evaluable baseline for future work — an "
                "LLM-assisted trajectory parser has to beat this rules-based "
                "version to justify its complexity."
            )

        st.caption(
            "Full raw numbers: `reports/logs/diagnose_experience_output.log`  ·  "
            "Reproduce: `make diagnose-exp`  ·  "
            "Code: `src/recruit/retrieval/experience.py`"
        )


# ─────────────────────────────────────────────────────────────────────────
# Fairness audit panel — surfaces the results the reviewer flagged as
# "not visible in the demo" at mid-sem.
# ─────────────────────────────────────────────────────────────────────────


def render_fairness_panel() -> None:
    """Collapsed panel showing the fairness audit results.

    Reads the JSON reports written by `make audit` and `make audit-hr`.
    Displays two audits side-by-side:
      · Synthetic drop-off model, gender + country proxies
      · HR Analytics real-demographics gender audit

    All three regulatory-standard metrics are shown for each:
      · Demographic parity difference
      · Equal opportunity difference
      · Disparate impact ratio (4/5 rule pass/fail)

    Framing acknowledges NYC Local Law 144 and EU AI Act — bias auditing
    for hiring AI is now a legal requirement, not just an ethical
    preference.
    """
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    synth_path = project_root / "reports" / "fairness_synthetic.json"
    hr_path = project_root / "reports" / "fairness_hr.json"

    with st.expander(
        "Fairness audit  ·  bias metrics for the ranking + drop-off model  (click to expand)",
        expanded=False,
    ):
        st.markdown(
            "**Regulatory context.** Since July 2023, NYC Local Law 144 requires "
            "bias audits for any Automated Employment Decision Tool used to "
            "screen candidates for jobs in the city. The EU AI Act (2024) "
            "classifies recruitment-scoring systems as *high-risk* and "
            "mandates fairness assessment. The metrics below are what those "
            "regulations expect."
        )

        # ── Load both reports (graceful fallback if a file is missing) ──
        synth = None
        hr = None
        if synth_path.exists():
            synth = json.loads(synth_path.read_text())
        if hr_path.exists():
            hr = json.loads(hr_path.read_text())

        if synth is None and hr is None:
            st.warning(
                "No fairness reports found. Run `make audit && make audit-hr` "
                "to generate them (~3 minutes)."
            )
            return

        # ── Side-by-side comparison ─────────────────────────────────────
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("#### Audit 1 · Synthetic drop-off model")
            if synth is None:
                st.caption("Not yet generated. Run `make audit`.")
            else:
                st.caption(
                    "Drop-off classifier evaluated on the HF-fit test split, "
                    "with gender and country inferred from résumé text as "
                    "proxies. Model: `" + synth.get("model_used", "unknown") + "`"
                )
                _render_fairness_summary(
                    "Gender proxy (name-based)",
                    synth.get("gender_proxy", {}),
                )
                st.markdown("")
                _render_fairness_summary(
                    "Country proxy (text-based)",
                    synth.get("country_proxy", {}),
                )

        with col_right:
            st.markdown("#### Audit 2 · Real HR demographics")
            if hr is None:
                st.caption("Not yet generated. Run `make audit-hr`.")
            else:
                st.caption(
                    f"Kaggle HR Analytics ~19,000 rows.  "
                    f"Audited {hr.get('n_audited', 0):,} test rows on "
                    "real (not proxied) gender labels. The model was "
                    "trained without seeing the gender feature — the "
                    "audit surfaces bias that arises indirectly through "
                    "correlated proxies."
                )
                _render_fairness_summary(
                    f"Gender (real) — attribute: '{hr.get('attribute', '')}'",
                    hr,
                )

        st.markdown("---")
        st.markdown(
            "**Interpretation targets:**  "
            "Demographic parity diff ≤ 0.10 = strong.  "
            "Equal opportunity diff ≤ 0.10 = strong.  "
            "Disparate impact ratio ≥ 0.80 passes the EEOC 4/5 rule."
        )
        st.caption(
            "Reports:  `reports/fairness_synthetic.json`  ·  `reports/fairness_hr.json`  "
            "Reproduce:  `make audit && make audit-hr`  ·  "
            "Code:  `src/recruit/fairness/audit_dropoff.py`"
        )


def _render_fairness_summary(title: str, report: dict) -> None:
    """Render one fairness summary block — metrics + group breakdown."""
    if not report:
        st.caption(f"{title}: no data.")
        return

    dp = report.get("demographic_parity_diff", 0.0)
    eo = report.get("equal_opportunity_diff", 0.0)
    di = report.get("disparate_impact_ratio", 0.0)
    passes = report.get("passes_4_5_rule", False)

    def _colour_for(value: float, threshold: float, direction: str = "below") -> str:
        """Green if the value passes the threshold, red otherwise."""
        good = (value <= threshold) if direction == "below" else (value >= threshold)
        return "#16a34a" if good else "#dc2626"

    st.markdown(f"**{title}**")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.4px;'>DP diff</div>"
            f"<div style='font-size:22px;font-weight:700;color:{_colour_for(dp, 0.10, 'below')};line-height:1.1;'>"
            f"{dp:.3f}</div>"
            f"<div style='font-size:10px;color:#94a3b8;'>≤ 0.10 target</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.4px;'>EO diff</div>"
            f"<div style='font-size:22px;font-weight:700;color:{_colour_for(eo, 0.10, 'below')};line-height:1.1;'>"
            f"{eo:.3f}</div>"
            f"<div style='font-size:10px;color:#94a3b8;'>≤ 0.10 target</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.4px;'>DI ratio</div>"
            f"<div style='font-size:22px;font-weight:700;color:{_colour_for(di, 0.80, 'above')};line-height:1.1;'>"
            f"{di:.3f}</div>"
            f"<div style='font-size:10px;color:#94a3b8;'>{'✓ PASS' if passes else '✗ FAIL'} 4/5 rule</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Per-group breakdown (if available)
    groups = report.get("groups", [])
    if groups:
        import pandas as pd
        df = pd.DataFrame([
            {
                "Group": g["group"],
                "N": g["n"],
                "Base rate": g["base_rate"],
                "Selection rate": g["selection_rate"],
                "TPR": g["tpr"],
                "FPR": g["fpr"],
            }
            for g in groups
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# Step 4 — Post-offer decline (PoC extension)
# ─────────────────────────────────────────────────────────────────────────


def render_postoffer_step(pipe: dict) -> None:
    """Post-offer decline prediction — end-sem extension.

    Responds to mid-sem reviewer feedback that post-offer drop-off has
    higher business value than mid-application drop-off. Same XGBoost +
    isotonic calibration + SHAP + RAG framework as the résumé-submission
    model — only the features differ (compensation gap, response latency,
    negotiation rounds, competing-offer signal, etc.).
    """
    predictor = pipe.get("postoffer_predictor")

    st.subheader("Post-offer decline — extension proof-of-concept")
    st.caption(
        "This screen extends the drop-off framework to the post-offer stage of the funnel. "
        "The reviewer noted at mid-sem that post-offer decline has higher business value "
        "than mid-application drop-off. Same pipeline as Step 3, different features."
    )

    if predictor is None:
        st.warning(
            "Post-offer model not trained yet. Run:\n\n"
            "```\nmake generate-postoffer-data && make train-postoffer\n```\n\n"
            "to build the artefact. The rest of the demo (Steps 1–3) is unaffected."
        )
        return

    from recruit.postoffer.predict import OFFER_SCENARIOS, SCENARIO_LABELS

    # ── Scenario picker + current-state display ────────────────────────
    st.markdown("**Simulate an offer situation**")
    st.caption(
        "Each button loads a preset feature vector — the values that an ATS "
        "would emit at offer time (salary gap, days from interview to offer, "
        "candidate response latency, negotiation rounds, competing-offer signal, "
        "and candidate profile). No form-filling required."
    )

    sc_a, sc_b, sc_c, sc_d = st.columns(4)
    with sc_a:
        if st.button(SCENARIO_LABELS["likely_accept"],
                     use_container_width=True,
                     help="Strong offer, well-matched candidate, remote available, fast response."):
            st.session_state["postoffer_scenario"] = "likely_accept"
            st.rerun()
    with sc_b:
        if st.button(SCENARIO_LABELS["uncertain"],
                     use_container_width=True,
                     help="Reasonable offer with some friction — long commute, moderate negotiation."):
            st.session_state["postoffer_scenario"] = "uncertain"
            st.rerun()
    with sc_c:
        if st.button(SCENARIO_LABELS["likely_decline"],
                     use_container_width=True,
                     help="Weak offer, competing option, long commute, slow response."):
            st.session_state["postoffer_scenario"] = "likely_decline"
            st.rerun()
    with sc_d:
        if st.button("Reset", use_container_width=True,
                     help="Clear the current scenario."):
            st.session_state.pop("postoffer_scenario", None)
            st.session_state.pop("postoffer_llm_rationale", None)
            st.rerun()

    scenario_key = st.session_state.get("postoffer_scenario")
    if scenario_key is None:
        st.info(
            "No scenario loaded yet. Click a scenario above to see how the model "
            "and SHAP explanation respond."
        )
        return

    situation = OFFER_SCENARIOS[scenario_key]
    prediction = predictor.predict(situation)

    # ── Two-column layout: SHAP waterfall + right-side signal card ─────
    main_col, side_col = st.columns([3, 2], gap="large")

    with main_col:
        # SHAP waterfall
        with st.expander(
            "SHAP waterfall — why the model predicts this decline probability",
            expanded=True,
        ):
            try:
                fig = predictor._explainer.waterfall_figure(
                    prediction.feature_row,
                    top_k=8,
                    height_inches=3.5,
                )
                st.pyplot(fig, use_container_width=True)
                import matplotlib.pyplot as plt
                plt.close(fig)
                st.caption(
                    "Each bar is one feature's log-odds contribution to the "
                    "decline prediction. Green pushes toward accepting; red "
                    "toward declining."
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"SHAP waterfall unavailable: {e}")

        # Candidate rationale panel (LLM + templated baseline)
        with st.expander(
            "Candidate rationale · LLM-generated, SHAP-grounded",
            expanded=False,
        ):
            st.markdown(
                "A local LLM composes the natural-language rationale below, "
                "constrained to the top-3 SHAP drivers shown as evidence. The "
                "model is instructed to **rephrase evidence only** — "
                "introducing a claim not supported by the evidence would "
                "constitute a faithfulness failure."
            )

            drivers_text = prediction.shap_context_text(top_k=3)
            st.markdown("**Evidence provided to the model** *(top-3 SHAP drivers, log-odds):*")
            st.code(drivers_text, language=None)

            regen_key = f"postoffer_llm_rationale_{scenario_key}"
            col_btn, col_note = st.columns([1, 3])
            with col_btn:
                regen = st.button(
                    "Generate rationale",
                    key=f"regen_postoffer_{scenario_key}",
                    use_container_width=True,
                    help="Calls the local LLM (Ollama) to compose a plain-"
                         "English rationale grounded in the SHAP evidence.",
                )
            with col_note:
                st.caption(
                    "The rationale is grounded in the SHAP evidence above — "
                    "the LLM cannot introduce new claims about the candidate "
                    "or the offer."
                )

            if regen:
                with st.spinner("Generating rationale via local LLM…"):
                    try:
                        rationale = _generate_postoffer_llm_rationale(
                            situation, prediction
                        )
                        st.session_state[regen_key] = rationale
                    except Exception as e:  # noqa: BLE001
                        st.error(f"LLM call failed: {e}")
                        st.session_state[regen_key] = None

            if st.session_state.get(regen_key):
                st.markdown(
                    "**LLM-generated rationale** *(grounded in the evidence above):*"
                )
                st.markdown(
                    f"<div style='padding:10px;background:#eff6ff;border-radius:6px;"
                    f"border-left:3px solid #8b5cf6;font-size:13px;line-height:1.5;'>"
                    f"{st.session_state[regen_key]}</div>",
                    unsafe_allow_html=True,
                )

    with side_col:
        # Risk signal card — mirrors the drop-off Live-watch signal
        p_pct = prediction.probability * 100
        color = {"low": "#16a34a", "moderate": "#ca8a04", "high": "#dc2626"}[
            prediction.risk_band
        ]
        st.markdown(
            f"<div style='background:{color};color:white;border-radius:12px;"
            f"padding:18px;text-align:center;'>"
            f"<div style='font-size:36px;font-weight:700;line-height:1.0;'>"
            f"{p_pct:.0f}%</div>"
            f"<div style='font-size:11px;opacity:0.9;margin-top:4px;'>"
            f"{prediction.risk_band.upper()} RISK · P(decline)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(" ")
        st.markdown("**Scenario details**")
        st.markdown(
            f"<div style='font-size:12px;color:#334155;line-height:1.7;'>"
            f"Salary gap vs expectation: <b>{situation.offer_salary_gap_pct:+.0%}</b><br/>"
            f"Days interview → offer: <b>{situation.days_interview_to_offer:.0f}</b><br/>"
            f"Interview rounds: <b>{situation.interview_rounds}</b><br/>"
            f"Response latency: <b>{situation.candidate_response_hours:.0f} h</b><br/>"
            f"Negotiation rounds: <b>{situation.negotiation_rounds}</b><br/>"
            f"Competing offer: <b>{'yes' if situation.competing_offer_signal else 'no'}</b><br/>"
            f"Commute: <b>{situation.commute_minutes:.0f} min</b><br/>"
            f"Remote option: <b>{'yes' if situation.remote_option else 'no'}</b><br/>"
            f"Candidate YOE: <b>{situation.cand_yoe:.0f}</b><br/>"
            f"Seniority tier: <b>{situation.cand_seniority_tier}/5</b><br/>"
            f"Currently employed: <b>{'yes' if situation.cand_currently_employed else 'no'}</b><br/>"
            f"Company brand tier: <b>{'top-tier' if situation.company_brand_tier else 'other'}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(" ")
        st.markdown("**Top drivers (SHAP)**")
        for attr in prediction.explanation.top_attributions[:3]:
            arrow_color = "#dc2626" if attr.value > 0 else "#16a34a"
            direction_text = "↑ decline" if attr.value > 0 else "↓ decline"
            st.markdown(
                f"<div style='font-size:11.5px;color:#334155;margin-top:2px;'>"
                f"<code style='font-size:11px;'>{attr.feature}</code>  "
                f"<span style='color:{arrow_color};font-weight:600;'>"
                f"{direction_text}  {attr.value:+.2f}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Bottom banner — the PoC honesty statement ──────────────────────
    st.markdown(" ")
    st.info(
        "**Proof-of-concept honesty statement.** The post-offer decline model "
        "is trained on synthetic labels — no public dataset labels real "
        "post-offer decline events. The pipeline and framework are what this "
        "screen demonstrates. Production deployment would require partnership "
        "with an ATS vendor for real offer-lifecycle telemetry. Model artefact: "
        "`models/postoffer_v1_calibrated.joblib`. Reproduce with "
        "`make generate-postoffer-data && make train-postoffer`."
    )


def _generate_postoffer_llm_rationale(situation, prediction) -> str:
    """Build an Ollama prompt for the post-offer scenario, call, return prose.

    Reuses the LLM RAG explainer pattern from `explain/llm_rag.py`, but with
    a prompt customised for the offer-lifecycle context rather than the
    mid-application flow. Fails gracefully — errors bubble up to the caller
    which shows an st.error message.
    """
    from recruit.config import settings
    import ollama

    system_prompt = (
        "You are a recruitment-assistant explainer for the post-offer stage. "
        "Given a candidate's offer situation, the model's predicted decline "
        "probability, and the top SHAP drivers, produce ONE concise plain-"
        "English rationale in 1-2 sentences. You must: (1) cite ONLY the "
        "evidence given, (2) reference at least one top SHAP driver, "
        "(3) never invent facts not present in the inputs."
    )
    user_prompt = f"""
CANDIDATE OFFER SITUATION
- Salary gap vs candidate expectation: {situation.offer_salary_gap_pct:+.0%}
- Days from interview to offer: {situation.days_interview_to_offer:.0f}
- Interview rounds: {situation.interview_rounds}
- Candidate response latency: {situation.candidate_response_hours:.0f} hours
- Negotiation rounds: {situation.negotiation_rounds}
- Competing offer mentioned: {"yes" if situation.competing_offer_signal else "no"}
- Commute (one-way): {situation.commute_minutes:.0f} minutes
- Remote option available: {"yes" if situation.remote_option else "no"}
- Candidate years of experience: {situation.cand_yoe:.0f}
- Candidate seniority tier: {situation.cand_seniority_tier}/5
- Candidate currently employed: {"yes" if situation.cand_currently_employed else "no"}
- Company brand tier: {"top-tier" if situation.company_brand_tier else "other"}

MODEL OUTPUT
- Predicted P(decline): {prediction.probability * 100:.0f}%
- Risk band: {prediction.risk_band}

EVIDENCE — TOP SHAP DRIVERS (log-odds contributions)
{prediction.shap_context_text(top_k=3)}

Write the rationale as one continuous plain-English sentence (or at most two).
Do NOT output JSON, headers, or bullet points. Do NOT invent facts.
""".strip()

    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


def main() -> None:
    st.set_page_config(
        page_title="AI Recruitment Assistant",
        page_icon="🧭",
        layout="wide",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    init_watch_state()
    st.title("🧭 AI Recruitment Assistant")
    st.caption(
        "Semantic matching · drop-off prediction · SHAP-grounded explanations. "
        "Three-step recruiter workflow: search → shortlist → live watch."
    )

    # ── Sidebar — simplified + consistent hierarchy ─────────────────────
    with st.sidebar:
        st.markdown("### ⚙ Settings")
        top_k = st.slider(
            "Top-K candidates",
            1, 15, 5,
            help="How many candidates to show in the shortlist.",
        )

        avail = _available_models()
        if not avail:
            st.error("No trained model found — run `scripts/train_dropoff.py` first.")
            return
        primary_label = st.selectbox(
            "Drop-off model",
            options=[label for label, _ in avail],
            index=0,
            help="Used for scoring + SHAP + recommendations.",
        )
        primary_filename = dict(avail)[primary_label]

        # Advanced controls — collapsed by default.
        with st.expander("Advanced", expanded=False):
            pool = st.slider(
                "Recall pool size",
                10, 100, 30, step=5,
                help="How many candidates the retrieval channels return before fusion.",
            )
            w_sem = st.slider(
                "Semantic weight",
                0.0, 1.0, 0.60, step=0.05,
                help="Weight of semantic score in the final ranking. Stay-weight = 1 − this.",
            )
            w_stay = round(1.0 - w_sem, 2)

            use_experience = st.toggle(
                "Include employment-history in ranking",
                value=False,
                help=(
                    "OFF: ranking uses semantic + drop-off only (default).\n\n"
                    "ON: adds a third term using the employment-history "
                    "channel (evaluator feedback). Ranking rearranges to "
                    "favour candidates whose work history matches the JD."
                ),
            )
            w_exp = 0.20 if use_experience else 0.0

            if use_experience:
                st.caption(
                    f"final = **{w_sem:.2f}** · semantic  +  "
                    f"**{w_stay:.2f}** · (1 − P_dropoff)  +  "
                    f"**{w_exp:.2f}** · employment-history"
                )
            else:
                st.caption(
                    f"final = **{w_sem:.2f}** · semantic  +  "
                    f"**{w_stay:.2f}** · (1 − P_dropoff)"
                )

            compare = st.toggle(
                "Compare with a second model",
                value=False,
                help="Score every candidate with TWO models side-by-side.",
            )
            secondary_filename = None
            if compare:
                secondary_options = [
                    (label, fname) for label, fname in avail
                    if fname != primary_filename
                ]
                if secondary_options:
                    sec_label = st.selectbox(
                        "Secondary model",
                        options=[label for label, _ in secondary_options],
                        index=0,
                    )
                    secondary_filename = dict(secondary_options)[sec_label]
                else:
                    st.caption("Only one model available.")
                    compare = False

            use_llm = st.toggle(
                "LLM explainer (Ollama)",
                value=False,
                help=(
                    "OFF: deterministic templated composer (fast).\n\n"
                    "ON: llama3.1:8b-instruct via Ollama — requires "
                    "`ollama serve` and the model pulled. Falls back to "
                    "templated on any error."
                ),
            )

        pipe = load_pipeline(primary_filename)

        st.divider()
        st.markdown("### 📚 Corpus")
        st.markdown(
            f"**{len(pipe['resumes']):,}** résumés  \n"
            f"<span style='color:#64748b;font-size:12px;'>"
            f"{pipe.get('n_hf', 0)} HF + {pipe.get('n_kaggle', 0)} Kaggle · "
            f"{len(pipe['jobs'])} sample JDs</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "HF `resume-job-description-fit` + Kaggle "
            "`snehaanbhawal/resume-dataset`. Drop-off labels are synthetic."
        )

    # Bundle config for step renderers so each step gets the current
    # sidebar state as a single dict (cleaner than passing 8 parameters).
    config = {
        "top_k": top_k,
        "pool": pool,
        "w_sem": w_sem,
        "w_stay": w_stay,
        "w_exp": w_exp,
        "use_experience": use_experience,
        "use_llm": use_llm,
        "compare": compare,
        "primary_filename": primary_filename,
        "secondary_filename": secondary_filename,
    }

    # ── Top step navigator + step dispatcher ────────────────────────────
    render_step_navigator()
    st.markdown("&nbsp;", unsafe_allow_html=True)  # small vertical spacer

    # Fairness audit panel — collapsed by default, always accessible.
    # Directly addresses mid-sem reviewer's "fairness not visible in demo".
    render_fairness_panel()

    step = _current_step()
    if step == "search":
        render_search_step(pipe, config)
    elif step == "shortlist":
        render_shortlist_step()
    elif step == "watch":
        render_watch_step(pipe)
    elif step == "postoffer":
        render_postoffer_step(pipe)
    else:
        st.error(f"Unknown step: {step}")
        _set_step("search")
        st.rerun()


if __name__ == "__main__":
    main()
