# Concept Note 01 — The Data Layer

> This note pairs with `src/recruit/data/` and `scripts/demo_data_layer.py`.
> It becomes part of the **Methodology chapter** of your thesis.

---

## What we built

Three small things:

1. **`schemas.py`** — Pydantic models for `Resume`, `Job`, and `Match`. These are *contracts*: every other module in the codebase consumes typed objects, not dictionaries or DataFrames.

2. **`loaders.py`** — Functions that read CSV files and emit lists of validated `Resume` / `Job` objects. The loader is the **only** place in the codebase that knows about raw file formats — swap the dataset, rewrite this file, nothing else changes.

3. **`data/sample/*.csv`** — Eight résumés and six jobs committed to the repo as fixtures. Real Kaggle datasets get downloaded to `data/raw/` which is gitignored.

## Concepts to defend in your viva

### Why a schema layer at all?

In a research prototype it is tempting to throw pandas DataFrames around everywhere. That's fine until your matcher crashes because someone passed `years_experience` as a string. With Pydantic:

- **Validation at the boundary.** Bad rows fail at load time, with a clear error pointing at the row and column. The matcher never sees an invalid `Resume`.
- **Type-checker assist.** IDE autocomplete and `mypy`-style checking work because the objects have known fields.
- **Thesis vocabulary.** When you write "the matcher consumes a Resume object with fields *resume_id*, *text*, *skills*, *years_experience*, *location*," you are referring to a precise artifact in your code, not a vague concept.

### Why is `frozen=True` set on the models?

Immutable models cannot be mutated after construction. This means parallel embedding workers cannot accidentally corrupt shared state, and reasoning about your code becomes simpler — once a `Resume` exists, it cannot change.

### Why separate `loaders.py` from `schemas.py`?

This is the **anti-corruption layer** pattern from domain-driven design. The schema describes *what your domain looks like*. The loader knows about *messy external data*. Keep them separate so:

- Real datasets with weird column names (e.g. "Job Description" with a space, NaN in skill fields) get cleaned in one place.
- The downstream code stays clean and doesn't need to know whether the data came from Kaggle, O*NET, or a Jobvite export.

### Why ship a tiny `data/sample/` fixture?

Two reasons:

1. **You can develop and test offline.** Before downloading the real ~1 GB Kaggle dataset, you can write the matcher, the predictor, and the UI against eight résumés and six jobs and they'll all work. This dramatically tightens your dev loop.
2. **Examiners can reproduce the demo.** When you submit, the fixtures are in the repo. They can `uv sync && uv run python scripts/demo_data_layer.py` and see something working in under a minute. This is exactly the kind of reproducibility your scope statement (§5) commits to.

## What the examiner might ask

> **"Why didn't you use a database?"**
> For a research prototype with thousands of records, file-based loading is simpler, version-controllable, and reproducible. A production deployment would use the ATS's existing PostgreSQL (and pgvector). I name pgvector in the outline as the production-equivalent vector store.

> **"How do you handle missing data?"**
> The `_split_skills` helper in `loaders.py` tolerates NaN, list, and comma-string forms. Optional schema fields tolerate missing values. Hard fields (resume_id, text) raise validation errors at load time — which is the desired behavior, because we cannot rank a résumé with no text.

> **"What if a real résumé has thousands of words and exceeds the embedding model's context window?"**
> That's handled at the next layer — `embeddings/chunker.py`, which we'll build in slice 2 of the matcher. The data layer's job ends at giving the chunker a clean `Resume.text` string.

## Mini-checklist before moving on to slice 2 (embeddings)

- [ ] Can you run `uv run python scripts/demo_data_layer.py` and see résumé R001 printed?
- [ ] Can you explain in one sentence why we use Pydantic schemas?
- [ ] Can you point to where in the schema the `years_experience` field is, and why it's `Optional[float]`?

Once those three are yes, we move on.
