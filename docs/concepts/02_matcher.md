# Concept Note 02 — The Semantic Matcher

> Pairs with `src/recruit/embeddings/`, `src/recruit/retrieval/`, and `scripts/demo_matcher.py`.
> Becomes the **retrieval section of your Methodology chapter** + the table-skeleton for the **Evaluation chapter**.

---

## What we built

Five things:

1. **`embeddings/sbert.py`** — A lazy-loading SBERT encoder that batches inputs, L2-normalises outputs, and caches embeddings to disk.
2. **`retrieval/faiss_index.py`** — A FAISS `IndexFlatIP` wrapper that returns top-K cosine-similar items.
3. **`retrieval/bm25.py`** — A `BM25Okapi` wrapper with a vanilla regex tokeniser (deliberately not over-tuned; reflects an out-of-box ATS baseline).
4. **`retrieval/hybrid.py`** — Reciprocal Rank Fusion + structured filters (location, min YOE, required certs).
5. **`scripts/demo_matcher.py`** — Runs all three rankings side-by-side for one job query so you can *see* where they agree and where they diverge.

This is the heart of Objective 2 in your submitted outline.

---

## Concepts to defend in your viva

### Embeddings (recap)

A *sentence embedding* is a fixed-length vector of floats encoding the meaning of a piece of text. Texts with similar meaning land near each other in vector space. We measure similarity with **cosine** — the cosine of the angle between two vectors, magnitude-invariant, range [-1, 1].

### Why SBERT in particular?

Plain BERT is trained for masked-token prediction. Its `[CLS]` token is not designed for sentence similarity — averaging or pooling it gives surprisingly bad results. **SBERT** [Reimers & Gurevych 2019] fine-tunes BERT inside a *siamese network* on NLI and STS data so that cosine similarity becomes a meaningful semantic distance. After training you keep the encoder and discard the siamese scaffolding.

We use `all-MiniLM-L6-v2` by default — 22M parameters, 384-dim output, ~5 ms per encode on CPU. Good speed/quality trade-off, MTEB-validated.

### Why FAISS Flat?

- Corpus < 100k → exact brute-force search runs in milliseconds.
- IVF and HNSW are approximate; they trade recall for speed and need training. Not worth defending in a viva at this scale.
- `IndexFlatIP` (inner product) on unit-length vectors gives exactly cosine similarity, with no extra arithmetic.

### BM25 in 30 seconds

The dominant sparse-retrieval scoring function, used in Lucene/Solr/OpenSearch/Elasticsearch:

```
BM25(d, q) = Σ_{t ∈ q}  IDF(t) · TF_normalised(t, d)
```

- `IDF(t)` = inverse document frequency — rare terms are worth more.
- `TF_normalised(t, d)` = term frequency in *d*, saturated by parameter `k1` and adjusted for document length by parameter `b`.
- Defaults: `k1 = 1.5`, `b = 0.75`. These are what `rank_bm25.BM25Okapi` uses.

BM25 is the production baseline named in your outline §1. **Beating it on Precision@10 and Recall@10 is your committed deliverable** for the retrieval objective.

### Reciprocal Rank Fusion (RRF) in 30 seconds

The fusion rule:

```
RRF_score(d) = Σ_systems  1 / (k_smooth + rank_system(d))
```

`k_smooth = 60` is the default from Cormack et al. 2009 and it just works.

**Why not weighted-sum?** Dense scores are cosine in [-1, 1]; BM25 scores are unbounded. Combining them on raw scores requires per-system normalisation that breaks when score distributions shift. RRF uses only *ranks*, so distribution changes don't matter. It's parameter-light and well-tested in TREC.

### Why structured filters live after ranking

We compute a single fused ranking, then apply filters as a post-processing step. Two reasons:

1. **Index simplicity** — we maintain one FAISS index and one BM25 index, not one per filter combination.
2. **Recruiter UX** — the same résumé can appear ranked for multiple jobs without re-indexing.

For a production deployment with millions of candidates, you'd switch to pre-filtering (cheap field lookups before the expensive vector scoring). Out of scope here.

---

## What the demo shows (and how to read it for your thesis)

Run `uv run python scripts/demo_matcher.py`. You'll see three rankings of résumés against job **J002** ("Machine Learning Engineer - NLP"):

- **Dense (SBERT+FAISS)** is expected to surface R002 Karthik Iyer (Data Scientist, NLP, PyTorch) and R006 Vikram Rao (ML Engineer, search ranking) at the top, even though their résumé text uses *different words* from the JD ("data scientist" vs "ML engineer").
- **BM25** will favour résumés that share more exact tokens with the JD — could still pick R002 because PyTorch/Hugging Face overlap is high, but won't necessarily handle paraphrases.
- **Hybrid + filters** combines both and drops candidates who fail YOE / location / cert constraints. This is the final "recruiter-shown" ranking.

The side-by-side print is exactly the figure you drop into the thesis evaluation chapter to **qualitatively** illustrate dense vs sparse trade-offs *before* you report the quantitative IR metrics (Precision@10, Recall@10, NDCG, MRR).

---

## Likely viva questions

> **"Why MiniLM and not a larger SBERT model?"**
> 384 dim, 22M params, 5ms per encode on CPU. The MTEB benchmark shows it's competitive with much larger models on retrieval. Larger models are an obvious future-work knob to turn.

> **"Why didn't you fine-tune SBERT on your domain?"**
> Three reasons. (1) Out of scope per §5 of the outline. (2) Fine-tuning requires labelled positive/negative pairs, which I don't have at scale. (3) The point of the dissertation is showing that *off-the-shelf* SBERT plus structured ranking plus RAG explanations is enough to beat BM25 — fine-tuning would muddy that claim.

> **"What if BM25 wins?"**
> Then I report it honestly. The thesis claim is that *hybrid* dense+BM25 strictly improves over BM25 alone, which is the established finding in IR. If on my domain dense alone fails to beat BM25, RRF still wins because it benefits from BM25's lexical precision. My evaluation chapter will break down win/loss by query type.

> **"Why post-filter and not pre-filter?"**
> Simplicity, recruiter-UX, and dataset size. A production deployment with 10M+ candidates would switch to pre-filtering. I note this in the limitations chapter.

> **"How does RRF compare to weighted sum?"**
> RRF is rank-based and parameter-light; weighted sum requires per-system score normalisation that's brittle. Cormack et al. 2009 showed RRF dominates weighted-sum on TREC tracks. Default `k_smooth = 60` is the field standard.

---

## Mini-checklist before slice 3 (drop-off predictor)

- [ ] `uv run python scripts/demo_matcher.py` prints three ranked lists.
- [ ] You can explain *cosine similarity* in one sentence ("inner product of L2-normalised vectors; magnitude-invariant; meaningful only on embeddings trained for it").
- [ ] You can write the RRF formula on paper without looking.
- [ ] You can name the BM25 parameters `k1` and `b` and what they control.

Once those four are yes, we move on to drop-off prediction.
