# Embedding Input Scaling — Design

**Date:** 2026-08-31
**Status:** Draft — pending approval

## Problem

The game embedding (`config.yaml` → `embeddings: algorithm: svd`, 64-dim,
deployed as `embeddings-v2026`) produces pathological geometry for games with
rare, uncorrelated features. Surfaced by the bgg-viewer `/dev/similar` bench and
its catalog-analysis table:

- **System Gateway** (the only rated fan-expansion) has near-zero cosine
  similarity to *everything* — it sits alone on a near-dedicated component.
- **White Castle / Sankoré / Project L / Inferno** form an artificially tight
  cluster, jointly detached from the catalog, because they're the ~4 popular
  games sharing the "Resource Queue" mechanic (~22 games total).
- Common features (auctions, hand management) embed fine — they're prevalent and
  correlated, so SVD folds them into shared components.

### Root cause

The preprocessor (`create_embedding_preprocessor`, `model_type="linear"`) ends
with a blanket `StandardScaler` over **every** column — the mechanic / category /
family 0/1 dummies included — then `TruncatedSVD`.

`StandardScaler` divides each column by its SD. For a binary feature on `p`
fraction of games, SD = √(p(1−p)); for a feature on 22 of ~35 600 games that SD
is ≈ 0.025, so the 22 games that have it land at ≈ +40 on the scaled axis and
everyone else at ≈ −0.025. For a 1-game feature it's ≈ +188.

`StandardScaler` forces **every** feature to variance 1 regardless of prevalence.
SVD/PCA allocate components by variance, so a rare feature — with a full unit of
variance sitting on its own uncorrelated axis — gets a near-dedicated component.
After L2-normalisation (which cosine similarity implies) the handful of games
with that feature point in a near-unique direction → minimal similarity to
anything.

Min-frequency filtering alone treats the symptom. The mechanism is
`StandardScaler` converting "rare" into "as important as everything else."

## Solution

Adopt **Gelman-style scaling** (Gelman 2008, *Scaling regression inputs by
dividing by two standard deviations*): scale continuous inputs by `(x − mean) /
(2·SD)` and leave binary inputs at raw 0/1. A balanced dummy (p ≈ 0.5) then has
SD ≈ 0.5 and a 2-SD-scaled continuous input also has SD ≈ 0.5 — they're
commensurate — while a rare dummy keeps its natural variance `p(1−p) ≈ p`,
hundreds to thousands of times below a real feature's.

PCA's 64-component budget then simply never spends a component on a feature whose
variance is that small: it lands in the discarded residual, and the games that
have it get their coordinates from their *other* features. No per-feature
special-casing, no threshold tuning to make it work.

Three changes, in the embedding pipeline only:

1. `TwoSDScaler` — a shared transformer replacing the blanket `StandardScaler`.
2. Switch the embedding algorithm `svd` → `pca` with `whiten=False` (the dummies
   are no longer centred, and PCA centres internally).
3. A minimum-frequency floor on dummy features (reliability, not the core fix).

A fourth lever — value clipping — is **documented as a fallback, not built** (see
below).

## Components

### 1. `TwoSDScaler` — `src/features/transformers.py`

New `BaseEstimator, TransformerMixin` alongside `LogTransformer` /
`YearTransformer` / `PrefixColumnDropper` (the house pattern). Reusable by any
preprocessor, not embedding-specific.

- `__init__(continuous_columns: list[str] | None = None, factor: float = 2.0)`
  — the columns to standardise-by-`factor`-SD; everything else passes through
  untouched. `None` → auto-detect (columns with > 2 distinct values / not
  strictly {0, 1}); an explicit list is preferred for stability.
- `fit` stores per-column mean and SD for the continuous set.
- `transform`: `(x − mean) / (factor · SD)` for continuous columns, identity for
  the rest. Returns a DataFrame (pipeline uses `set_output(transform="pandas")`).
- Picklable and stable — embedding + downstream models are pickled artifacts.
- Unit tests: continuous column gets mean 0 / SD 0.5; 0/1 column is byte-identical
  through the transform; unknown columns pass through; `fit`/`transform` on
  disjoint frames (train vs score) is consistent.

### 2. Embedding pipeline — `src/models/embeddings/transformer.py`

In `create_embedding_preprocessor`, `model_type == "linear"` branch:

- Replace the final `("scaler", StandardScaler())` with
  `("scaler", TwoSDScaler(continuous_columns=<resolved list>))`.
- Keep `LogTransformer` and `YearTransformer` as-is (they run before the scaler;
  the log already compresses playtime / word-count outliers).
- The continuous set is **game numerics + description-embedding dims**
  (`emb_*`, present when `use_embeddings: true`); the binary set is the
  mechanic / category / family dummies **and** the `SimpleImputer(add_indicator=
  True)` missing-indicators. Enumerating the exact split at this pipeline stage
  is a plan step (see Open questions).

### 3. Algorithm swap — `config.yaml` + verify `PCAEmbedding`

- `config.yaml` → `embeddings.algorithm: pca`, and
  `embeddings.algorithms.pca.whiten: false` (currently `true`).
- `PCAEmbedding` already exists (`algorithms.py`) and already emits
  `components` / `feature_names` / `top_features_per_component` via
  `get_artifacts()`. Confirm it takes `svd_solver="randomized"` (or set it) —
  centring densifies the matrix (~35 k × [dummies + ~100 emb dims]); randomized
  PCA keeps that tractable.
- Rationale for `whiten=False`: whitening rescales every component to unit
  variance, re-inflating exactly the low-variance directions this change
  suppresses, and is wrong for a cosine-similarity space (you want the
  high-variance structural axes to dominate the metric).
- Alternative considered: keep `TruncatedSVD` + insert
  `StandardScaler(with_mean=True, with_std=False)` before it. Mathematically
  identical to PCA; rejected as less clear than using the class that exists.

### 4. Minimum-frequency floor — `src/models/embeddings/transformer.py`

Drop dummy features present on fewer than **N** games (N ≈ 10–15, informed by the
component diagnostic below) before the decomposition. Either a small
`MinFrequencySelector` transformer or a tightened `VarianceThreshold` threshold
targeting binary columns (`threshold = p(1−p)` at the chosen `p`). With the new
scaling this is a reliability / matrix-width measure — a feature on 8 games can't
be estimated — not the geometry fix.

### 5. Component diagnostic (deliverable)

A script / notebook that loads the fitted PCA and prints, per component: top
features by |loading|, each with its **prevalence**, plus a concentration metric
(`max(loading)² / Σ loading²`). Used to (a) confirm no component is
dominated by a sub-100-game feature after the change, (b) set N in #4, (c)
compare old vs new. This is the standing tool for the "is a rare feature
capturing a dimension" question.

## Validation

- **Intrinsic — the bench.** Re-run the new embedding through bgg-viewer
  `/dev/similar` + the catalog-analysis table. System Gateway's isolation score
  should fall into the normal range; the White Castle / Project L / Inferno
  cluster should dissolve (they're not similar games). Spot-check that genuine
  neighbours (Catan → Settlers-likes, etc.) are unchanged or better.
- **Component diagnostic** (#5): no post-change component dominated by a
  low-prevalence feature.
- **Explained variance / scree**: compare old vs new; a modest drop in total
  explained variance is expected and fine (we're declining to chase rare-feature
  variance).
- **Reconstruction / neighbour stability**: for a panel of unambiguous games
  (Catan, Gloomhaven, Wingspan, …) the new nearest-neighbour lists should be
  recognisably the same or better — the change should fix the tails, not churn
  the core.
- **Derived tables:** after regenerating `raw.game_embeddings`, the
  `analytics.game_similarity_search` enrichment and the 2D coordinates job
  rebuild; the bgg-data-warehouse `game_neighbors` profile rebuilds off the new
  similarity table. Spot-check a few `game_neighbors` lists for sanity.

## Sequencing & risks

- **Blast radius is the similar-games surface only.** The game SVD embedding
  (`raw.game_embeddings` → `analytics.game_similarity_search` → `game_neighbors`
  → the bgg-viewer game page) and the 2D coordinates job are its *only*
  consumers. The predictive models (`complexity`, `rating`, `users_rated`,
  `geek_rating`) consume the **text** embeddings (`bgg_description_embeddings`,
  PMI/SIF) via `use_embeddings` — confirmed in `src/data/loader.py`, which joins
  `predictions.bgg_description_embeddings`, not `raw.game_embeddings`. **No
  rating-model refit.**
- **Ship as a new experiment version** anyway (registry pattern —
  `embeddings-v2026` → e.g. `embeddings-v2027`): train, evaluate on the bench +
  analysis table + component diagnostic, and only then regenerate the derived
  BigQuery tables / promote. Nothing repoints until the new version clears
  evaluation.
- **Text-embedding dims on the continuous side.** ~100 `emb_*` dims (the text
  embedding, an *input* to the game embedding) will be 2-SD-scaled as continuous
  and will outnumber the ~7 game numerics. They're already a dense learned
  representation; 2-SD scaling keeps them commensurate with a balanced dummy,
  which is the intent — but confirm this vs. scaling that block separately
  (Open question).
- **PCA memory.** Centring densifies; use randomized solver, verify peak memory
  in the training job.
- **Adopting `TwoSDScaler` in the deployed predictive models** (`create_bgg_
  preprocessor`) is **out of scope** here — Gelman's argument (comparable
  coefficients) applies most directly to the linear rating models, but each is a
  deployed pickled artifact needing its own refit + eval, and none of them are
  affected by this change. Tracked as a separate follow-up; tree models are
  scale-invariant and get nothing.

## What this doesn't change

- Feature engineering: `EmbeddingTransformer`, `DEFAULT_EMBEDDING_FAMILY_
  PATTERNS`, description-embedding inclusion, `max_family_features` — untouched.
- `embedding_dim` stays 64.
- The predictive models (`complexity`, `rating`, `users_rated`, `geek_rating`) —
  they consume the **text** embeddings, not this one; nothing about them moves.
- `create_bgg_preprocessor` and the predictive-model pipelines.
- The text-embedding pipeline (`text_embeddings`, PMI/SIF) — unchanged; it's an
  input to this embedding, not a target.
- The BigQuery embedding tables / schema (new rows under a new version, same
  shape); `game_similarity_search` and the coordinates job regenerate off the
  new embedding once it's promoted.
- No value clipping is added (documented fallback only): the dummies are now
  bounded to ≈ [−1, 1] by construction, and the continuous columns are already
  log-compressed. Revisit only if the diagnostic shows a specific column still
  distorting a component, in which case winsorise that raw input rather than
  clip in the 2-SD space.

## Open questions for the plan

1. Exact continuous-vs-binary column split at the scaler stage — enumerate,
   including imputer missing-indicators (binary) and `emb_*` dims (continuous).
2. `TwoSDScaler` column selection: explicit list vs. auto-detect vs. both.
3. Minimum-frequency N — set from the component diagnostic on the current model.
4. `MinFrequencySelector` transformer vs. tuned `VarianceThreshold`.
5. New embedding version name, and the order of operations for regenerating
   `game_similarity_search` + coordinates + the `game_neighbors` profile.
6. Do the `emb_*` text-embedding dims get the same 2-SD treatment or a separate
   block scale?
7. Delivery: branch `feat/embedding-input-scaling` → PR to `main`; the plan
   states the full branch/PR workflow. User reviews and merges.
