# Embedding Similarity Evaluation — Design

**Date:** 2026-08-31
**Status:** Draft — pending approval

## Problem

The game embedding has no ground truth. Every change to it (the 2-SD scaling
change in `2026-08-31-embedding-input-scaling-design.md`, and every future
iteration) currently gets judged by eyeballing a few neighbour lists in the
bgg-viewer bench. We need a **repeatable evaluation harness** that produces
quantitative signals and a promote / no-promote decision, so embedding changes
stop being vibes.

Scope: evaluating the **game similarity** that the embedding produces
(`analytics.game_similarity_search` → `game_neighbors` → the viewer). Not the
text embeddings, not the predictive models.

## Solution

A reusable `src/models/embeddings/eval/` module with four evaluation stages,
cheapest and most diagnostic first, plus a documented promote rule. Each stage is
a function that takes two embedding matrices (old, new), aligned to `game_id`,
and returns a metrics dict. A top-level `compare_embeddings(old, new)` runs all
stages and prints a report.

The stages are ordered so a failure short-circuits: no point running the
expensive behavioural comparison if the pathology regression already failed.

### Stage A — Pathology regression (gate)

Confirms the known failures are fixed without new ones. Deterministic, seconds to
run, the CI-style gate for every future embedding.

- **Component diagnostic** (built in the scaling plan —
  `diagnose_components.py`): no component with `min_prevalence_in_top < 0.005`.
- **Known-bad games**: `System Gateway` (game_id TBD) isolation score — mean
  cosine to its 25 nearest non-duplicate games — must be within the catalogue's
  central 90%, not in the extreme tail.
- **Known-bad cluster**: the four "Resource Queue" games (White Castle, Sankoré,
  Project L, Inferno) must not be each other's mutual top-5 *and* jointly in the
  isolated tail.
- **Isolation-tail composition**: of the 50 most-isolated games (min ratings
  100+), the share with < 4 mechanic+category tags must not exceed a threshold
  (calibrate on the current embedding; new should be lower).

Output: pass/fail per check.

### Stage B — Relational validity (quantitative, warehouse data)

BGG relationships the embedding does not directly ingest but similarity should
still respect. Guards against *over*-shrinking real structure.

- **Reimplementation / expansion pairs** — `core.game_implementations`,
  `core.game_expansions` (symmetric). For each pair `(a, b)` where both are in
  the embedding, record `rank_b` = position of `b` in `a`'s full cosine ranking.
  Metrics: median rank, P90 rank, % of partners in top-50, % in top-200.
  **Regression flag:** new median rank > 1.5× old, or top-50 share drops > 10
  points.
- **Shared-designer pairs** (weaker) — sample pairs of games by the same
  designer; same rank metrics. Expect worse-than-reimplementations but
  better-than-random; used only to catch a gross regression.
- **Random-pair baseline** — the rank distribution for random game pairs, so the
  above numbers have a reference.

### Stage C — Behavioural alignment (conditional on a data source)

The real proxy for "similar": do the same users rate both? A content embedding
should track this **better, not worse**, even though the two notions
legitimately differ (behavioural similarity also absorbs hype, release timing,
publisher).

- Build an item–item collaborative similarity from a user × game rating matrix
  (mean-centred cosine, shrinkage on low-overlap pairs).
- For ~500 sampled games: `recall@10` (collaborative top-10 ∩ embedding top-10)
  and Spearman correlation of the two rankings over the top-50.
- Old vs new. **Regression flag:** mean `recall@10` drops materially.

**Data-source open question — this stage is descoped if none is workable:**
`collections.user_collections` only holds the 3 tracked users — not enough. Needs
either (a) a broad per-user ratings table (likely in bgg-data-warehouse — to
confirm), or (b) a scraped sample of BGG's own per-game "recommendations"
(bgg-data-warehouse scraping territory), used as the comparison ranking instead
of a hand-built CF matrix.

### Stage D — Human A/B over a fixed panel (product goal)

Already implemented — the bgg-viewer `/dev/similar` **Evaluate mode**. This spec
just fixes the protocol:

- Load old and new embeddings as experiments A and B.
- Fixed panel: the bench's stratified sample + hard cases (~40 games), frozen for
  the comparison.
- Blind per-source verdict A / = / B; the running tally is the headline.
- Record the panel, the tally, and any notable individual disagreements in the
  scaling-change PR.

No code in this module — a checklist and the panel definition.

### Stage E — Intrinsic / stability (supporting)

- **Neighbour churn** — for an "unambiguous" panel (Catan, Wingspan, Gloomhaven,
  Ark Nova, Codenames, Azul, …), the fraction of each top-10 that changed
  old→new. Want the core stable and the tails fixed; a full reshuffle of obvious
  games is a red flag.
- **Trustworthiness / continuity** (`sklearn.manifold.trustworthiness`) of the
  64-d embedding vs its own input feature space (the Gelman-scaled matrix for the
  new one, the StandardScaled matrix for the old — each against the space it was
  actually fit on). Local-structure preservation; reported for context, not as a
  gate (the whole point of the change is to *not* preserve rare-feature
  structure).

## Components

```
src/models/embeddings/eval/
    __init__.py
    loaders.py         # align two embedding sources to a common game_id index
    pathology.py       # Stage A
    relational.py      # Stage B — reads core.game_implementations / game_expansions
    behavioural.py     # Stage C — item-item CF or scraped-recs comparison
    stability.py       # Stage E — churn, trustworthiness
    compare.py         # compare_embeddings(old, new) -> report; CLI entrypoint
```

- CLI: `python -m src.models.embeddings.eval.compare --old <exp/ver> --new <exp/ver>`.
- Each stage function is independently testable with small synthetic matrices.
- Stage B needs a BigQuery read of the two bridge tables (addressed by full name,
  same as `catalog`/`loader` do — no Dataform source declaration for a read).
- Reused by every future embedding change, not just this one.

## Validation

- Unit tests per stage function on synthetic embeddings where the answer is
  known (e.g. a planted near-duplicate pair → rank 1; a planted isolated point →
  extreme isolation score).
- `compare.py` run end-to-end on **current SVD vs current SVD** (identical
  inputs) → every "regression flag" is false, churn is 0, recall is ~1.0. This is
  the harness's own sanity check.
- Then the real run: current SVD vs the new PCA embedding, output pasted into the
  scaling-change PR.

## Promote / no-promote rule

Promote the new embedding when:

1. **Stage A** — all checks pass (hard gate).
2. **Stage B** — no regression flag (reimplementation median rank within 1.5×,
   top-50 share within 10 points).
3. **Stage C** — recall@10 improves or holds (or Stage C is descoped for lack of
   data, and the decision rests on A/B/D/E).
4. **Stage D** — the human tally favours new (or is neutral with A/B/E clearly
   positive).
5. **Stage E** — core-panel churn is moderate, not a reshuffle.

On promote, the scaling-change follow-up runs: regenerate `raw.game_embeddings` →
`analytics.game_similarity_search` → the bgg-data-warehouse `game_neighbors`
profile → the 2D coordinates job.

## What this doesn't change

- No change to the embedding itself — this is measurement only.
- No new BigQuery tables; Stage B does read-only queries against existing bridge
  tables.
- The bench (`/dev/similar`) is used as-is for Stage D; no viewer changes.
- Does not evaluate the text embeddings or any predictive model.

## Sequencing

This harness is a **sibling** of the scaling-change plan, not a blocker for
starting it:

- Stages A and E can be built alongside or just after the scaling change (they
  need only the embeddings).
- Stage B needs the bridge-table queries — small, independent.
- Stage C is gated on the data-source question and may be dropped.
- Stage D exists today.

A reasonable order: land the scaling change with Stage A + D + a manual Stage E,
then build the full `eval/` module so the *next* embedding iteration has the
whole harness.

## Open questions for the plan

1. Behavioural-similarity data source (Stage C) — broad ratings table location,
   or scrape BGG recs, or drop the stage.
2. `System Gateway` and the Resource-Queue game_ids — pull the exact ids for the
   Stage A fixtures.
3. Isolation-tail sparse-tag threshold — calibrate on the current embedding.
4. The "unambiguous panel" for Stage E — agree the list (~15 games).
5. Where the aligned embedding matrices come from for a not-yet-promoted version
   — a scratch output path vs. re-running inference in the eval CLI.
