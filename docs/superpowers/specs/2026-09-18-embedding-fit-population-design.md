# Embedding Fit Population — Design Note

**Date:** 2026-09-18
**Status:** Note — not scheduled. Current behaviour is accepted for now; revisit
when the embedding is next retrained.

## What the deployed embedding is fit on

`embeddings-v2026` v6 (= experiment `game-embeddings/v4`, 2026-09-02) is a 64-d
PCA fit on **53,777 games**: `users_rated >= 5`, `year_published <= 2023`.

- The years come from the shared `years.training` block in `config.yaml`
  (`train_through: 2022`, `tune_through: 2023`, `test_through: 2024`). The
  trainer (`src/models/embeddings/trainer.py`, "Refitting model on combined
  train + tune data") refits the final model on train + tune, so the 2024
  test-year games are never in the fit, and `years.current` is 2026.
- The ratings floor is `embeddings.min_ratings: 5`, deliberate (comment: "more
  meaningful data").
- The preprocessor (2·SD Gelman scaling of the 8 continuous inputs, dummy
  vocabulary, `min_feature_count`) is fit on the **train split only**
  (through 2022) and reused as-is for the refit (`prepare_features(...,
  fit=False)`).

Two things follow, neither of which is a bug today:

1. **Every game from 2024 on is projected into a space that has never seen a
   post-2023 game.** That includes all "upcoming" games, which the viewer's
   embedding map places and reasons about. New mechanics/families that only
   appear from 2024 on are absent from the dummy vocabulary entirely.
2. **The component structure is shaped by the whole ≥5-ratings universe** (~54k
   games), not just the ≥30-ratings working set the viewer shows (~31k). This is
   the intended design: the map's job is to show where *every* game sits, and a
   PCA fit only on already-popular games would learn what popular games look
   like rather than the shape of the space. Reading the loadings (2026-09-18
   side-analysis): PC1 = complexity/length, PC2 = player count, PC3 = modern
   solo co-op card game vs. classic, PC4/5/6 are dice/card sub-splits — the
   poles are anchored by the long tail, which is what makes them honest.

## What the collection module already does

`collection:` has `finalize_through: 2025`; `collection_model.finalize()`
refits the tuned model on train+val+test filtered to
`year_published <= finalize_through` and records `finalize_through` in
`registration.json`. The embeddings module has no equivalent — the pattern was
never carried over.

## Proposed revision (when retraining)

1. Add `embeddings.finalize_through` (default = `years.current - 1`, i.e. 2025
   today) and have the embeddings trainer's final refit use
   train + tune + test filtered to `year_published <= finalize_through`,
   mirroring `collection_model.finalize`. Keep the time-based split for
   evaluation exactly as it is — only the *final* fit changes.
2. Refit the **preprocessor** on the same final population, not the train
   split, so scaling and the dummy vocabulary reflect what the model is
   actually fit on. (Requires the diagnostics that compare tune/test
   embeddings to be computed before the refit, which they already are.)
3. Record `finalize_through`, `min_ratings` and the final sample count in
   `model_info.json` / `registration.json` so the fit population is visible
   from the registry without reading the trainer.
4. Keep `min_ratings: 5`. The floor is a deliberate choice to fit on the full
   universe (see above), and the revision here is in the same direction — see
   *more* of it, not less. If it is ever revisited, evaluate with the existing
   similarity eval (`2026-08-31-embedding-similarity-eval-design.md`).
5. Retrain → new experiment version → register as `embeddings-v2026` v7 via
   the justfile `register` recipe; the embeddings workflow picks it up by name.
   Downstream (`bgg_game_embeddings`, `bgg_game_coordinates`, the viewer's
   coordinates/neighbours artifacts) are version-joined, so a bump is safe as
   long as embeddings and coordinates land together.

## Out of scope

- Changing the algorithm, dimensionality or input feature set.
- Adding text/description embeddings (see the input-scaling design note).
