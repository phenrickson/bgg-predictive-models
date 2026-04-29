# Collection modules — design notes

Decisions about the `src/collection/` modules, captured so future-me (and
anyone else reading the code) understands the reasoning, not just the
result. Decisions evolve; when a decision is superseded, update its
section rather than appending a new one.

## Candidates live in `config.yaml`, not per-file YAMLs

A single `collections.candidates:` list inside `config.yaml`, parallel to
the existing `collections.outcomes`. Loaded via `load_candidates(config)`
in [src/collection/candidates.py](../../src/collection/candidates.py).

**Why:** for a solo project, branching on `config.yaml` is the most
natural way to track experiments — `git diff main..my-branch` shows
exactly what's different. Per-file YAMLs paid a cost (separate directory,
new loader, extra indirection) for benefits that mostly matter on multi-
person teams (parallel iteration on different candidate sets without
conflicts).

**Revisit if:** `config.yaml` grows past ~500 lines, or two people start
needing to iterate on candidates on different branches and hit merge
conflicts repeatedly.

## Train and save are separate functions

`train_candidate(candidate, outcome, train_df, val_df, test_df, …)` does
pure compute and returns a `CandidateRunResult` carrying the fitted
pipeline + metrics in memory. `save_candidate_run(result, storage)` is a
separate function that writes to disk.

**Why:** the original design did training + persistence atomically.
That meant inspecting predictions in a notebook required a disk
round-trip (`storage.load_candidate_run(...)`) for a model you just
trained. Splitting them makes the notebook flow natural — train, look
at metrics, decide whether to save — and lets the CLI compose the two
operations explicitly.

Each function does one thing, returns its output, doesn't reach out to
side effects. Higher-level code chains them:

```
train_candidates → save_candidate_run → load_candidate_runs → compare_runs → save_model
       ↑                  ↑                    ↑                  ↑              ↑
   pure compute       persistence            read              pure         persistence
```

## Feature loading belongs on `BGGDataLoader`, not `CollectionProcessor`

`CollectionProcessor.load_features()` was the canonical universe loader
— it built the SQL with optional joins for `predicted_complexity` and
embeddings. But "load the universe of games" has nothing to do with
processing a user's collection; the method only ended up there because
the SQL was convenient to share.

The notebook reached for `BGGDataLoader.load_training_data` for the
universe (more discoverable, lives on `BGGDataLoader` where you'd
expect a "load games" function to live), which didn't have the join
flags. Result: train data with positives that had `predicted_complexity`
and negatives that didn't — a real schema-mismatch leak.

**Fix:** moved `load_features` to `BGGDataLoader`. `CollectionProcessor`
still has an internal loader instance and uses it inside `process()`,
but external callers (pipeline, CLI, notebook) construct `BGGDataLoader`
directly with their flags. One canonical universe loader, one place to
look.

## `viz.py` has one function per plot, with `interactive=True`

Two consumers — notebooks (static, for Quarto rendering) and a future
Dash app (interactive). Considered three approaches:

- **A.** Two modules (`viz.py` static + `viz_interactive.py` plotly).
  Clean separation but every plot has to be implemented twice.
- **B.** One module with `interactive: bool = False`. Shared data prep,
  branched render. Hidden return type but minimal duplication.
- **C.** Functions return tidy DataFrames + thin renderer modules. Best
  scaling, overkill for two plots.

Picked **B**. Each function does ~30 lines of data prep (filter, top-N,
sort) shared across renderers, with ~10 lines of plotnine and ~25 lines
of plotly per render. Duplicating the data prep would be the bigger
smell.

Default is plotnine (static, ggplot grammar — matches the R references
the user works from). `interactive=True` returns a plotly figure for
Dash. Caller knows which mode it asked for.

## `users-sweep` is a justfile recipe, not a Python script

Incremental processing across many users is a loop with a
skip-already-done check.

- Python script — overkill for 15 lines of bash
- justfile recipe — fits while it stays small
- Both — confusing duplication

Picked the justfile. The rule of thumb: **justfiles document, scripts
compute.** As long as `users-sweep` is "loop + existence check +
accumulator," bash is fine. The moment it grows a third knob beyond
`users` and `outcome`, it should become a Python script that the justfile
calls.

The skip check uses presence of `{outcome}/{candidate}/v*` directories
under the user's path — a finished candidate run is the signal. Stale
splits don't count as "done", which matches the failure-recovery
behavior we want.

## `_next_version` is public

Renamed `CollectionArtifactStorage._next_version` → `next_version`. It's
called by `CollectionPipeline` (always was) and by `promote_candidate`
(new). Sprinkling `# noqa: SLF001` across external callers is a smell;
making the method public is the honest fix.

## Default downsample seed is 1999, not 42

`downsample_negatives` previously defaulted to `random_seed=42`, but the
notebook used `random_seed=1999` explicitly. Changing the library default
to 1999 means notebook and candidate-runner produce identical downsampled
rows from the same input, no explicit seed required.

If you ever pass a seed explicitly elsewhere in the codebase, use 1999
to match.
