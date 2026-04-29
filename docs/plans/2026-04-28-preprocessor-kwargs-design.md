# Design: `preprocessor_kwargs` pass-through on Collection model configs

## Goal

Allow candidate YAML to override any of the 25+ knobs in
`create_preprocessing_pipeline`'s `default_config` without promoting them to
first-class fields on `ClassificationModelConfig` / `RegressionModelConfig`.

## Approach

Open dict pass-through. One new field per config dataclass; the dict is splatted
verbatim into `create_preprocessing_pipeline` at the two existing call sites in
`build_pipeline()`.

This was chosen over two alternatives:

- **Promote frequently-changed knobs to first-class fields.** Requires
  predicting which knobs matter; still needs an escape hatch for the rest.
- **Mirror `create_preprocessing_pipeline` parameters in a `PreprocessorConfig`
  dataclass.** Typed and validated, but introduces a 25-field dataclass that
  must stay in sync with the underlying function.

The pass-through approach is the smallest change that delivers full
expressivity, and is easy to upgrade to either alternative later if the dict
gets unwieldy.

## Validation

No explicit validation layer. Typos fail loudly at pipeline-construction time:
`create_preprocessing_pipeline` forwards `**kwargs` to `create_bgg_preprocessor`
at `src/models/training.py:289-292`, which forwards them to
`BaseBGGTransformer(**kwargs)` at `src/features/preprocessor.py:71`.
`BaseBGGTransformer.__init__` is explicitly typed with no `**kwargs` catch-all
(`src/features/transformers.py:498-537`), so an unknown key raises
`TypeError: __init__() got an unexpected keyword argument '<bad_key>'` before
any training runs.

A `KNOWN_PREPROCESSOR_KEYS` allowlist would duplicate the source of truth and
require cross-module sync; rejected on those grounds.

## Changes

Four lines plus an import.

1. `src/collection/collection_model.py:8` — extend the `dataclasses` import:

   ```python
   from dataclasses import dataclass, field
   ```

2. `src/collection/collection_model.py:45-52` — add field to
   `ClassificationModelConfig`:

   ```python
   preprocessor_kwargs: Dict[str, Any] = field(default_factory=dict)
   ```

3. `src/collection/collection_model.py:55-60` — same field on
   `RegressionModelConfig`.

4. `src/collection/collection_model.py:144-146` — splat into the classification
   call:

   ```python
   preprocessor = create_preprocessing_pipeline(
       model_type=cfg.preprocessor_type,
       model_name=cfg.model_type,
       **cfg.preprocessor_kwargs,
   )
   ```

5. `src/collection/collection_model.py:157-159` — same splat on the regression
   call.

## What works automatically

- **YAML rehydration.** `CollectionCandidate.from_dict` at
  `src/collection/candidates.py:147-154` splats the YAML dict into the
  dataclass constructor, so `preprocessor_kwargs: {...}` in YAML is picked up
  without further code changes.
- **Round-trip through `registration.json`.** `to_dict` is `asdict()`, which
  serializes the new dict field as-is.
- **All training paths.** `_tune_classification`, `_tune_classification_cv`,
  `_tune_regression`, `_tune_regression_cv`, and `train` all build their
  pipeline through `build_pipeline()` — the two splats above cover every code
  path that constructs a preprocessor.

## YAML shape

```yaml
collections:
  candidates:
    - name: lgbm_strict
      classification_config:
        model_type: lightgbm
        preprocessor_kwargs:
          designer_min_freq: 25
          max_publisher_features: 100
```

## Out of scope

- No new tests. `tests/test_pipeline.py` already exercises
  `create_preprocessing_pipeline`'s `**kwargs` handling; the splat introduces
  no new behavior.
- No documentation updates. The `preprocessor_kwargs` field is
  self-documenting via the `default_config` dict at
  `src/models/training.py:256-286`.
