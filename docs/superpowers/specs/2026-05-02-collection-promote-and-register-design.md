# Collection Promote + Register — design

Make `just promote` (and a new `just promote-all`) atomically deploy a
finalized user collection model: upload to GCS *and* insert a row into the
new `raw.collection_models_registry` table that the scoring service queries.
Closes the gap left by the scoring-service plan
([2026-05-01-collection-scoring-service-design.md](2026-05-01-collection-scoring-service-design.md)),
which built the registry-reader side without a writer.

Scoped to the `own` (classification) outcome for v1; multi-outcome support
falls out naturally because `register_all.py` already loops outcomes.

## Scope

In scope:

- A new `RegistryWriter` that inserts a row into
  `raw.collection_models_registry` and flips any prior `active` row for the
  same `(username, outcome)` to `inactive`.
- Modify `services/collections/register_model.py` to be strict about
  finalized artifacts (no `model.pkl` fallback) and to call
  `RegistryWriter` after a successful GCS upload.
- A `collections.users` flat list and `collections.deploy.{outcome}.candidate`
  map in `config.yaml`, with two new accessors on `Config`.
- A `just promote-all` justfile recipe that loops users from config and
  promotes each one's `own` model, skipping users without a finalized
  artifact.

Out of scope:

- Other outcomes (`ever_owned`, `rated`, `rating`, `love`). Same code path
  works once you add a `deploy.{outcome}` entry; v1 wires only `own`.
- Per-user candidate overrides.
- A scheduled / automated promote-all (it stays manual).
- A rollback CLI. Rollback is a single `UPDATE … SET status='active' …`
  done by hand.

## Architecture

```
just promote-all
  ├─ Python one-liner: read config.yaml
  │    → users: [phenrickson, ...]
  │    → deploy.own.candidate: logistic_row_norm
  │
  ├─ for each user:
  │    ├─ exists models/collections/{env}/{user}/own/{candidate}/v*/finalized.pkl?
  │    │     → no:  log "skip {user}: no finalized artifact"; continue
  │    │     → yes: just promote outcome=own candidate={candidate}
  │    │
  │    └─ uv run -m services.collections.register_model
  │         ├─ verify finalized.pkl exists (else FileNotFoundError)
  │         ├─ load finalized.pkl + threshold.json + registration.json
  │         ├─ RegisteredCollectionModel.register() → GCS, returns v{N}
  │         └─ RegistryWriter.register_deployment(...)
  │              ├─ UPDATE collection_models_registry
  │              │    SET status='inactive'
  │              │    WHERE username=@u AND outcome=@o AND status='active'
  │              └─ INSERT collection_models_registry
  │                   (username, outcome, model_version, finalize_through_year,
  │                    gcs_path, registered_at, status='active')
  │
  └─ summary: deployed=N, skipped=M, failed=K
```

## Components

### 1. `services/collections/registry_writer.py` (new)

```python
class RegistryWriter:
    def __init__(self, table_id: str, client: bigquery.Client | None = None): ...

    def register_deployment(
        self,
        *,
        username: str,
        outcome: str,
        model_version: int,
        gcs_path: str,
        finalize_through_year: int | None,
        registered_at: datetime,
    ) -> None:
        """
        1. UPDATE existing active rows for (username, outcome) → status='inactive'.
        2. INSERT new row as active.
        Both run sequentially. Not wrapped in a BQ transaction; the
        lookup_latest reader uses ORDER BY model_version DESC LIMIT 1, so the
        intermediate state (zero active rows for ~50ms) is still correct.
        """
```

Sibling of `services/collections/registry.py` (the read-side), keeps the
read/write surfaces separate so neither has to grow a multi-mode interface.

### 2. `services/collections/register_model.py` (modified)

Two changes to `register_collection(...)`:

1. **Drop the `model.pkl` fallback.** Today the function accepts either
   `finalized.pkl` or `model.pkl`. Replace the branch so missing
   `finalized.pkl` raises `FileNotFoundError(f"Finalized artifact not
   found: {finalized_path}. Run finalize before promoting.")`. The
   `pipeline_kind` field in `source_metadata` becomes a constant
   `"finalized"`.

2. **Append a registry insert.** After
   `RegisteredCollectionModel.register()` returns, instantiate
   `RegistryWriter` with the registry table id from
   `Config.get_collection_registry_table()` and call
   `register_deployment(...)` with:
   - `model_version=registration["version"]` (the GCS version just written)
   - `gcs_path` constructed as `gs://{bucket}/{env}/services/collections/{username}/{outcome}/v{version}/`
   - `finalize_through_year=cand_reg.get("finalize_through_year")` (from the
     candidate's `registration.json`; can be `None`)
   - `registered_at=datetime.now(timezone.utc)`

Errors from the registry insert propagate to the CLI. If GCS succeeded but
the registry insert failed, the GCS artifact is orphaned but harmless: the
next promote increments to `v{N+1}` (because
`RegisteredCollectionModel.list_versions` reads GCS, not the registry) and
the registry catches up. Logged as a warning when this happens so the
operator notices.

### 3. `config.yaml` (modified)

```yaml
collections:
  users:
    - phenrickson
  deploy:
    own:
      candidate: logistic_row_norm
  outcomes: { ... }      # existing
  candidates: [ ... ]    # existing
  scoring: { ... }       # existing
```

Two new accessors on `Config` (using the `raw_config` escape hatch, same
pattern as `get_collection_registry_table`):

```python
def get_collection_users(self) -> list[str]: ...
def get_collection_deploy_candidate(self, outcome: str) -> str: ...
```

`get_collection_deploy_candidate("own")` returns `"logistic_row_norm"`.
Raises `KeyError` if the outcome is not in `deploy:`.

### 4. `justfile` (modified)

New recipe:

```just
# Promote logistic_row_norm (or whatever's in collections.deploy.own.candidate)
# for every user in collections.users. Skips users without a finalized artifact.
promote-all:
    @python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
        print(c['collections']['users'][0])" >/dev/null  # validates the keys exist
    @users=$(python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
        print('\n'.join(c['collections']['users']))"); \
    cand=$(python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
        print(c['collections']['deploy']['own']['candidate'])"); \
    deployed=0; skipped=0; failed=0; \
    while IFS= read -r u; do \
        [ -z "$u" ] && continue; \
        path="{{local_root}}/{{environment}}/$u/own/$cand"; \
        if ! ls $path/v*/finalized.pkl 2>/dev/null | grep -q .; then \
            echo "skip $u: no finalized.pkl under $path"; \
            skipped=$((skipped + 1)); continue; \
        fi; \
        if just username=$u outcome=own candidate=$cand promote; then \
            deployed=$((deployed + 1)); \
        else \
            echo "FAIL: $u"; failed=$((failed + 1)); \
        fi; \
    done <<< "$users"; \
    echo "promote-all: deployed=$deployed skipped=$skipped failed=$failed"; \
    [ $failed -eq 0 ]
```

Continue-on-error per user; exits non-zero only if any user genuinely
failed. Skipped users don't count as failures.

The recipe stays in the justfile per the design rule "justfiles document,
scripts compute" — it's loop + existence check + accumulator. Becomes a
Python script the moment it needs a third knob (e.g., per-user candidate
overrides, multi-outcome iteration).

## Error handling

- **`finalized.pkl` missing in the local candidate dir** → `FileNotFoundError`
  in `register_model.py`. CLI exits 1 with the path that was looked for.
- **`promote-all` skip case**: the user is in `config.users` but has no
  finalized artifact for the configured candidate. Logged with the path
  searched, recipe continues.
- **GCS upload fails**: existing behavior, exception propagates, no registry
  insert.
- **Registry insert fails after GCS succeeds**: GCS artifact is orphaned but
  harmless. Next promote writes `v{N+1}` and the registry catches up.
  `RegistryWriter` logs a warning before re-raising.
- **`get_collection_deploy_candidate` for an outcome not in
  `deploy:`** → `KeyError`. The recipe only references `deploy.own` for
  v1, so this can only fire if the user manually invokes the accessor with
  an outcome they haven't configured.

## Testing

Unit tests for `RegistryWriter`:

- First-ever deploy: no prior active row. UPDATE runs (zero rows
  affected, fine), INSERT adds the new row. Assert both calls were made
  with the expected query parameters.
- Re-deploy: prior active row exists. UPDATE flips it, INSERT adds new.
  Assert the SQL parameters on both.
- Registry insert fails: assert the warning is logged before re-raise.

Unit test for strict-finalized in `register_collection`:

- `finalized.pkl` present → registers normally, GCS + registry both called.
  Assert `pipeline_kind="finalized"` in the source metadata.
- Only `model.pkl` present → `FileNotFoundError` raised, neither GCS nor
  registry touched. Assert error message includes the expected
  `finalized.pkl` path.

The `promote-all` recipe is bash; verified by running it locally end-to-end
once the implementation is done.
