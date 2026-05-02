# Collection Promote + Register Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `just promote` atomically deploy a finalized collection model — GCS upload *and* a row in `raw.collection_models_registry` — and add a `promote-all` recipe that sweeps every user listed in `config.collections.users`.

**Architecture:** New `RegistryWriter` class writes registry rows (UPDATE prior active rows → inactive, INSERT new active row). `register_model.py` calls it after `RegisteredCollectionModel.register()` succeeds. Strict-finalized: drop the `model.pkl` fallback. New `collections.users` flat list and `collections.deploy.{outcome}.candidate` map in config drive a `promote-all` justfile recipe that loops users, skipping any without a finalized artifact.

**Tech Stack:** Python 3.12, `google-cloud-bigquery`, `pyyaml`, `just`, pytest, existing collection module code.

**Spec:** [docs/superpowers/specs/2026-05-02-collection-promote-and-register-design.md](../specs/2026-05-02-collection-promote-and-register-design.md)

---

## File Structure

**Files created:**
- `services/collections/registry_writer.py` — `RegistryWriter.register_deployment(...)` (UPDATE + INSERT)
- `tests/test_collections_registry_writer.py` — unit tests

**Files modified:**
- `services/collections/register_model.py` — strict-finalized + registry insert
- `tests/test_collections_register_model.py` — new test file (it does not exist yet for this module; add it)
- `config.yaml` — add `collections.users` and `collections.deploy.own.candidate`
- `src/utils/config.py` — add `get_collection_users()` and `get_collection_deploy_candidate(outcome)`
- `justfile` — add `promote-all` recipe

**Files NOT modified (intentionally):**
- `services/collections/registered_model.py` — read/upload side already correct
- `services/collections/register_all.py` — uses `register_collection`, inherits new behavior automatically
- `services/collections/registry.py` — read side; the writer is a sibling

---

## Task 1: `RegistryWriter` class with TDD

**Files:**
- Create: `services/collections/registry_writer.py`
- Create: `tests/test_collections_registry_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collections_registry_writer.py`:

```python
"""Tests for services.collections.registry_writer.RegistryWriter."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from services.collections.registry_writer import RegistryWriter


def _build_writer():
    bq = MagicMock()
    return RegistryWriter("proj.raw.collection_models_registry", bq), bq


def test_register_deployment_updates_then_inserts():
    writer, bq = _build_writer()

    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=1,
        gcs_path="gs://bucket/dev/services/collections/alice/own/v1/",
        finalize_through_year=2025,
        registered_at=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )

    # Two BQ calls: UPDATE then INSERT (in that order)
    assert bq.query.call_count == 2
    update_sql, _ = bq.query.call_args_list[0].args, bq.query.call_args_list[0].kwargs
    insert_sql, _ = bq.query.call_args_list[1].args, bq.query.call_args_list[1].kwargs
    assert "UPDATE" in update_sql[0] and "status = 'inactive'" in update_sql[0]
    assert "INSERT" in insert_sql[0]


def test_register_deployment_threads_correct_parameters():
    writer, bq = _build_writer()

    ts = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=2,
        gcs_path="gs://bucket/dev/services/collections/alice/own/v2/",
        finalize_through_year=2025,
        registered_at=ts,
    )

    # UPDATE params
    upd_call = bq.query.call_args_list[0]
    upd_params = {p.name: p.value for p in upd_call.kwargs["job_config"].query_parameters}
    assert upd_params == {"username": "alice", "outcome": "own"}

    # INSERT params
    ins_call = bq.query.call_args_list[1]
    ins_params = {p.name: p.value for p in ins_call.kwargs["job_config"].query_parameters}
    assert ins_params == {
        "username": "alice",
        "outcome": "own",
        "model_version": 2,
        "finalize_through_year": 2025,
        "gcs_path": "gs://bucket/dev/services/collections/alice/own/v2/",
        "registered_at": ts,
    }


def test_register_deployment_first_ever_row_still_runs_update():
    """No prior active row → UPDATE matches 0 rows, INSERT still runs."""
    writer, bq = _build_writer()

    writer.register_deployment(
        username="bob",
        outcome="own",
        model_version=1,
        gcs_path="gs://bucket/dev/services/collections/bob/own/v1/",
        finalize_through_year=None,  # may be missing on first finalize
        registered_at=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert bq.query.call_count == 2  # UPDATE + INSERT regardless


def test_register_deployment_handles_null_finalize_through_year():
    writer, bq = _build_writer()

    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=1,
        gcs_path="gs://x/v1/",
        finalize_through_year=None,
        registered_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    ins_call = bq.query.call_args_list[1]
    params = {p.name: p.value for p in ins_call.kwargs["job_config"].query_parameters}
    assert params["finalize_through_year"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -m pytest tests/test_collections_registry_writer.py -v`
Expected: ImportError on `services.collections.registry_writer` (module doesn't exist).

- [ ] **Step 3: Implement `services/collections/registry_writer.py`**

```python
"""Write-side BigQuery client for the collection_models_registry table.

Sibling of services.collections.registry (the read side). Inserts a new
active row and flips any prior active row for (username, outcome) to inactive.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from google.cloud import bigquery

logger = logging.getLogger(__name__)


class RegistryWriter:
    """Append a new active deployment row, demoting any prior active row."""

    def __init__(self, table_id: str, client: Optional[bigquery.Client] = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def register_deployment(
        self,
        *,
        username: str,
        outcome: str,
        model_version: int,
        gcs_path: str,
        finalize_through_year: Optional[int],
        registered_at: datetime,
    ) -> None:
        """Flip prior active row(s) to inactive, then insert the new active row.

        Two sequential BigQuery statements (not a transaction). The reader's
        lookup_latest uses ORDER BY model_version DESC LIMIT 1, so the
        intermediate state — zero active rows for ~50ms — is still correct.
        """
        self._deactivate_prior(username, outcome)
        try:
            self._insert_active(
                username=username,
                outcome=outcome,
                model_version=model_version,
                gcs_path=gcs_path,
                finalize_through_year=finalize_through_year,
                registered_at=registered_at,
            )
        except Exception:
            logger.warning(
                "registry insert FAILED after GCS upload — "
                "GCS artifact at %s is orphaned. Retry promote to recover.",
                gcs_path,
            )
            raise

    def _deactivate_prior(self, username: str, outcome: str) -> None:
        sql = f"""
            UPDATE `{self.table_id}`
            SET status = 'inactive'
            WHERE username = @username
              AND outcome = @outcome
              AND status = 'active'
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            ]
        )
        self.client.query(sql, job_config=job_config).result()

    def _insert_active(
        self,
        *,
        username: str,
        outcome: str,
        model_version: int,
        gcs_path: str,
        finalize_through_year: Optional[int],
        registered_at: datetime,
    ) -> None:
        sql = f"""
            INSERT INTO `{self.table_id}`
                (username, outcome, model_version, finalize_through_year,
                 gcs_path, registered_at, status)
            VALUES
                (@username, @outcome, @model_version, @finalize_through_year,
                 @gcs_path, @registered_at, 'active')
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
                bigquery.ScalarQueryParameter("model_version", "INT64", model_version),
                bigquery.ScalarQueryParameter(
                    "finalize_through_year", "INT64", finalize_through_year
                ),
                bigquery.ScalarQueryParameter("gcs_path", "STRING", gcs_path),
                bigquery.ScalarQueryParameter("registered_at", "TIMESTAMP", registered_at),
            ]
        )
        self.client.query(sql, job_config=job_config).result()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -m pytest tests/test_collections_registry_writer.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/collections/registry_writer.py tests/test_collections_registry_writer.py
git commit -m "feat(collections): RegistryWriter for atomic deploy bookkeeping"
```

---

## Task 2: Add `collections.users` and `collections.deploy.own.candidate` to config

**Files:**
- Modify: `config.yaml`
- Modify: `src/utils/config.py`

- [ ] **Step 1: Add `users` and `deploy` blocks to `config.yaml`**

Find the existing `collections:` block (it already has `outcomes:`, `candidates:`, `scoring:`). Add `users:` and `deploy:` as siblings. Be careful with indentation — match the existing 2-space style.

```yaml
collections:
  users:
    - phenrickson
  deploy:
    own:
      candidate: logistic_row_norm
  outcomes:
    # ... existing ...
  candidates:
    # ... existing ...
  scoring:
    # ... existing ...
```

- [ ] **Step 2: Verify YAML parses**

```bash
uv run python -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
print('users:', c['collections']['users'])
print('deploy:', c['collections']['deploy'])
"
```
Expected:
```
users: ['phenrickson']
deploy: {'own': {'candidate': 'logistic_row_norm'}}
```

- [ ] **Step 3: Add accessors to `src/utils/config.py`**

Open `src/utils/config.py`. After the existing `get_collection_landing_table` method (around line 317), add:

```python
def get_collection_users(self) -> list[str]:
    """Flat list of usernames whose collection models should be deployed."""
    return list(self.raw_config["collections"]["users"])

def get_collection_deploy_candidate(self, outcome: str) -> str:
    """Candidate name to deploy for the given outcome (e.g. 'logistic_row_norm')."""
    return self.raw_config["collections"]["deploy"][outcome]["candidate"]
```

- [ ] **Step 4: Smoke-test the accessors**

```bash
uv run python -c "
from src.utils.config import load_config
c = load_config()
print(c.get_collection_users())
print(c.get_collection_deploy_candidate('own'))
"
```
Expected:
```
['phenrickson']
logistic_row_norm
```

- [ ] **Step 5: Commit**

```bash
git add config.yaml src/utils/config.py
git commit -m "feat(config): collections.users + collections.deploy.own.candidate"
```

---

## Task 3: Strict-finalized + registry insert in `register_model.py`

**Files:**
- Modify: `services/collections/register_model.py`
- Create: `tests/test_collections_register_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_collections_register_model.py`:

```python
"""Tests for services.collections.register_model.register_collection."""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_candidate_dir(tmp_path: Path, *, finalized: bool, train_only: bool):
    """Build a candidate dir with optional finalized.pkl / model.pkl."""
    cand_root = (
        tmp_path / "models" / "collections" / "dev" / "alice" / "own" / "lgbm_default"
    )
    v1 = cand_root / "v1"
    v1.mkdir(parents=True)

    if finalized:
        (v1 / "finalized.pkl").write_bytes(pickle.dumps("FINALIZED_PIPELINE"))
    if train_only:
        (v1 / "model.pkl").write_bytes(pickle.dumps("TRAIN_PIPELINE"))

    (v1 / "threshold.json").write_text(json.dumps({"threshold": 0.42}))
    (v1 / "registration.json").write_text(
        json.dumps({"splits_version": 7, "finalize_through": 2025})
    )
    return cand_root.parent.parent.parent.parent.parent  # = tmp_path / "models" ... unwound


def test_register_collection_strict_requires_finalized_pkl(tmp_path):
    """Only model.pkl present (no finalized.pkl) → FileNotFoundError."""
    from services.collections import register_model as rm

    _make_candidate_dir(tmp_path, finalized=False, train_only=True)
    local_root = str(tmp_path / "models" / "collections")

    with pytest.raises(FileNotFoundError, match="Finalized artifact not found"):
        rm.register_collection(
            username="alice",
            outcome="own",
            candidate="lgbm_default",
            description="test",
            version="latest",
            environment="dev",
            local_root=local_root,
        )


def test_register_collection_calls_gcs_then_registry(tmp_path):
    """Happy path: finalized.pkl present → GCS register, then registry insert."""
    from services.collections import register_model as rm

    _make_candidate_dir(tmp_path, finalized=True, train_only=False)
    local_root = str(tmp_path / "models" / "collections")

    with patch.object(rm, "RegisteredCollectionModel") as RCM_cls, \
         patch.object(rm, "RegistryWriter") as RW_cls, \
         patch.object(rm, "load_config") as load_cfg:
        rcm = MagicMock()
        rcm.register.return_value = {"version": 3, "username": "alice", "outcome": "own"}
        RCM_cls.return_value = rcm

        rw = MagicMock()
        RW_cls.return_value = rw

        cfg = MagicMock()
        cfg.get_collection_registry_table.return_value = "p.raw.collection_models_registry"
        cfg.get_bucket_name.return_value = "bgg-predictive-models"
        cfg.get_environment_prefix.return_value = "dev"
        load_cfg.return_value = cfg

        rm.register_collection(
            username="alice",
            outcome="own",
            candidate="lgbm_default",
            description="test",
            version="latest",
            environment="dev",
            local_root=local_root,
        )

        # GCS upload happened first
        rcm.register.assert_called_once()
        kwargs = rcm.register.call_args.kwargs
        assert kwargs["pipeline"] == "FINALIZED_PIPELINE"
        assert kwargs["threshold"] == 0.42
        assert kwargs["source_metadata"]["pipeline_kind"] == "finalized"

        # Then registry insert with the version from GCS
        rw.register_deployment.assert_called_once()
        rw_kwargs = rw.register_deployment.call_args.kwargs
        assert rw_kwargs["username"] == "alice"
        assert rw_kwargs["outcome"] == "own"
        assert rw_kwargs["model_version"] == 3
        assert rw_kwargs["finalize_through_year"] == 2025
        assert rw_kwargs["gcs_path"] == (
            "gs://bgg-predictive-models/dev/services/collections/alice/own/v3/"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -m pytest tests/test_collections_register_model.py -v`

Expected — both tests fail:
- `test_register_collection_strict_requires_finalized_pkl`: currently the function falls back to `model.pkl`, so no error is raised.
- `test_register_collection_calls_gcs_then_registry`: `RegistryWriter` is not yet imported into the module, and `register_collection` doesn't accept the load_config branch.

- [ ] **Step 3: Read the current `register_model.py`**

Run: `cat services/collections/register_model.py`

Locate the block that picks `pipeline_path` (around lines 47–54): currently it tries `finalized.pkl`, then falls back to `model.pkl`. We will replace that with a strict check, and add the registry-insert call after the existing `registry.register(...)` call (around lines 68–80).

- [ ] **Step 4: Modify `services/collections/register_model.py`**

Make these three changes to the file:

(a) Add the imports near the top (after the existing `from services.collections.registered_model import ...`):

```python
from datetime import datetime, timezone

from services.collections.registry_writer import RegistryWriter
from src.utils.config import load_config
```

(b) Replace the finalized/train-only branch. Find the lines:

```python
    finalized = version_dir / "finalized.pkl"
    train_only = version_dir / "model.pkl"
    if finalized.exists():
        pipeline_path, kind = finalized, "finalized"
    elif train_only.exists():
        pipeline_path, kind = train_only, "train_only"
    else:
        raise FileNotFoundError(f"No finalized.pkl or model.pkl in {version_dir}")
```

Replace with:

```python
    finalized = version_dir / "finalized.pkl"
    if not finalized.exists():
        raise FileNotFoundError(
            f"Finalized artifact not found: {finalized}. "
            f"Run finalize before promoting."
        )
    pipeline_path, kind = finalized, "finalized"
```

(c) After the existing `registration = registry.register(...)` call and the `logger.info(...)` line below it, insert the registry-writer call. The block around it changes from:

```python
    registration = registry.register(
        pipeline=pipeline,
        threshold=threshold,
        source_metadata={...},
        description=description,
    )
    logger.info(
        "Registered %s/%s as v%d (kind=%s, source v%d)",
        username, outcome, registration["version"], kind, resolved,
    )
    return registration
```

…to:

```python
    registration = registry.register(
        pipeline=pipeline,
        threshold=threshold,
        source_metadata={
            "candidate": candidate,
            "candidate_version": resolved,
            "pipeline_kind": kind,
            "splits_version": cand_reg.get("splits_version"),
            "candidate_registration": cand_reg,
        },
        description=description,
    )
    logger.info(
        "Registered %s/%s as v%d (kind=%s, source v%d)",
        username, outcome, registration["version"], kind, resolved,
    )

    # Insert into the BQ registry so the scoring service can find this deployment.
    cfg = load_config()
    registry_table = cfg.get_collection_registry_table()
    bucket = cfg.get_bucket_name()
    env = cfg.get_environment_prefix()
    gcs_path = (
        f"gs://{bucket}/{env}/services/collections/"
        f"{username}/{outcome}/v{registration['version']}/"
    )
    RegistryWriter(registry_table).register_deployment(
        username=username,
        outcome=outcome,
        model_version=registration["version"],
        gcs_path=gcs_path,
        # The finalize step writes "finalize_through" to candidate
        # registration.json (see src/collection/collection_artifact_storage.py
        # save_finalized_pipeline). Map it to the registry's
        # finalize_through_year column.
        finalize_through_year=cand_reg.get("finalize_through"),
        registered_at=datetime.now(timezone.utc),
    )

    return registration
```

Note: leave the existing `source_metadata` dict intact (it already has the right keys); the listing above just shows it in context.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run -m pytest tests/test_collections_register_model.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Confirm all collection-service tests still pass**

Run: `uv run -m pytest tests/test_collections_*.py tests/test_collection_*.py -v 2>&1 | tail -25`
Expected: all collection-related tests pass (counts depend on what's in the repo, but no NEW failures vs. the prior commit).

- [ ] **Step 7: Commit**

```bash
git add services/collections/register_model.py tests/test_collections_register_model.py
git commit -m "feat(collections): strict-finalized promote + registry insert"
```

---

## Task 4: `promote-all` justfile recipe

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Read the existing recipes**

Run: `grep -n "^promote\|^promote-many\|^username\|^environment\|^local_root" justfile`

Locate the existing `promote` and `promote-many` recipes (they reference `{{username}}`, `{{environment}}`, `{{local_root}}`). The new recipe sits after them.

- [ ] **Step 2: Add `promote-all` to `justfile`**

After the `promote-many` recipe, append:

```just
# Promote the configured candidate (collections.deploy.{outcome}.candidate)
# for every user in collections.users. Skips users without a finalized
# artifact for that candidate. Continue-on-error; exits non-zero if any
# user genuinely failed.
#
# Example: just promote-all
#          just promote-all outcome=own
promote-all outcome="own":
    @users=$(uv run python -c "import yaml; \
        c = yaml.safe_load(open('config.yaml')); \
        print('\n'.join(c['collections']['users']))"); \
    cand=$(uv run python -c "import yaml; \
        c = yaml.safe_load(open('config.yaml')); \
        print(c['collections']['deploy']['{{outcome}}']['candidate'])"); \
    deployed=0; skipped=0; failed=0; \
    while IFS= read -r u; do \
        [ -z "$u" ] && continue; \
        path="{{local_root}}/{{environment}}/$u/{{outcome}}/$cand"; \
        if ! ls $path/v*/finalized.pkl 2>/dev/null | grep -q .; then \
            echo "skip $u: no finalized.pkl under $path"; \
            skipped=$((skipped + 1)); \
            continue; \
        fi; \
        echo "=== promote $u {{outcome}} $cand ==="; \
        if just username=$u outcome={{outcome}} candidate=$cand promote; then \
            deployed=$((deployed + 1)); \
        else \
            echo "FAIL: $u"; \
            failed=$((failed + 1)); \
        fi; \
    done <<< "$users"; \
    echo "promote-all: deployed=$deployed skipped=$skipped failed=$failed"; \
    [ $failed -eq 0 ]
```

- [ ] **Step 3: Verify the recipe is parseable and listed**

Run: `just --list 2>&1 | grep promote-all`
Expected: line includes `promote-all` with its outcome arg.

- [ ] **Step 4: Smoke-test the skip path**

Pick a username that you know has no finalized artifact locally (e.g. `nonexistent_user_xyz`). Temporarily edit `config.yaml` to add it to `collections.users`:

```yaml
collections:
  users:
    - phenrickson
    - nonexistent_user_xyz
```

Run: `just promote-all`
Expected: `phenrickson` either deploys or skips depending on local artifacts; `nonexistent_user_xyz` logs `skip nonexistent_user_xyz: no finalized.pkl under ...` and is counted as skipped. Final summary line appears.

Then revert `config.yaml` (remove the test user). Do NOT commit the test user.

- [ ] **Step 5: Commit**

```bash
git add justfile
git commit -m "feat(collections): promote-all recipe driven by collections.users"
```

---

## Task 5: Full sweep

- [ ] **Step 1: Run all collection tests**

Run: `uv run -m pytest tests/test_collections_*.py tests/test_collection_*.py -v 2>&1 | tail -30`
Expected: all pass. New additions:
- `test_collections_registry_writer.py` — 4 tests
- `test_collections_register_model.py` — 2 tests

- [ ] **Step 2: Lint**

Run: `uv run ruff check services/collections/ tests/test_collections_*.py 2>&1 | tail -10`
Expected: `All checks passed!`

- [ ] **Step 3: Smoke-test the full module graph**

```bash
uv run python -c "
from services.collections.registry_writer import RegistryWriter
from services.collections.register_model import register_collection
from src.utils.config import load_config
cfg = load_config()
print('users:', cfg.get_collection_users())
print('deploy.own:', cfg.get_collection_deploy_candidate('own'))
print('registry table:', cfg.get_collection_registry_table())
print('OK')
"
```
Expected:
```
users: ['phenrickson']
deploy.own: logistic_row_norm
registry table: bgg-predictive-models.raw.collection_models_registry
OK
```

- [ ] **Step 4: Final commit (if any fixes were needed)**

If steps 1–3 surfaced issues, fix and commit:

```bash
git add -A
git commit -m "fix(collections): resolve sweep issues"
```

---
