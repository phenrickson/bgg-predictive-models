# Collection Scoring Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Cloud Run service that hosts deployed per-user collection models, expose `/predict_own` for daily scoring of unscored games, append predictions to a new BigQuery landing table, and drive the daily run from a GitHub Action that reads from a new BQ registry table.

**Architecture:** New `services/collections/` FastAPI app on Cloud Run, sibling of `services/scoring/`. Loads pipelines via the existing `RegisteredCollectionModel` from `gs://.../{env}/services/collections/{username}/own/v{N}/`. Server-side change detection joins against an append-only landing table; rows accumulate across model versions. A BQ registry (`raw.collection_models_registry`) is the source of truth for active deployments and the loop variable for the daily GitHub Action.

**Tech Stack:** Python 3.12, FastAPI, polars/pandas, scikit-learn, google-cloud-bigquery, google-cloud-storage, Cloud Run, Cloud Build, GitHub Actions.

**Spec:** [docs/superpowers/specs/2026-05-01-collection-scoring-service-design.md](../specs/2026-05-01-collection-scoring-service-design.md)

---

## File Structure

**Files created:**
- `services/collections/main.py` — FastAPI app (`/health`, `/predict_own`, `/models`, `/model/{username}/{outcome}/info`)
- `services/collections/auth.py` — thin re-export of `services.scoring.auth` (or local copy)
- `services/collections/change_detection.py` — SQL builder for "unscored game_ids" lookup
- `services/collections/registry.py` — BQ registry client (read active rows, lookup latest by user/outcome)
- `services/collections/landing_uploader.py` — append-only uploader for `collection_predictions_landing`
- `services/collections/cloudbuild.yaml` — Cloud Build deploy config
- `docker/collections.Dockerfile` — Cloud Run image
- `terraform/collection_models_registry.tf` — registry table schema
- `terraform/collection_predictions_landing.tf` — landing table schema (partitioned, clustered)
- `tests/test_collections_service_change_detection.py` — unit tests for SQL builder
- `tests/test_collections_service_registry.py` — unit tests for registry client
- `tests/test_collections_service_endpoint.py` — endpoint integration test (mocked GCS + BQ)
- `.github/workflows/run-collection-scoring.yml` — daily action

**Files modified:**
- `services/collections/__init__.py` — no functional change; verify importability
- `config.yaml` — add `collections.scoring` section (registry table id, landing table id)
- `src/utils/config.py` — add accessors for the two new table ids
- `Makefile` — add `start-collections-scoring`, `stop-collections-scoring`, `collections-scoring-service` targets

**Files NOT modified (intentionally):**
- `services/scoring/main.py` — game scoring service untouched
- `services/collections/registered_model.py` — already exists and is correct
- `services/collections/register_model.py` — that's the training-side workflow, separate plan

---

## Task 1: Create BQ landing table via Terraform

**Files:**
- Create: `terraform/collection_predictions_landing.tf`

- [ ] **Step 1: Read the existing landing-table terraform to copy the pattern**

Run: `ls terraform/`

Then: `grep -rn "ml_predictions_landing" terraform/ | head -5`

Open the file that defines `ml_predictions_landing` and read it. The new table follows the same pattern (partitioned by `score_ts`, clustered, etc.).

- [ ] **Step 2: Write `terraform/collection_predictions_landing.tf`**

```hcl
resource "google_bigquery_table" "collection_predictions_landing" {
  dataset_id = "raw"
  table_id   = "collection_predictions_landing"
  project    = "bgg-predictive-models"

  description = "Append-only landing table for per-user collection predictions. Rows accumulate across model versions; downstream Dataform deduplicates."

  time_partitioning {
    type  = "DAY"
    field = "score_ts"
  }

  clustering = ["username", "game_id"]

  schema = jsonencode([
    { name = "job_id",          type = "STRING",    mode = "REQUIRED" },
    { name = "username",        type = "STRING",    mode = "REQUIRED" },
    { name = "game_id",         type = "INT64",     mode = "REQUIRED" },
    { name = "outcome",         type = "STRING",    mode = "REQUIRED" },
    { name = "predicted_prob",  type = "FLOAT64",   mode = "REQUIRED" },
    { name = "predicted_label", type = "BOOL",      mode = "REQUIRED" },
    { name = "threshold",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "model_name",      type = "STRING",    mode = "REQUIRED" },
    { name = "model_version",   type = "INT64",     mode = "REQUIRED" },
    { name = "score_ts",        type = "TIMESTAMP", mode = "REQUIRED" }
  ])

  deletion_protection = true
}
```

- [ ] **Step 3: Validate the file parses**

Run: `cd terraform && terraform fmt -check collection_predictions_landing.tf && terraform validate`
Expected: no formatting diffs and `Success! The configuration is valid.` (terraform deployments happen via GitHub Action, do NOT run `terraform apply` locally).

- [ ] **Step 4: Commit**

```bash
git add terraform/collection_predictions_landing.tf
git commit -m "feat(terraform): add collection_predictions_landing BQ table"
```

---

## Task 2: Create BQ registry table via Terraform

**Files:**
- Create: `terraform/collection_models_registry.tf`

- [ ] **Step 1: Write `terraform/collection_models_registry.tf`**

```hcl
resource "google_bigquery_table" "collection_models_registry" {
  dataset_id = "raw"
  table_id   = "collection_models_registry"
  project    = "bgg-predictive-models"

  description = "Registry of deployed per-user collection models. Active rows drive the daily scoring job."

  schema = jsonencode([
    { name = "username",              type = "STRING",    mode = "REQUIRED" },
    { name = "outcome",               type = "STRING",    mode = "REQUIRED" },
    { name = "model_version",         type = "INT64",     mode = "REQUIRED" },
    { name = "finalize_through_year", type = "INT64",     mode = "NULLABLE" },
    { name = "gcs_path",              type = "STRING",    mode = "REQUIRED" },
    { name = "registered_at",         type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "status",                type = "STRING",    mode = "REQUIRED" }
  ])

  deletion_protection = true
}
```

- [ ] **Step 2: Validate the file parses**

Run: `cd terraform && terraform fmt -check collection_models_registry.tf && terraform validate`
Expected: no formatting diffs and `Success! The configuration is valid.` (terraform deployments happen via GitHub Action, do NOT run `terraform apply` locally).

- [ ] **Step 3: Commit**

```bash
git add terraform/collection_models_registry.tf
git commit -m "feat(terraform): add collection_models_registry BQ table"
```

---

## Task 3: Add config accessors for the two new tables

**Files:**
- Modify: `config.yaml`
- Modify: `src/utils/config.py`

- [ ] **Step 1: Locate the existing `scoring` block in `config.yaml`**

Run: `grep -n "^scoring:\|^collections:" config.yaml`

Identify where to add a parallel `collections.scoring` block. It should sit alongside the existing `collections.outcomes` and `collections.candidates` if those exist, or as a new top-level block.

- [ ] **Step 2: Add the `collections.scoring` block**

Append (or merge into the existing `collections:` section):

```yaml
collections:
  scoring:
    registry_table: bgg-predictive-models.raw.collection_models_registry
    landing_table: bgg-predictive-models.raw.collection_predictions_landing
```

- [ ] **Step 3: Verify YAML parses and keys load**

Run:
```bash
uv run python -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
print(c['collections']['scoring'])
"
```
Expected: `{'registry_table': 'bgg-predictive-models.raw.collection_models_registry', 'landing_table': 'bgg-predictive-models.raw.collection_predictions_landing'}`

- [ ] **Step 4: Add accessor methods to `src/utils/config.py`**

Open `src/utils/config.py` and find the existing `Config` (or similar) class. Add two methods:

```python
def get_collection_registry_table(self) -> str:
    return self._raw["collections"]["scoring"]["registry_table"]

def get_collection_landing_table(self) -> str:
    return self._raw["collections"]["scoring"]["landing_table"]
```

If the class uses Pydantic models rather than `_raw` dict access, follow that pattern instead — add the fields to the relevant model and expose via `.collections.scoring.registry_table` etc. Either pattern works as long as it matches what's already there.

- [ ] **Step 5: Smoke-test the accessors**

Run:
```bash
uv run python -c "
from src.utils.config import load_config
c = load_config()
print(c.get_collection_registry_table())
print(c.get_collection_landing_table())
"
```
Expected: prints the two fully-qualified table ids.

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/utils/config.py
git commit -m "feat(config): collections.scoring registry+landing table ids"
```

---

## Task 4: Registry client (read active rows, lookup latest)

**Files:**
- Create: `services/collections/registry.py`
- Create: `tests/test_collections_service_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collections_service_registry.py`:

```python
"""Tests for services.collections.registry.CollectionRegistry."""

from unittest.mock import MagicMock
import pytest

from services.collections.registry import CollectionRegistry, RegistryEntry


def _row(username, outcome, version, gcs_path, status="active", year=2025):
    r = MagicMock()
    r.username = username
    r.outcome = outcome
    r.model_version = version
    r.finalize_through_year = year
    r.gcs_path = gcs_path
    r.status = status
    return r


def test_lookup_latest_returns_highest_active_version():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = [
        _row("alice", "own", 1, "gs://b/v1"),
        _row("alice", "own", 2, "gs://b/v2"),
    ]
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    entry = reg.lookup_latest("alice", "own")

    assert entry.username == "alice"
    assert entry.outcome == "own"
    assert entry.model_version == 2
    assert entry.gcs_path == "gs://b/v2"
    assert entry.status == "active"


def test_lookup_latest_returns_none_when_no_active_rows():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = []
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    assert reg.lookup_latest("ghost", "own") is None


def test_list_active_returns_one_entry_per_user_outcome():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = [
        _row("alice", "own", 2, "gs://b/alice/v2"),
        _row("bob",   "own", 1, "gs://b/bob/v1"),
    ]
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    entries = reg.list_active(outcome="own")

    assert len(entries) == 2
    assert {e.username for e in entries} == {"alice", "bob"}
    assert all(e.status == "active" for e in entries)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -m pytest tests/test_collections_service_registry.py -v`
Expected: ImportError — `services.collections.registry` doesn't exist yet.

- [ ] **Step 3: Implement `services/collections/registry.py`**

```python
"""BigQuery client for the collection_models_registry table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from google.cloud import bigquery


@dataclass(frozen=True)
class RegistryEntry:
    username: str
    outcome: str
    model_version: int
    finalize_through_year: Optional[int]
    gcs_path: str
    status: str


class CollectionRegistry:
    """Read access to the collection_models_registry table."""

    def __init__(self, table_id: str, client: Optional[bigquery.Client] = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def lookup_latest(self, username: str, outcome: str) -> Optional[RegistryEntry]:
        """Return the highest-version active row for (username, outcome), or None."""
        sql = f"""
            SELECT username, outcome, model_version, finalize_through_year,
                   gcs_path, status
            FROM `{self.table_id}`
            WHERE username = @username
              AND outcome = @outcome
              AND status = 'active'
            ORDER BY model_version DESC
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            ]
        )
        rows = list(self.client.query(sql, job_config=job_config).result())
        if not rows:
            return None
        r = rows[0]
        return RegistryEntry(
            username=r.username,
            outcome=r.outcome,
            model_version=r.model_version,
            finalize_through_year=r.finalize_through_year,
            gcs_path=r.gcs_path,
            status=r.status,
        )

    def list_active(self, outcome: Optional[str] = None) -> List[RegistryEntry]:
        """Return all active rows, optionally filtered by outcome."""
        where_outcome = "AND outcome = @outcome" if outcome else ""
        sql = f"""
            SELECT username, outcome, model_version, finalize_through_year,
                   gcs_path, status
            FROM `{self.table_id}`
            WHERE status = 'active'
              {where_outcome}
        """
        params = []
        if outcome:
            params.append(bigquery.ScalarQueryParameter("outcome", "STRING", outcome))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = self.client.query(sql, job_config=job_config).result()
        return [
            RegistryEntry(
                username=r.username,
                outcome=r.outcome,
                model_version=r.model_version,
                finalize_through_year=r.finalize_through_year,
                gcs_path=r.gcs_path,
                status=r.status,
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -m pytest tests/test_collections_service_registry.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/collections/registry.py tests/test_collections_service_registry.py
git commit -m "feat(collections): registry client for active deployed models"
```

---

## Task 5: Change-detection SQL builder

**Files:**
- Create: `services/collections/change_detection.py`
- Create: `tests/test_collections_service_change_detection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collections_service_change_detection.py`:

```python
"""Tests for services.collections.change_detection."""

from unittest.mock import MagicMock

from services.collections.change_detection import build_unscored_query, find_unscored


def test_build_unscored_query_uses_left_anti_join_against_landing():
    sql = build_unscored_query(
        landing_table="proj.raw.collection_predictions_landing",
        candidate_table="proj.analytics.games_features",
    )
    # Must filter by username, outcome, model_version on the landing side
    assert "username = @username" in sql
    assert "outcome = @outcome" in sql
    assert "model_version = @model_version" in sql
    # Must reference both tables
    assert "proj.raw.collection_predictions_landing" in sql
    assert "proj.analytics.games_features" in sql


def test_find_unscored_returns_game_ids_only_for_missing_rows():
    bq_client = MagicMock()
    row1 = MagicMock(); row1.game_id = 7
    row2 = MagicMock(); row2.game_id = 42
    bq_client.query.return_value.result.return_value = [row1, row2]

    unscored = find_unscored(
        username="alice",
        outcome="own",
        model_version=3,
        landing_table="proj.raw.collection_predictions_landing",
        candidate_table="proj.analytics.games_features",
        bq_client=bq_client,
    )

    assert unscored == [7, 42]
    # Verify the parameters were threaded through
    call = bq_client.query.call_args
    job_config = call.kwargs["job_config"]
    params = {p.name: p.value for p in job_config.query_parameters}
    assert params == {"username": "alice", "outcome": "own", "model_version": 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -m pytest tests/test_collections_service_change_detection.py -v`
Expected: ImportError on `services.collections.change_detection`.

- [ ] **Step 3: Implement `services/collections/change_detection.py`**

```python
"""SQL helpers for finding game_ids not yet scored under a given user/version."""

from __future__ import annotations

from typing import List, Optional

from google.cloud import bigquery


def build_unscored_query(landing_table: str, candidate_table: str) -> str:
    """Build the LEFT ANTI JOIN SQL.

    Returns game_ids present in the candidate table that have no row in the
    landing table for the given (username, outcome, model_version).
    """
    return f"""
        SELECT gf.game_id
        FROM `{candidate_table}` gf
        LEFT JOIN `{landing_table}` lp
          ON lp.game_id = gf.game_id
         AND lp.username = @username
         AND lp.outcome = @outcome
         AND lp.model_version = @model_version
        WHERE lp.game_id IS NULL
    """


def find_unscored(
    username: str,
    outcome: str,
    model_version: int,
    landing_table: str,
    candidate_table: str,
    bq_client: Optional[bigquery.Client] = None,
    limit: Optional[int] = None,
) -> List[int]:
    """Return the list of game_ids not yet scored for this user/version."""
    client = bq_client or bigquery.Client()
    sql = build_unscored_query(landing_table, candidate_table)
    if limit is not None:
        sql = sql + f"\n        LIMIT {int(limit)}"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("username", "STRING", username),
            bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            bigquery.ScalarQueryParameter("model_version", "INT64", model_version),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [r.game_id for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -m pytest tests/test_collections_service_change_detection.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/collections/change_detection.py tests/test_collections_service_change_detection.py
git commit -m "feat(collections): change-detection SQL builder"
```

---

## Task 6: Landing-table uploader

**Files:**
- Create: `services/collections/landing_uploader.py`

- [ ] **Step 1: Read the existing uploader pattern**

Run: `cat src/data/bigquery_uploader.py | head -80`

Note the pattern: a class wrapping a `bigquery.Client`, with an `insert_rows_json` (or `load_table_from_dataframe`) call against a target table id, plus a schema constant. We'll mirror this for the collection landing table — simpler schema, no Dataform-specific extras.

- [ ] **Step 2: Implement `services/collections/landing_uploader.py`**

```python
"""Append-only uploader for collection_predictions_landing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from google.cloud import bigquery
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CollectionPredictionRow(BaseModel):
    job_id: str
    username: str
    game_id: int
    outcome: str
    predicted_prob: float
    predicted_label: bool
    threshold: float | None
    model_name: str
    model_version: int
    score_ts: datetime


class CollectionPredictionsUploader:
    """Append rows to raw.collection_predictions_landing."""

    def __init__(self, table_id: str, client: bigquery.Client | None = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def upload(self, rows: List[CollectionPredictionRow]) -> int:
        """Insert rows, return count inserted. Raises on any insertion error."""
        if not rows:
            return 0
        payload = [
            {
                "job_id": r.job_id,
                "username": r.username,
                "game_id": r.game_id,
                "outcome": r.outcome,
                "predicted_prob": r.predicted_prob,
                "predicted_label": r.predicted_label,
                "threshold": r.threshold,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "score_ts": r.score_ts.astimezone(timezone.utc).isoformat(),
            }
            for r in rows
        ]
        errors = self.client.insert_rows_json(self.table_id, payload)
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")
        logger.info(f"Inserted {len(payload)} rows into {self.table_id}")
        return len(payload)
```

- [ ] **Step 3: Smoke-test the import**

Run:
```bash
uv run python -c "
from services.collections.landing_uploader import (
    CollectionPredictionsUploader, CollectionPredictionRow,
)
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add services/collections/landing_uploader.py
git commit -m "feat(collections): append-only uploader for landing table"
```

---

## Task 7: Auth re-export

**Files:**
- Create: `services/collections/auth.py`

The collections service uses the same GCP auth as the scoring service. Re-export rather than duplicate.

- [ ] **Step 1: Create `services/collections/auth.py`**

```python
"""Re-export of services.scoring.auth so the collections service can share the
exact same authentication code without duplicating it.

Importing from this module instead of services.scoring.auth keeps the
collections service self-contained at the import-statement level.
"""

from services.scoring.auth import (  # noqa: F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
)
```

- [ ] **Step 2: Smoke-test**

Run: `uv run python -c "from services.collections.auth import GCPAuthenticator; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add services/collections/auth.py
git commit -m "feat(collections): re-export scoring auth for service use"
```

---

## Task 8: FastAPI app skeleton (`/health`, `/models`, `/model/.../info`)

**Files:**
- Create: `services/collections/main.py`

- [ ] **Step 1: Create `services/collections/main.py`** with the skeleton (predict comes in Task 9)

```python
"""FastAPI app for the collection scoring service.

Endpoints:
- GET  /health
- GET  /models
- GET  /model/{username}/{outcome}/info
- POST /predict_own   (Task 9)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import bigquery

# Make project root importable when running from services/collections/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from services.collections.auth import GCPAuthenticator, AuthenticationError  # noqa: E402
from services.collections.registry import CollectionRegistry  # noqa: E402
from services.collections.registered_model import RegisteredCollectionModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize once
try:
    authenticator = GCPAuthenticator()
    GCP_PROJECT_ID = authenticator.project_id
    config = load_config()
    BUCKET_NAME = config.get_bucket_name()
    ENVIRONMENT_PREFIX = config.get_environment_prefix()
    REGISTRY_TABLE = config.get_collection_registry_table()
    LANDING_TABLE = config.get_collection_landing_table()
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    registry = CollectionRegistry(REGISTRY_TABLE, bq_client)
except AuthenticationError as e:
    logger.error(f"Auth failed: {e}")
    raise

app = FastAPI(title="BGG Collection Scoring", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project_id": GCP_PROJECT_ID,
        "bucket": BUCKET_NAME,
        "environment": ENVIRONMENT_PREFIX,
        "registry_table": REGISTRY_TABLE,
        "landing_table": LANDING_TABLE,
    }


@app.get("/models")
def list_models(outcome: Optional[str] = None):
    entries = registry.list_active(outcome=outcome)
    return {
        "count": len(entries),
        "models": [
            {
                "username": e.username,
                "outcome": e.outcome,
                "model_version": e.model_version,
                "gcs_path": e.gcs_path,
                "finalize_through_year": e.finalize_through_year,
            }
            for e in entries
        ],
    }


@app.get("/model/{username}/{outcome}/info")
def model_info(username: str, outcome: str):
    entry = registry.lookup_latest(username, outcome)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active registry entry for ({username!r}, {outcome!r})",
        )
    return {
        "username": entry.username,
        "outcome": entry.outcome,
        "model_version": entry.model_version,
        "gcs_path": entry.gcs_path,
        "finalize_through_year": entry.finalize_through_year,
        "status": entry.status,
    }
```

- [ ] **Step 2: Smoke-test the import path**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from services.collections import main as m
print(type(m.app).__name__)
"
```
Expected: `FastAPI`.

- [ ] **Step 3: Commit**

```bash
git add services/collections/main.py
git commit -m "feat(collections): FastAPI app skeleton with /health, /models, /model info"
```

---

## Task 9: `/predict_own` endpoint

**Files:**
- Modify: `services/collections/main.py`

This is the load-bearing endpoint. It composes the registry, the registered model, change detection (optional), feature loading from BQ, scoring, and (optional) upload.

- [ ] **Step 1: Add request/response models and the endpoint**

Append to `services/collections/main.py` (above the closing of the file):

```python
from datetime import datetime, timezone  # noqa: E402
from typing import List  # noqa: E402

import pandas as pd  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from services.collections.change_detection import find_unscored  # noqa: E402
from services.collections.landing_uploader import (  # noqa: E402
    CollectionPredictionRow, CollectionPredictionsUploader,
)
from src.data.loader import BGGDataLoader  # noqa: E402


class PredictOwnRequest(BaseModel):
    username: str
    game_ids: Optional[List[int]] = None
    use_change_detection: bool = False
    upload_to_data_warehouse: bool = True
    model_version: Optional[int] = None  # None = latest active


class PredictOwnPrediction(BaseModel):
    game_id: int
    predicted_prob: float
    predicted_label: bool


class PredictOwnResponse(BaseModel):
    job_id: str
    username: str
    outcome: str
    model_version: int
    n_scored: int
    score_ts: datetime
    predictions: List[PredictOwnPrediction]


# Reuse a single BGGDataLoader for feature pulls
_loader: Optional[BGGDataLoader] = None


def _get_loader() -> BGGDataLoader:
    global _loader
    if _loader is None:
        _loader = BGGDataLoader(config.get_bigquery_config())
    return _loader


# Cache loaded pipelines: (username, outcome, version) -> (pipeline, threshold)
_PIPELINE_CACHE: dict = {}


def _load_pipeline(username: str, outcome: str, version: int):
    key = (username, outcome, version)
    if key not in _PIPELINE_CACHE:
        rcm = RegisteredCollectionModel(
            username=username,
            outcome=outcome,
            bucket_name=BUCKET_NAME,
            environment_prefix=ENVIRONMENT_PREFIX,
            project_id=GCP_PROJECT_ID,
        )
        pipeline, threshold, _ = rcm.load(version=version)
        _PIPELINE_CACHE[key] = (pipeline, threshold)
    return _PIPELINE_CACHE[key]


@app.post("/predict_own", response_model=PredictOwnResponse)
def predict_own(req: PredictOwnRequest):
    job_id = str(__import__("uuid").uuid4())

    # 1. Resolve registry entry
    entry = registry.lookup_latest(req.username, "own")
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active 'own' model for user {req.username!r}",
        )
    version = req.model_version if req.model_version is not None else entry.model_version

    # 2. Determine target game_ids
    if req.use_change_detection and req.game_ids:
        raise HTTPException(
            status_code=400,
            detail="Pass either game_ids or use_change_detection=true, not both",
        )
    if req.use_change_detection:
        game_ids = find_unscored(
            username=req.username,
            outcome="own",
            model_version=version,
            landing_table=LANDING_TABLE,
            candidate_table="bgg-data-warehouse.analytics.games_features",
            bq_client=bq_client,
        )
    elif req.game_ids:
        game_ids = list(req.game_ids)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide game_ids or use_change_detection=true",
        )

    score_ts = datetime.now(timezone.utc)

    if not game_ids:
        return PredictOwnResponse(
            job_id=job_id,
            username=req.username,
            outcome="own",
            model_version=version,
            n_scored=0,
            score_ts=score_ts,
            predictions=[],
        )

    # 3. Load pipeline + threshold
    try:
        pipeline, threshold = _load_pipeline(req.username, "own", version)
    except Exception as e:  # GCS load / pickle errors
        logger.exception("Failed loading pipeline")
        raise HTTPException(status_code=502, detail=f"Pipeline load failed: {e}")

    # 4. Pull features
    try:
        features_df = _get_loader().load_data().filter(
            __import__("polars").col("game_id").is_in(game_ids)
        )
        X = features_df.to_pandas()
    except Exception as e:
        logger.exception("Feature load failed")
        raise HTTPException(status_code=502, detail=f"Feature load failed: {e}")

    # 5. Score
    proba = pipeline.predict_proba(X)[:, 1]
    thr = threshold if threshold is not None else 0.5
    labels = (proba >= thr)

    predictions = [
        PredictOwnPrediction(
            game_id=int(gid),
            predicted_prob=float(p),
            predicted_label=bool(lbl),
        )
        for gid, p, lbl in zip(X["game_id"].tolist(), proba, labels)
    ]

    # 6. Upload
    if req.upload_to_data_warehouse and predictions:
        rows = [
            CollectionPredictionRow(
                job_id=job_id,
                username=req.username,
                game_id=p.game_id,
                outcome="own",
                predicted_prob=p.predicted_prob,
                predicted_label=p.predicted_label,
                threshold=threshold,
                model_name=f"collection_own_{req.username}",
                model_version=version,
                score_ts=score_ts,
            )
            for p in predictions
        ]
        CollectionPredictionsUploader(LANDING_TABLE, bq_client).upload(rows)

    return PredictOwnResponse(
        job_id=job_id,
        username=req.username,
        outcome="own",
        model_version=version,
        n_scored=len(predictions),
        score_ts=score_ts,
        predictions=predictions,
    )
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
uv run python -c "
from services.collections import main as m
print([r.path for r in m.app.routes if hasattr(r, 'path')])
"
```
Expected: list includes `/health`, `/models`, `/model/{username}/{outcome}/info`, `/predict_own`.

- [ ] **Step 3: Commit**

```bash
git add services/collections/main.py
git commit -m "feat(collections): /predict_own with change detection + landing upload"
```

---

## Task 10: Endpoint integration test (mocked GCS + BQ)

**Files:**
- Create: `tests/test_collections_service_endpoint.py`

- [ ] **Step 1: Write the test**

Create `tests/test_collections_service_endpoint.py`:

```python
"""End-to-end test of /predict_own with all GCP calls mocked."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mocked_app():
    """Build the app with all external deps mocked at import time."""
    with patch("services.collections.main.GCPAuthenticator") as auth_cls, \
         patch("services.collections.main.bigquery.Client") as bq_cls, \
         patch("services.collections.main.load_config") as cfg_fn:
        auth = MagicMock()
        auth.project_id = "test-project"
        auth_cls.return_value = auth
        bq_cls.return_value = MagicMock()

        cfg = MagicMock()
        cfg.get_bucket_name.return_value = "test-bucket"
        cfg.get_environment_prefix.return_value = "dev"
        cfg.get_collection_registry_table.return_value = "p.raw.collection_models_registry"
        cfg.get_collection_landing_table.return_value = "p.raw.collection_predictions_landing"
        cfg.get_bigquery_config.return_value = MagicMock()
        cfg_fn.return_value = cfg

        # Force a fresh import so module-level code re-runs under mocks
        import importlib
        from services.collections import main as m
        importlib.reload(m)
        yield m


def test_predict_own_with_explicit_game_ids_returns_predictions(mocked_app):
    m = mocked_app

    # Stub registry.lookup_latest
    entry = MagicMock()
    entry.model_version = 3
    entry.gcs_path = "gs://x/v3"
    m.registry.lookup_latest = MagicMock(return_value=entry)

    # Stub pipeline load
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    m._PIPELINE_CACHE.clear()
    with patch.object(m, "RegisteredCollectionModel") as rcm_cls:
        rcm = MagicMock()
        rcm.load.return_value = (pipeline, 0.5, {})
        rcm_cls.return_value = rcm

        # Stub feature loader
        m._loader = MagicMock()
        m._loader.load_data.return_value = pl.DataFrame({"game_id": [10, 11], "x": [1.0, 2.0]})

        # Stub uploader so we don't hit BQ
        with patch("services.collections.main.CollectionPredictionsUploader") as up_cls:
            up = MagicMock()
            up.upload.return_value = 2
            up_cls.return_value = up

            client = TestClient(m.app)
            resp = client.post(
                "/predict_own",
                json={
                    "username": "alice",
                    "game_ids": [10, 11],
                    "use_change_detection": False,
                    "upload_to_data_warehouse": True,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"
    assert body["outcome"] == "own"
    assert body["model_version"] == 3
    assert body["n_scored"] == 2
    assert isinstance(body["job_id"], str) and len(body["job_id"]) > 0
    probs = {p["game_id"]: p["predicted_prob"] for p in body["predictions"]}
    assert probs == {10: 0.8, 11: 0.4}
    labels = {p["game_id"]: p["predicted_label"] for p in body["predictions"]}
    assert labels == {10: True, 11: False}


def test_predict_own_returns_404_when_user_not_registered(mocked_app):
    m = mocked_app
    m.registry.lookup_latest = MagicMock(return_value=None)
    client = TestClient(m.app)

    resp = client.post(
        "/predict_own",
        json={"username": "ghost", "game_ids": [1], "upload_to_data_warehouse": False},
    )
    assert resp.status_code == 404


def test_predict_own_rejects_both_game_ids_and_change_detection(mocked_app):
    m = mocked_app
    entry = MagicMock(); entry.model_version = 1; entry.gcs_path = "gs://x"
    m.registry.lookup_latest = MagicMock(return_value=entry)
    client = TestClient(m.app)

    resp = client.post(
        "/predict_own",
        json={
            "username": "alice",
            "game_ids": [1],
            "use_change_detection": True,
            "upload_to_data_warehouse": False,
        },
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run the test**

Run: `uv run -m pytest tests/test_collections_service_endpoint.py -v`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_collections_service_endpoint.py
git commit -m "test(collections): /predict_own integration test with mocked GCP"
```

---

## Task 11: Dockerfile

**Files:**
- Create: `docker/collections.Dockerfile`

- [ ] **Step 1: Read the existing scoring Dockerfile**

Run: `cat docker/scoring.Dockerfile`

We will mirror its structure exactly — only the entrypoint differs.

- [ ] **Step 2: Create `docker/collections.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy lockfile and install deps first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY services/scoring/auth.py ./services/scoring/auth.py
COPY services/scoring/__init__.py ./services/scoring/__init__.py
COPY services/collections/ ./services/collections/
COPY config.yaml ./config.yaml

ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "services.collections.main:app", \
     "--host", "0.0.0.0", "--port", "8080"]
```

If the existing `scoring.Dockerfile` differs in base image, dep install command, or copy strategy, follow its pattern instead and only change the `CMD` line and the source copy paths.

- [ ] **Step 3: Build the image locally**

Run: `docker build -f docker/collections.Dockerfile -t bgg-collection-scoring:dev .`
Expected: builds successfully.

- [ ] **Step 4: Commit**

```bash
git add docker/collections.Dockerfile
git commit -m "feat(collections): Dockerfile for Cloud Run service"
```

---

## Task 12: Cloud Build config

**Files:**
- Create: `services/collections/cloudbuild.yaml`

- [ ] **Step 1: Create `services/collections/cloudbuild.yaml`**

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args:
    - 'build'
    - '-f'
    - 'docker/collections.Dockerfile'
    - '-t'
    - 'us-central1-docker.pkg.dev/$PROJECT_ID/bgg-predictive-models/collections:$COMMIT_SHA'
    - '.'

- name: 'gcr.io/cloud-builders/docker'
  args:
    - 'push'
    - 'us-central1-docker.pkg.dev/$PROJECT_ID/bgg-predictive-models/collections:$COMMIT_SHA'

- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
    - 'run'
    - 'deploy'
    - 'bgg-collection-scoring'
    - '--image'
    - 'us-central1-docker.pkg.dev/$PROJECT_ID/bgg-predictive-models/collections:$COMMIT_SHA'
    - '--platform'
    - 'managed'
    - '--region'
    - 'us-central1'
    - '--allow-unauthenticated'
    - '--max-instances'
    - '5'
    - '--memory'
    - '4Gi'
    - '--cpu'
    - '2'

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/bgg-predictive-models/collections:$COMMIT_SHA'
```

- [ ] **Step 2: Commit (do not submit yet — Cloud Build trigger is wired separately)**

```bash
git add services/collections/cloudbuild.yaml
git commit -m "feat(collections): Cloud Build config for bgg-collection-scoring"
```

---

## Task 13: Makefile targets for local dev

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Append targets to `Makefile`**

```makefile
### collection scoring service
.PHONY: start-collections-scoring stop-collections-scoring collections-scoring-service

start-collections-scoring:
	docker build -f docker/collections.Dockerfile -t bgg-collection-scoring:dev .
	docker run -d --name bgg-collection-scoring -p 8088:8080 \
		-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/sa.json \
		-v $(PWD)/credentials:/app/credentials:ro \
		bgg-collection-scoring:dev

stop-collections-scoring:
	-docker stop bgg-collection-scoring
	-docker rm bgg-collection-scoring

collections-scoring-service:
	@if [ -z "$(USERNAME)" ]; then echo "USERNAME required"; exit 1; fi
	curl -X POST http://localhost:8088/predict_own \
		-H "Content-Type: application/json" \
		-d '{"username":"$(USERNAME)","use_change_detection":true,"upload_to_data_warehouse":true}'
```

- [ ] **Step 2: Verify the make target announces correctly**

Run: `make collections-scoring-service`
Expected: prints `USERNAME required` and exits non-zero.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(collections): Makefile targets for local docker run + curl"
```

---

## Task 14: Daily GitHub Action

**Files:**
- Create: `.github/workflows/run-collection-scoring.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: Run Collection Scoring Service

on:
  schedule:
    - cron: '0 8 * * *'   # 1 hour after run-scoring-service.yml
  workflow_dispatch: {}

env:
  GCP_PROJECT_ID: bgg-predictive-models

jobs:
  score-collections:
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'PROD' || 'DEV' }}

    permissions:
      contents: read
      id-token: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set Environment Variables
        id: env
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "env_name=prod" >> $GITHUB_OUTPUT
          else
            echo "env_name=dev" >> $GITHUB_OUTPUT
          fi

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_BGG_ML }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Get Service URL
        id: get-url
        run: |
          SERVICE_URL=$(gcloud run services describe bgg-collection-scoring \
            --region us-central1 --format 'value(status.url)')
          echo "service_url=$SERVICE_URL" >> $GITHUB_OUTPUT

      - name: Get ID Token
        id: id-token
        run: |
          ID_TOKEN=$(gcloud auth print-identity-token)
          echo "::add-mask::$ID_TOKEN"
          echo "token=$ID_TOKEN" >> $GITHUB_OUTPUT

      - name: List Active Users from Registry
        id: users
        run: |
          USERS=$(bq query --use_legacy_sql=false --format=csv \
            "SELECT DISTINCT username FROM \`bgg-predictive-models.raw.collection_models_registry\` WHERE outcome='own' AND status='active'" \
            | tail -n +2)
          echo "users<<EOF" >> $GITHUB_OUTPUT
          echo "$USERS" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Score Each User
        id: score
        run: |
          TOTAL=0
          FAIL_USERS=()
          while IFS= read -r USER; do
            [ -z "$USER" ] && continue
            echo "=== Scoring $USER ==="
            RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
              "${{ steps.get-url.outputs.service_url }}/predict_own" \
              -H "Authorization: Bearer ${{ steps.id-token.outputs.token }}" \
              -H "Content-Type: application/json" \
              -d "{\"username\":\"$USER\",\"use_change_detection\":true,\"upload_to_data_warehouse\":true}" \
              --max-time 1800)
            HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
            BODY=$(echo "$RESPONSE" | head -n-1)
            if [ "$HTTP_CODE" != "200" ]; then
              echo "FAIL ($HTTP_CODE) for $USER: $BODY"
              FAIL_USERS+=("$USER")
              continue
            fi
            N=$(echo "$BODY" | jq -r '.n_scored')
            echo "$USER: scored $N"
            TOTAL=$((TOTAL + N))
          done <<< "${{ steps.users.outputs.users }}"

          echo "total=$TOTAL" >> $GITHUB_OUTPUT
          echo "failed=${FAIL_USERS[*]}" >> $GITHUB_OUTPUT
          if [ ${#FAIL_USERS[@]} -gt 0 ]; then
            echo "::error::failed users: ${FAIL_USERS[*]}"
            exit 1
          fi

      - name: Job Summary
        if: always()
        run: |
          echo "## Collection Scoring Results" >> $GITHUB_STEP_SUMMARY
          echo "- Environment: ${{ steps.env.outputs.env_name }}" >> $GITHUB_STEP_SUMMARY
          echo "- Total scored: ${{ steps.score.outputs.total }}" >> $GITHUB_STEP_SUMMARY
          echo "- Failed users: ${{ steps.score.outputs.failed }}" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Validate workflow syntax**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/run-collection-scoring.yml'))"`
Expected: parses cleanly (no exception).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/run-collection-scoring.yml
git commit -m "feat(collections): daily GitHub Action driven by registry"
```

---

## Task 15: Full test sweep + smoke

- [ ] **Step 1: Run all collection-service tests**

Run: `uv run -m pytest tests/test_collections_service_*.py -v`
Expected: 8 tests pass (3 registry + 2 change_detection + 3 endpoint).

- [ ] **Step 2: Lint**

Run: `uv run ruff check services/collections/ tests/test_collections_service_*.py`
Expected: no errors related to the new code.

- [ ] **Step 3: Smoke-test the full module graph**

Run:
```bash
uv run python -c "
from services.collections.main import app
from services.collections.registry import CollectionRegistry, RegistryEntry
from services.collections.change_detection import find_unscored, build_unscored_query
from services.collections.landing_uploader import (
    CollectionPredictionsUploader, CollectionPredictionRow
)
print('routes:', sorted(r.path for r in app.routes if hasattr(r, 'path')))
print('OK')
"
```
Expected: routes list includes `/health`, `/models`, `/model/{username}/{outcome}/info`, `/predict_own`; final `OK`.

- [ ] **Step 4: Commit any final fixes**

If 1–3 surfaced issues, fix and commit:

```bash
git add -A
git commit -m "fix(collections): post-implementation issues from full sweep"
```

---
