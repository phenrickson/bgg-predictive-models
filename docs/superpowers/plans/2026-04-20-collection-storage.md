# Collection Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime-created `analytics.collections` table with a Terraform-managed `collections.user_collections` table keyed on `(username, game_id)`, written via idempotent MERGE with soft-delete semantics.

**Architecture:** Terraform owns the BQ dataset, table, schema (JSON file), primary key, and clustering. Python `CollectionStorage` becomes a thin MERGE runner: filter to boardgames, dedupe, assert non-empty, stage rows, MERGE. Reads become trivial `SELECT ... WHERE removed_at IS NULL`.

**Tech Stack:** Terraform (hashicorp/google ~> 5.0), BigQuery, Python 3.12, polars, google-cloud-bigquery, pytest, uv.

**Spec:** [2026-04-20-collection-storage-design.md](../specs/2026-04-20-collection-storage-design.md)

---

## File Structure

**Terraform (new/modified):**
- Create: `terraform/schemas/user_collections.json` — BQ schema JSON, loaded by Terraform.
- Modify: `terraform/bigquery.tf` — add `google_bigquery_dataset.collections` and `google_bigquery_table.user_collections` with `table_constraints`.

**Config (modified):**
- Modify: `src/utils/config.py` — extend `BigQueryConfig` with `collections_dataset: str = "collections"`; populate in `load_config`.
- Delete: `config/bigquery.yaml` after the `collections` section is no longer read (storage no longer consumes it).

**Source (modified):**
- Modify: `src/collection/collection_loader.py` — add `excludesubtype=boardgameexpansion` to request params.
- Rewrite: `src/collection/collection_storage.py` — new, minimal surface: `save_collection`, `get_latest_collection`, `get_owned_game_ids`. No YAML schema loading, no `create_table`, no `collection_version`.

**Tests (modified/new):**
- Rewrite: `tests/test_collection_storage.py` — pytest-based integration test against dev BQ.
- Create: `tests/test_collection_loader_filter.py` — unit test that `excludesubtype` is in request params.

---

## Working Directory and Commands

All commands run from `/Users/phenrickson/Documents/projects/bgg-predictive-models` unless otherwise specified.

Python commands use `uv run python ...` and `uv run pytest ...` (per project convention).

Terraform commands run from `terraform/` subdirectory with dev workspace selected. The engineer must authenticate to GCP before running Terraform: `gcloud auth application-default login` and have access to the `bgg-predictive-models` project.

---

## Task 1: Terraform schema file for `user_collections`

Create the schema JSON that Terraform will load. This is the source of truth for the column list.

**Files:**
- Create: `terraform/schemas/user_collections.json`

- [ ] **Step 1: Create the schema JSON**

Write to `terraform/schemas/user_collections.json`:

```json
[
  {"name": "username", "type": "STRING", "mode": "REQUIRED", "description": "BGG username"},
  {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED", "description": "BGG game ID"},
  {"name": "game_name", "type": "STRING", "mode": "NULLABLE", "description": "Game name as returned by BGG collection API"},
  {"name": "subtype", "type": "STRING", "mode": "NULLABLE", "description": "Always 'boardgame' — filter applied at write"},
  {"name": "collection_id", "type": "INTEGER", "mode": "NULLABLE", "description": "BGG collection item ID (collid)"},
  {"name": "owned", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "previously_owned", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "for_trade", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "want", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "want_to_play", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "want_to_buy", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "wishlist", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "wishlist_priority", "type": "INTEGER", "mode": "NULLABLE", "description": "Wishlist priority (1-5)"},
  {"name": "preordered", "type": "BOOL", "mode": "NULLABLE"},
  {"name": "user_rating", "type": "FLOAT", "mode": "NULLABLE", "description": "User's rating (BGG 1-10)"},
  {"name": "user_comment", "type": "STRING", "mode": "NULLABLE"},
  {"name": "last_modified", "type": "TIMESTAMP", "mode": "NULLABLE", "description": "When BGG's collection item was last modified"},
  {"name": "first_seen_at", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Row insert time, never updated"},
  {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Row insert or update time"},
  {"name": "removed_at", "type": "TIMESTAMP", "mode": "NULLABLE", "description": "Soft-delete marker; NULL if game is currently in user's collection"}
]
```

- [ ] **Step 2: Validate JSON**

Run: `uv run python -c "import json; json.load(open('terraform/schemas/user_collections.json'))"`

Expected: no output (valid JSON).

- [ ] **Step 3: Commit**

```bash
git add terraform/schemas/user_collections.json
git commit -m "feat(terraform): add user_collections BQ schema"
```

---

## Task 2: Terraform dataset + table resource

Add the `collections` dataset and `user_collections` table to `terraform/bigquery.tf`.

**Files:**
- Modify: `terraform/bigquery.tf` — append new resources at end of file.

- [ ] **Step 1: Append dataset + table resources**

Add to the end of `terraform/bigquery.tf`:

```hcl
# =============================================================================
# Collections Dataset (user collection state)
# =============================================================================

resource "google_bigquery_dataset" "collections" {
  dataset_id  = "collections"
  project     = var.project_id
  location    = var.location
  description = "User collection state from BGG, upserted per user"

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

resource "google_bigquery_table" "user_collections" {
  dataset_id          = google_bigquery_dataset.collections.dataset_id
  table_id            = "user_collections"
  project             = var.project_id
  description         = "One row per (username, game_id); soft-deletes via removed_at"
  deletion_protection = true

  clustering = ["username", "game_id"]

  schema = file("${path.module}/schemas/user_collections.json")

  table_constraints {
    primary_key {
      columns = ["username", "game_id"]
    }
  }

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }
}
```

- [ ] **Step 2: Run `terraform fmt`**

```bash
cd terraform && terraform fmt
```

Expected: file reformatted if indentation was off; no errors.

- [ ] **Step 3: Run `terraform init` (if not already initialized)**

```bash
cd terraform && terraform init
```

Expected: `Terraform has been successfully initialized!`

- [ ] **Step 4: Run `terraform plan`**

```bash
cd terraform && terraform plan
```

Expected: plan shows 2 resources to add: `google_bigquery_dataset.collections` and `google_bigquery_table.user_collections`. No other changes. Review output for any unexpected diffs elsewhere.

- [ ] **Step 5: Run `terraform apply`**

```bash
cd terraform && terraform apply
```

Type `yes` when prompted. Expected: `Apply complete! Resources: 2 added, 0 changed, 0 destroyed.`

- [ ] **Step 6: Verify table via `bq show`**

```bash
bq show --format=prettyjson bgg-predictive-models:collections.user_collections | head -80
```

Expected: JSON output includes the full schema, `"clusteringFields": ["username", "game_id"]`, and a `tableConstraints.primaryKey` block listing `["username", "game_id"]`.

- [ ] **Step 7: Verify no drift on re-plan**

```bash
cd terraform && terraform plan
```

Expected: `No changes. Your infrastructure matches the configuration.`

- [ ] **Step 8: Commit**

```bash
git add terraform/bigquery.tf
git commit -m "feat(terraform): add collections dataset and user_collections table"
```

---

## Task 3: Add `collections_dataset` to `BigQueryConfig`

Extend the config so Python code can read the dataset name without hardcoding.

**Files:**
- Modify: `src/utils/config.py:52-68` — add field to `BigQueryConfig`.
- Modify: `src/utils/config.py:297-304` — populate in `get_bigquery_config`.

- [ ] **Step 1: Add field to `BigQueryConfig`**

Edit `src/utils/config.py`, in the `BigQueryConfig` dataclass (around line 52), add `collections_dataset` after `datasets`:

```python
@dataclass
class BigQueryConfig:
    """Configuration for BigQuery connection."""

    project_id: str
    dataset: str
    table: str
    credentials_path: Optional[str] = None
    location: str = "US"
    datasets: Optional[Dict[str, str]] = None
    collections_dataset: str = "collections"

    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client using Google Application Default Credentials."""
        try:
            credentials, _ = default()
            return bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception:
            raise
```

- [ ] **Step 2: Populate in `get_bigquery_config`**

In `src/utils/config.py` around line 297, update `get_bigquery_config`. The collection storage lives in the **ML project**, not the data warehouse project — so `get_bigquery_config()` needs to be aware of which project owns the collections. Edit the method to route the project id through `ml_project_id`:

```python
    def get_bigquery_config(self) -> BigQueryConfig:
        """Get BigQuery configuration. `project_id` is the data warehouse project
        (for reads of game features); `collections_dataset` resolves inside
        `ml_project_id`, which the collection storage reads separately.
        """
        return BigQueryConfig(
            project_id=self.data_warehouse.project_id,
            dataset=self.data_warehouse.features_dataset,
            table=self.data_warehouse.features_table,
            location=self.data_warehouse.location,
            collections_dataset="collections",
        )
```

(The collection storage uses `config.ml_project_id` directly for the project, and `bq_config.collections_dataset` for the dataset. See Task 6.)

- [ ] **Step 3: Run existing tests to confirm no regression**

```bash
uv run pytest tests/test_collection_storage.py -x 2>&1 | tail -30
```

Expected: the existing test may fail for other reasons (we're about to rewrite it), but no `ImportError` or `TypeError` from the config change. If everything else breaks, stop and fix.

- [ ] **Step 4: Commit**

```bash
git add src/utils/config.py
git commit -m "feat(config): add collections_dataset to BigQueryConfig"
```

---

## Task 4: Loader request-side filter (exclude expansions)

Add `excludesubtype=boardgameexpansion` to the BGG API request.

**Files:**
- Modify: `src/collection/collection_loader.py:96-103` — add param.
- Create: `tests/test_collection_loader_filter.py` — unit test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_collection_loader_filter.py`:

```python
"""Unit test that BGGCollectionLoader requests exclude expansions."""

from unittest.mock import patch, MagicMock

from src.collection.collection_loader import BGGCollectionLoader


def test_get_collection_excludes_expansions_in_request(monkeypatch):
    """Verify the request params include excludesubtype=boardgameexpansion."""
    monkeypatch.setenv("BGG_API_TOKEN", "fake-token-for-test")
    loader = BGGCollectionLoader(username="anyuser")

    captured_params = {}

    def fake_make_request(endpoint, params):
        captured_params.update(params)
        mock_resp = MagicMock()
        mock_resp.content = b"<items totalitems='0'></items>"
        return mock_resp

    with patch.object(loader, "_make_request", side_effect=fake_make_request):
        loader.get_collection()

    assert captured_params.get("subtype") == "boardgame"
    assert captured_params.get("excludesubtype") == "boardgameexpansion"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_collection_loader_filter.py -v
```

Expected: FAIL with `AssertionError` on the `excludesubtype` line.

- [ ] **Step 3: Add the param to the loader**

Edit `src/collection/collection_loader.py` in `get_collection`, around line 96-103:

```python
        try:
            # Build parameters for collection request
            params = {
                "username": self.username,
                "subtype": "boardgame",  # only retrieve boardgame items
                "excludesubtype": "boardgameexpansion",  # exclude expansions
                "stats": "1",  # Include game statistics
            }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_collection_loader_filter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/collection_loader.py tests/test_collection_loader_filter.py
git commit -m "feat(collection): exclude expansions from BGG collection request"
```

---

## Task 5: Rewrite `CollectionStorage` — failing tests first

Write the new integration tests against the Terraform-managed table. These drive the rewrite.

**Context for the engineer:** Integration tests run against real dev BigQuery. They require:
- `GOOGLE_APPLICATION_CREDENTIALS` or ADC set up (`gcloud auth application-default login`).
- `ML_PROJECT_ID=bgg-predictive-models` (or `GCP_PROJECT_ID`) in `.env`.
- The table from Task 2 already applied.

Tests use unique usernames (prefixed with `_test_`) so they can run against real dev without colliding with real users. Teardown deletes those rows.

**Files:**
- Rewrite: `tests/test_collection_storage.py`

- [ ] **Step 1: Write the new integration test file**

Overwrite `tests/test_collection_storage.py`:

```python
"""Integration tests for CollectionStorage against dev BigQuery.

Requires ADC (gcloud auth application-default login) and ML_PROJECT_ID set.
Uses `_test_*` username prefixes and cleans them up at teardown.
"""

from datetime import datetime
from typing import Iterable

import polars as pl
import pytest

from src.collection.collection_storage import CollectionStorage


TEST_USER_A = "_test_user_a"
TEST_USER_B = "_test_user_b"


def _row(game_id: int, *, name: str = "G", owned: bool = True,
         user_rating: float | None = None,
         subtype: str = "boardgame") -> dict:
    """Build a minimal collection row matching what the loader emits."""
    return {
        "game_id": game_id,
        "game_name": name,
        "subtype": subtype,
        "collection_id": None,
        "owned": owned,
        "previously_owned": False,
        "for_trade": False,
        "want": False,
        "want_to_play": False,
        "want_to_buy": False,
        "wishlist": False,
        "wishlist_priority": None,
        "preordered": False,
        "last_modified": None,
        "user_rating": user_rating,
        "user_comment": None,
    }


def _df(rows: Iterable[dict]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


@pytest.fixture
def storage():
    return CollectionStorage(environment="dev")


@pytest.fixture(autouse=True)
def cleanup(storage):
    yield
    storage.delete_user_rows(TEST_USER_A)
    storage.delete_user_rows(TEST_USER_B)


def test_initial_load_inserts_all_rows(storage):
    df = _df([_row(1), _row(2), _row(3)])
    storage.save_collection(TEST_USER_A, df)

    result = storage.get_latest_collection(TEST_USER_A)
    assert result is not None
    assert result.height == 3
    assert set(result["game_id"].to_list()) == {1, 2, 3}
    # first_seen_at == updated_at on fresh insert
    assert result.filter(
        pl.col("first_seen_at") != pl.col("updated_at")
    ).height == 0


def test_idempotent_repull(storage):
    df = _df([_row(1), _row(2)])
    storage.save_collection(TEST_USER_A, df)
    first = storage.get_latest_collection(TEST_USER_A)

    storage.save_collection(TEST_USER_A, df)
    second = storage.get_latest_collection(TEST_USER_A)

    # Still 2 rows, same game_ids, no soft-deletes.
    assert second.height == 2
    assert set(second["game_id"].to_list()) == {1, 2}
    # first_seen_at unchanged; updated_at advanced.
    first_by_id = {row["game_id"]: row for row in first.iter_rows(named=True)}
    second_by_id = {row["game_id"]: row for row in second.iter_rows(named=True)}
    for gid in (1, 2):
        assert first_by_id[gid]["first_seen_at"] == second_by_id[gid]["first_seen_at"]
        assert second_by_id[gid]["updated_at"] >= first_by_id[gid]["updated_at"]


def test_modified_row_updates_data(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1, user_rating=5.0), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1, user_rating=9.5), _row(2)]))

    result = storage.get_latest_collection(TEST_USER_A)
    rating = result.filter(pl.col("game_id") == 1)["user_rating"].item()
    assert rating == 9.5


def test_new_row_inserts(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2), _row(3)]))

    result = storage.get_latest_collection(TEST_USER_A)
    assert result.height == 3
    assert set(result["game_id"].to_list()) == {1, 2, 3}


def test_removed_row_soft_deletes(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2), _row(3)]))
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(3)]))

    # get_latest_collection filters removed_at IS NULL.
    visible = storage.get_latest_collection(TEST_USER_A)
    assert set(visible["game_id"].to_list()) == {1, 3}

    # Raw query to verify game_id=2 has removed_at set.
    raw = storage.get_all_rows_including_removed(TEST_USER_A)
    row2 = raw.filter(pl.col("game_id") == 2)
    assert row2.height == 1
    assert row2["removed_at"].item() is not None


def test_readded_row_clears_removed_at(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1)]))  # soft-delete 2
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))  # re-add

    visible = storage.get_latest_collection(TEST_USER_A)
    assert set(visible["game_id"].to_list()) == {1, 2}

    # first_seen_at on game_id=2 should be the original insert, not the re-add.
    raw = storage.get_all_rows_including_removed(TEST_USER_A)
    row2 = raw.filter(pl.col("game_id") == 2)
    assert row2["removed_at"].item() is None


def test_cross_user_isolation(storage):
    storage.save_collection(TEST_USER_A, _df([_row(10), _row(11)]))
    storage.save_collection(TEST_USER_B, _df([_row(20)]))
    storage.save_collection(TEST_USER_B, _df([]))  # should raise; see next test

    # User A untouched after whatever happened to B.
    a = storage.get_latest_collection(TEST_USER_A)
    assert a.height == 2


def test_empty_dataframe_is_rejected(storage):
    with pytest.raises(ValueError, match=r"empty"):
        storage.save_collection(TEST_USER_A, _df([]))


def test_duplicate_rows_are_rejected(storage):
    with pytest.raises(ValueError, match=r"duplicate"):
        storage.save_collection(TEST_USER_A, _df([_row(1), _row(1)]))


def test_expansion_rows_are_filtered_out(storage):
    df = _df([_row(1), _row(2, subtype="boardgameexpansion"), _row(3)])
    storage.save_collection(TEST_USER_A, df)

    result = storage.get_latest_collection(TEST_USER_A)
    assert set(result["game_id"].to_list()) == {1, 3}


def test_get_owned_game_ids_filters_to_owned(storage):
    df = _df([_row(1, owned=True), _row(2, owned=False), _row(3, owned=True)])
    storage.save_collection(TEST_USER_A, df)

    ids = storage.get_owned_game_ids(TEST_USER_A)
    assert set(ids) == {1, 3}
```

Fix up the `test_cross_user_isolation` test — it calls `save_collection` with an empty frame, which will raise. Remove that line so the test is well-formed:

Actually, rewrite the test to not overload it. Replace the `test_cross_user_isolation` body with:

```python
def test_cross_user_isolation(storage):
    storage.save_collection(TEST_USER_A, _df([_row(10), _row(11)]))
    storage.save_collection(TEST_USER_B, _df([_row(20)]))

    # Re-pull B with a different set.
    storage.save_collection(TEST_USER_B, _df([_row(21)]))

    # User A untouched by anything that happened to user B.
    a = storage.get_latest_collection(TEST_USER_A)
    assert set(a["game_id"].to_list()) == {10, 11}

    b = storage.get_latest_collection(TEST_USER_B)
    assert set(b["game_id"].to_list()) == {21}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_collection_storage.py -v 2>&1 | tail -40
```

Expected: all tests fail. Most will fail with `AttributeError` (methods like `delete_user_rows`, `get_all_rows_including_removed` don't exist yet) or with the old behavior leaking through. A few may error during `__init__` if the config doesn't resolve. **Do not fix by monkey-patching tests — fix the storage class in Task 6.**

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_collection_storage.py
git commit -m "test(collection): integration tests for new CollectionStorage (failing)"
```

---

## Task 6: Rewrite `CollectionStorage`

Replace the file with the new minimal implementation. The tests from Task 5 will drive this.

**Files:**
- Rewrite: `src/collection/collection_storage.py`

- [ ] **Step 1: Overwrite `src/collection/collection_storage.py`**

Replace the entire file contents with:

```python
"""Storage layer for BGG user collections in BigQuery.

One row per (username, game_id) in `collections.user_collections`. Writes go
through a single MERGE that inserts new rows, updates changed rows, and
soft-deletes rows that are no longer present in the source. Schema is managed
by Terraform — this module does not create tables.
"""

import logging
from typing import Optional

import pandas as pd
import polars as pl
from google.cloud import bigquery

from src.utils.config import load_config

logger = logging.getLogger(__name__)


TABLE_COLUMNS = [
    "game_id",
    "game_name",
    "subtype",
    "collection_id",
    "owned",
    "previously_owned",
    "for_trade",
    "want",
    "want_to_play",
    "want_to_buy",
    "wishlist",
    "wishlist_priority",
    "preordered",
    "user_rating",
    "user_comment",
    "last_modified",
]


class CollectionStorage:
    """Upsert and read user collections in `collections.user_collections`."""

    def __init__(self, environment: str = "dev"):
        self.environment = environment
        config = load_config()
        bq_config = config.get_bigquery_config()

        self.project_id = config.ml_project_id
        self.dataset_id = bq_config.collections_dataset
        self.table_id = "user_collections"
        self.location = bq_config.location

        self.client = bigquery.Client(project=self.project_id)
        self.fq_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        logger.info(
            f"CollectionStorage initialized: {self.fq_table} (env={environment})"
        )

    def _prepare_rows(self, username: str, df: pl.DataFrame) -> pd.DataFrame:
        """Validate input and return a pandas DataFrame ready for staging.

        - Filters to subtype == 'boardgame'.
        - Rejects empty input.
        - Rejects duplicate (username, game_id) after filtering.
        - Casts `last_modified` from string to datetime.
        """
        if df.height == 0:
            raise ValueError(
                f"Cannot save empty collection for user '{username}'. "
                "Refusing to soft-delete every row in this user's collection."
            )

        if "subtype" in df.columns:
            df = df.filter(pl.col("subtype") == "boardgame")

        if df.height == 0:
            raise ValueError(
                f"Collection for user '{username}' is empty after filtering to "
                "boardgame subtype. Refusing to soft-delete every row."
            )

        dupes = df.group_by("game_id").len().filter(pl.col("len") > 1)
        if dupes.height > 0:
            raise ValueError(
                f"duplicate game_id rows in collection for '{username}': "
                f"{dupes['game_id'].to_list()}"
            )

        missing = [c for c in TABLE_COLUMNS if c not in df.columns]
        for col in missing:
            df = df.with_columns(pl.lit(None).alias(col))

        df = df.select(TABLE_COLUMNS)
        pdf = df.to_pandas()

        if "last_modified" in pdf.columns:
            pdf["last_modified"] = pd.to_datetime(
                pdf["last_modified"], errors="coerce"
            )

        return pdf

    def save_collection(self, username: str, collection_df: pl.DataFrame) -> None:
        """Upsert one user's collection via MERGE.

        Soft-deletes rows present in the table but not in `collection_df`.
        Raises ValueError on empty input or duplicate (username, game_id).
        """
        pdf = self._prepare_rows(username, collection_df)

        # Stage to a temp table — MERGE's USING clause needs a table, not params.
        staging_table = (
            f"{self.project_id}.{self.dataset_id}._staging_{username}"
        )
        staging_table = staging_table.replace("-", "_")  # BQ identifiers

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=[
                bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("game_name", "STRING"),
                bigquery.SchemaField("subtype", "STRING"),
                bigquery.SchemaField("collection_id", "INTEGER"),
                bigquery.SchemaField("owned", "BOOL"),
                bigquery.SchemaField("previously_owned", "BOOL"),
                bigquery.SchemaField("for_trade", "BOOL"),
                bigquery.SchemaField("want", "BOOL"),
                bigquery.SchemaField("want_to_play", "BOOL"),
                bigquery.SchemaField("want_to_buy", "BOOL"),
                bigquery.SchemaField("wishlist", "BOOL"),
                bigquery.SchemaField("wishlist_priority", "INTEGER"),
                bigquery.SchemaField("preordered", "BOOL"),
                bigquery.SchemaField("user_rating", "FLOAT"),
                bigquery.SchemaField("user_comment", "STRING"),
                bigquery.SchemaField("last_modified", "TIMESTAMP"),
            ],
        )

        logger.info(
            f"Staging {len(pdf)} rows for '{username}' to {staging_table}"
        )
        self.client.load_table_from_dataframe(
            pdf, staging_table, job_config=job_config
        ).result()

        merge_sql = f"""
        MERGE `{self.fq_table}` T
        USING (
          SELECT @username AS username, * FROM `{staging_table}`
        ) S
        ON T.username = S.username AND T.game_id = S.game_id

        WHEN MATCHED THEN UPDATE SET
          game_name         = S.game_name,
          subtype           = S.subtype,
          collection_id     = S.collection_id,
          owned             = S.owned,
          previously_owned  = S.previously_owned,
          for_trade         = S.for_trade,
          want              = S.want,
          want_to_play      = S.want_to_play,
          want_to_buy       = S.want_to_buy,
          wishlist          = S.wishlist,
          wishlist_priority = S.wishlist_priority,
          preordered        = S.preordered,
          user_rating       = S.user_rating,
          user_comment      = S.user_comment,
          last_modified     = S.last_modified,
          updated_at        = CURRENT_TIMESTAMP(),
          removed_at        = NULL

        WHEN NOT MATCHED BY TARGET THEN INSERT (
          username, game_id, game_name, subtype, collection_id,
          owned, previously_owned, for_trade, want, want_to_play, want_to_buy,
          wishlist, wishlist_priority, preordered,
          user_rating, user_comment, last_modified,
          first_seen_at, updated_at, removed_at
        ) VALUES (
          S.username, S.game_id, S.game_name, S.subtype, S.collection_id,
          S.owned, S.previously_owned, S.for_trade, S.want, S.want_to_play, S.want_to_buy,
          S.wishlist, S.wishlist_priority, S.preordered,
          S.user_rating, S.user_comment, S.last_modified,
          CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), NULL
        )

        WHEN NOT MATCHED BY SOURCE
          AND T.username = @username
          AND T.removed_at IS NULL
        THEN UPDATE SET
          removed_at = CURRENT_TIMESTAMP(),
          updated_at = CURRENT_TIMESTAMP()
        """

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        logger.info(f"Running MERGE for '{username}'")
        self.client.query(merge_sql, job_config=query_config).result()

        self.client.delete_table(staging_table, not_found_ok=True)
        logger.info(f"MERGE complete for '{username}'")

    def get_latest_collection(self, username: str) -> Optional[pl.DataFrame]:
        """Return currently active rows for `username` (removed_at IS NULL)."""
        sql = f"""
        SELECT *
        FROM `{self.fq_table}`
        WHERE username = @username
          AND removed_at IS NULL
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        pdf = self.client.query(sql, job_config=cfg).to_dataframe()
        if len(pdf) == 0:
            return None
        return pl.from_pandas(pdf)

    def get_all_rows_including_removed(
        self, username: str
    ) -> Optional[pl.DataFrame]:
        """Return every row for `username`, including soft-deleted ones.

        Used by tests and by callers that need to inspect removal history.
        """
        sql = f"""
        SELECT *
        FROM `{self.fq_table}`
        WHERE username = @username
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        pdf = self.client.query(sql, job_config=cfg).to_dataframe()
        if len(pdf) == 0:
            return None
        return pl.from_pandas(pdf)

    def get_owned_game_ids(self, username: str) -> Optional[list[int]]:
        """Return game_ids where owned = TRUE and the row is not soft-deleted."""
        df = self.get_latest_collection(username)
        if df is None:
            return None
        return df.filter(pl.col("owned") == True)["game_id"].to_list()

    def delete_user_rows(self, username: str) -> None:
        """Hard-delete every row for `username`. Used by test teardown."""
        sql = f"DELETE FROM `{self.fq_table}` WHERE username = @username"
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        self.client.query(sql, job_config=cfg).result()
```

- [ ] **Step 2: Run the integration tests**

```bash
uv run pytest tests/test_collection_storage.py -v 2>&1 | tail -60
```

Expected: all 11 tests pass.

If a test fails, read the error carefully. Common failure modes:
- `NotFound: 404 dataset collections` — Task 2's `terraform apply` wasn't run.
- `Invalid field name` — schema JSON and MERGE column list drift; compare Task 1 JSON to `TABLE_COLUMNS` and the MERGE body.
- `Permission denied` — ADC isn't set up, run `gcloud auth application-default login`.
- `Staging table not found` when running MERGE — the load job failed silently; add `print(job.errors)` after `.result()` to surface it.

Do not skip or mock any test. If a test is genuinely revealing a bug in the storage class, fix the storage class.

- [ ] **Step 3: Commit**

```bash
git add src/collection/collection_storage.py
git commit -m "feat(collection): upsert-based CollectionStorage keyed on (username, game_id)"
```

---

## Task 7: Remove obsolete config entries

The old `config/bigquery.yaml` had a `collections` table section that CollectionStorage used to read. It's dead now.

**Files:**
- Modify: `config/bigquery.yaml` — remove `collections` entry under `tables:`.

- [ ] **Step 1: Verify nothing else reads `bigquery.yaml` for `collections`**

```bash
grep -rn "bigquery.yaml" src/ 2>&1 | grep -v __pycache__
grep -rn "tables.*collections\|collections.*tables" src/ 2>&1 | grep -v __pycache__ | grep -v test_
```

Expected: no hits referencing `bigquery.yaml` from `collection_storage.py` (we removed that code in Task 6). Any other consumer of the `collections` table entry should be investigated — if found, stop and ask.

- [ ] **Step 2: Remove the `collections` section from `config/bigquery.yaml`**

Edit `config/bigquery.yaml`. Delete lines 155 through the end of the file (the `# User Collections` comment and the entire `collections:` block).

- [ ] **Step 3: Run the full collection test suite**

```bash
uv run pytest tests/test_collection_loader.py tests/test_collection_loader_filter.py tests/test_collection_storage.py tests/test_collection_processor.py tests/test_collection_splitter.py -v 2>&1 | tail -40
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add config/bigquery.yaml
git commit -m "chore(config): remove dead collections entry from bigquery.yaml"
```

---

## Task 8: Update the walkthrough notebook

The walkthrough at `notebooks/collection_walkthrough.qmd` has a "Store a collection" section referencing `create_table()` and the old behavior. Update so the narration matches the new storage model.

**Files:**
- Modify: `notebooks/collection_walkthrough.qmd:87-106` — update Section 2.

- [ ] **Step 1: Rewrite Section 2 of the notebook**

In `notebooks/collection_walkthrough.qmd`, replace the Section 2 block (lines 87-106) with:

````markdown
## 2. Store a collection

Upsert the raw snapshot into `collections.user_collections`. The table is
managed by Terraform (`terraform/bigquery.tf` + `terraform/schemas/user_collections.json`)
so the Python code never creates it — if the table is missing, run
`terraform apply` from the `terraform/` directory.

Each pull MERGEs against the existing state for this user: new games are
inserted, existing games are updated in place, and games that have disappeared
from the collection are soft-deleted via `removed_at`. Re-pulling the same
data is a no-op (idempotent).

```{python}
from src.collection.collection_storage import CollectionStorage

bq_storage = CollectionStorage(environment="dev")
bq_storage.save_collection(USERNAME, raw_collection)

reloaded = bq_storage.get_latest_collection(USERNAME)
print(f"active rows: {reloaded.height if reloaded is not None else 0}")
```

### Inspect lifecycle timestamps

```{python}
if reloaded is not None:
    reloaded.select(["game_id", "game_name", "first_seen_at", "updated_at"]).head(5)
```
````

- [ ] **Step 2: Commit**

```bash
git add notebooks/collection_walkthrough.qmd
git commit -m "docs(walkthrough): document upsert storage model"
```

---

## Task 9: Final end-to-end verification

Run everything together to catch cross-task drift.

- [ ] **Step 1: Full collection test suite against real dev BQ**

```bash
uv run pytest tests/test_collection_loader.py tests/test_collection_loader_filter.py tests/test_collection_storage.py tests/test_collection_processor.py tests/test_collection_splitter.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Terraform no-drift check**

```bash
cd terraform && terraform plan
```

Expected: `No changes. Your infrastructure matches the configuration.`

- [ ] **Step 3: Manual smoke test (one real user)**

```bash
uv run python -c "
from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_storage import CollectionStorage

loader = BGGCollectionLoader('phenrickson')
df = loader.get_collection()
print(f'pulled: {df.height} rows, subtypes: {df[\"subtype\"].unique().to_list()}')

storage = CollectionStorage(environment='dev')
storage.save_collection('phenrickson', df)
active = storage.get_latest_collection('phenrickson')
print(f'active after save: {active.height} rows')
"
```

Expected:
- `pulled: N rows, subtypes: ['boardgame']` (no `boardgameexpansion`).
- `active after save: N rows` matches.
- Re-run the same command: still `N rows`, no errors (idempotent).

- [ ] **Step 4: Done**

If all three verification steps pass, the implementation is complete. The spec's "Risks" section items are covered:
- Empty-input guard: `test_empty_dataframe_is_rejected` passes.
- Duplicate-source guard: `test_duplicate_rows_are_rejected` passes.
- Schema drift: the full test suite touches every column path.

---

## Self-review notes

Spec coverage audit (each spec section → task):

- Scope: new `collections` dataset → Task 2. Table with PK → Task 1 + Task 2. Rewritten `save_collection` with pre-MERGE filter → Task 6. Loader exclude-subtype → Task 4. Integration test → Task 5 + Task 6. ✓
- Design § Table: every column in Task 1 JSON matches the spec's table. ✓
- Design § Upsert logic: MERGE SQL in Task 6 matches spec's MERGE (scoped NOT MATCHED BY SOURCE, `removed_at = NULL` on re-match). ✓
- Design § Python changes: Task 6 removes `_get_table_schema`, `_create_table_if_not_exists`, `get_collection_history`, `delete_collection`, `collection_version`. Retains reshaped `save_collection`, `get_latest_collection`, `get_owned_game_ids`. Adds `delete_user_rows` and `get_all_rows_including_removed` for tests and lifecycle inspection (not in original spec but minor — both are read/cleanup utilities). ✓
- Design § Load order (`last_modified` cast): handled in `_prepare_rows` via `pd.to_datetime`. ✓
- Testing § Terraform: Task 2 steps 4, 6, 7 cover plan, `bq show`, no-drift. ✓
- Testing § Python integration: Task 5 covers all 7 numbered scenarios plus the empty-input and duplicate-rows guards from Risks. ✓
- Risks: guards implemented in `_prepare_rows` (Task 6) and asserted in tests (Task 5). ✓

No placeholders or TBDs. Method names and column orderings match between tasks.
