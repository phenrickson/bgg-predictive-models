# Collection Storage Design

**Date:** 2026-04-20
**Status:** Approved (pending user review of this document)
**Owner:** bgg-predictive-models

## Goal

Make "given a BGG username, collect → store → model" a single reliable flow, with
collection storage managed by Terraform and refreshed via idempotent upserts.

## Motivation

`src/collection/collection_storage.py` today creates its target table at runtime
from `config/bigquery.yaml` and appends a full snapshot on every pull, stamping
each row with `loaded_at` and `collection_version`. This is the wrong shape for
the actual workflow: the modeling pipeline wants the *current* state of a user's
collection and expects to refresh that state frequently. Append-on-every-pull
grows the table without bound, forces a `MAX(loaded_at)` subquery on every read,
and keeps schema out of version control.

The fix: Terraform owns the schema, the writer upserts into a PK-keyed table,
and "current collection" is a trivial `SELECT` filtered by `removed_at IS NULL`.

## Scope

**In scope**

- New `collections` BigQuery dataset owned by `bgg-predictive-models/terraform/`.
- New `collections.user_collections` table with primary key `(username, game_id)`.
- Schema defined in a Terraform-managed JSON file; table declared in
  `terraform/bigquery.tf`.
- Rewritten `CollectionStorage.save_collection` using a single `MERGE`, with a
  pre-MERGE filter to `subtype == "boardgame"`.
- `BGGCollectionLoader` request params extended with
  `excludesubtype=boardgameexpansion` so expansions are excluded at the source.
- Simplified `get_latest_collection` and `get_owned_game_ids`.
- Integration test against a real dev BigQuery table covering insert / update /
  soft-delete / re-insert.

**Out of scope**

- Migrating rows from the existing `analytics.collections` table (greenfield —
  no preservation or backwards-compatibility shims).
- A separate history/audit table.
- Storage Write API CDC streaming.
- Dataform-derived views on top of `user_collections`.

## Design

### Table: `collections.user_collections`

One row per `(username, game_id)`. Columns mirror what `BGGCollectionLoader`
returns, plus three lifecycle timestamps.

| Column              | Type      | Mode     | Notes |
|---------------------|-----------|----------|-------|
| `username`          | STRING    | REQUIRED | PK |
| `game_id`           | INTEGER   | REQUIRED | PK |
| `game_name`         | STRING    | NULLABLE | As loader emits |
| `subtype`           | STRING    | NULLABLE | Always `"boardgame"` — filter applied at write |
| `collection_id`     | INTEGER   | NULLABLE | BGG's `collid` |
| `owned`             | BOOL      | NULLABLE | |
| `previously_owned`  | BOOL      | NULLABLE | As loader emits (not `prev_owned`) |
| `for_trade`         | BOOL      | NULLABLE | |
| `want`              | BOOL      | NULLABLE | |
| `want_to_play`      | BOOL      | NULLABLE | |
| `want_to_buy`       | BOOL      | NULLABLE | |
| `wishlist`          | BOOL      | NULLABLE | |
| `wishlist_priority` | INTEGER   | NULLABLE | 1–5 |
| `preordered`        | BOOL      | NULLABLE | |
| `user_rating`       | FLOAT     | NULLABLE | BGG 1–10 |
| `user_comment`      | STRING    | NULLABLE | As loader emits |
| `last_modified`     | TIMESTAMP | NULLABLE | From BGG (cast from string) |
| `first_seen_at`     | TIMESTAMP | REQUIRED | Set on insert, never updated |
| `updated_at`        | TIMESTAMP | REQUIRED | Set on insert or update |
| `removed_at`        | TIMESTAMP | NULLABLE | Soft-delete marker |

**Table-level configuration**

- Primary key: `(username, game_id)`, declared via Terraform
  `table_constraints { primary_key { columns = ["username", "game_id"] } }`.
  BigQuery's PK is `NOT ENFORCED` — integrity is the writer's responsibility —
  but the optimizer uses the constraint for join elimination and MERGE planning.
- Clustering: `["username", "game_id"]`. Matches the PK; keeps per-user MERGE
  scans and `SELECT ... WHERE username = X` reads cheap.
- Partitioning: none. The table is upserted in place; there is no monotonic
  date column that would make a useful partition key.
- `deletion_protection = var.environment == "prod"`.

### Dataset

New `google_bigquery_dataset.collections` in
`bgg-predictive-models/terraform/bigquery.tf`, alongside the existing `raw`
dataset. Labels: `environment`, `managed_by = "terraform"`.

### Terraform layout

```
bgg-predictive-models/terraform/
├── bigquery.tf                          # add dataset + table resource
└── schemas/
    └── user_collections.json            # new
```

`bigquery.tf` loads the schema via `file("${path.module}/schemas/user_collections.json")`,
matching the pattern established in `bgg-data-warehouse/terraform/bigquery.tf`.

### Upsert logic

One `MERGE` per pull. The writer stages the pulled rows, then issues a single
statement that inserts new rows, updates changed rows, and soft-deletes rows
that are no longer present in the pull.

```sql
MERGE `proj.collections.user_collections` T
USING (
  SELECT
    @username AS username,
    game_id, game_name, subtype, collection_id,
    owned, previously_owned, for_trade, want, want_to_play, want_to_buy,
    wishlist, wishlist_priority, preordered,
    user_rating, user_comment, last_modified,
    CURRENT_TIMESTAMP() AS now
  FROM UNNEST(@rows)
) S
ON T.username = S.username AND T.game_id = S.game_id

WHEN MATCHED THEN UPDATE SET
  game_name        = S.game_name,
  subtype          = S.subtype,
  collection_id    = S.collection_id,
  owned            = S.owned,
  previously_owned = S.previously_owned,
  for_trade        = S.for_trade,
  want             = S.want,
  want_to_play     = S.want_to_play,
  want_to_buy      = S.want_to_buy,
  wishlist         = S.wishlist,
  wishlist_priority= S.wishlist_priority,
  preordered       = S.preordered,
  user_rating      = S.user_rating,
  user_comment     = S.user_comment,
  last_modified    = S.last_modified,
  updated_at       = S.now,
  removed_at       = NULL

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
  S.now, S.now, NULL
)

WHEN NOT MATCHED BY SOURCE
  AND T.username = @username
  AND T.removed_at IS NULL
THEN UPDATE SET
  removed_at = CURRENT_TIMESTAMP(),
  updated_at = CURRENT_TIMESTAMP()
```

**Critical invariants**

1. `NOT MATCHED BY SOURCE` is scoped to `T.username = @username`. Without this,
   every pull would soft-delete every other user's rows.
2. `removed_at = NULL` on re-match means a game that disappears and later
   reappears un-soft-deletes cleanly.
3. The writer is a pure function of `(username, rows)`. Re-running the same
   pull moves `updated_at` but changes no data — the operation is idempotent.

### Python changes

`src/collection/collection_storage.py` is rewritten (greenfield, per project
convention — no preservation of the old surface).

Removed:
- `_get_table_schema`, `_create_table_if_not_exists`, `config/bigquery.yaml`
  schema plumbing. Terraform owns schema.
- `get_collection_history` — no longer applicable; there is no version history.
- `delete_collection` — no longer applicable; there are no versions to delete.
- `collection_version` parameter on `save_collection`.

Retained / reshaped:
- `__init__(environment)` — reads project/dataset/location from `config.yaml`
  only. No YAML schema loading.
- `save_collection(username, collection_df: pl.DataFrame) -> None` — filters
  the DataFrame to `subtype == "boardgame"`, asserts the result is non-empty
  and has no duplicate `(username, game_id)` rows, then runs the MERGE. Empty
  input is rejected with a clear error: we cannot distinguish "user truly has
  no games" from "pull failed upstream," and silently soft-deleting every row
  would be destructive. If a caller really needs to clear a user, they do it
  explicitly (not in scope for this spec).
- `get_latest_collection(username) -> pl.DataFrame | None` —
  `SELECT * FROM collections.user_collections WHERE username = @username
   AND removed_at IS NULL`. Returns `None` if no rows.
- `get_owned_game_ids(username) -> list[int] | None` — same semantics as today.

Config change: `get_bigquery_config()` gains a `collections_dataset` field
(default `"collections"`) so Python reads the same name Terraform writes.

### Load order

The `last_modified` column comes from the BGG API as a string
(e.g. `"2025-03-14 18:22:01"`). The writer must cast to TIMESTAMP before the
MERGE, either via `pd.to_datetime` in the staging DataFrame (as today) or via
`PARSE_TIMESTAMP` in the USING clause.

## Testing

### Terraform

1. `terraform plan` shows dataset + table as new resources and reports no drift
   on a second run.
2. `terraform apply` in dev creates the dataset and table.
3. `bq show --format=prettyjson <project>:collections.user_collections` returns
   the expected schema, clustering fields, and declared primary key.

### Python integration test

Against the real dev `collections.user_collections` table, no mocks:

1. **Initial load:** pull user A with 3 games → assert 3 rows, all with
   `first_seen_at = updated_at`, `removed_at IS NULL`.
2. **Idempotent re-pull:** re-run same pull → assert 3 rows, `first_seen_at`
   unchanged, `updated_at` advanced, no `removed_at`.
3. **Modified row:** change `user_rating` on game #1 → assert game #1's
   `user_rating` reflects the new value. Note that *every* matched row's
   `updated_at` advances on every pull (MERGE `WHEN MATCHED` always fires);
   the test asserts data correctness, not that unchanged rows kept their old
   `updated_at`.
4. **New row:** add game #4 → assert 4 rows, game #4 has `first_seen_at = now`.
5. **Removed row:** drop game #2 from the pull → assert game #2 has
   `removed_at IS NOT NULL`, other rows untouched beyond `updated_at`.
6. **Re-added row:** put game #2 back in a later pull → assert `removed_at IS NULL`
   again, `first_seen_at` still the original value.
7. **Cross-user isolation:** run steps 1–6 for user A, then pull user B →
   assert user A's rows are untouched.

Fixture cleanup: drop user A/B rows at test teardown via
`DELETE FROM collections.user_collections WHERE username IN (...)`.

## Risks

- **Declared PK is not enforced.** If the writer ever produces duplicate
  `(username, game_id)` rows in a single source, the MERGE will fail with
  "UPDATE/MERGE must match at most one source row for each target row."
  Mitigation: the loader already returns one row per BGG item; add a dedup
  assertion in `save_collection` before the MERGE as a defensive guard.
- **`NOT MATCHED BY SOURCE` on empty input.** A MERGE with an empty staged
  source will soft-delete *every* row for that user. `save_collection` must
  reject an empty DataFrame and surface the error rather than silently wiping
  the user's collection.
- **Schema drift between Terraform and Python.** The loader produces columns;
  the MERGE references columns; Terraform defines columns. If any of the three
  falls out of sync, writes fail. Mitigation: the integration test catches this
  on every run.

## Open questions

None at design time. Will reassess during implementation.
