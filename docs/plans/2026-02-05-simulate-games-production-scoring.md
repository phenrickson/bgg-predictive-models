# Switch Production Scoring to simulate_games Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `predict_games` with `simulate_games` as the production scoring method, adding hurdle predictions, BigQuery upload with prediction intervals, and updating the CI/CD pipeline and Dataform layer.

**Architecture:** The `simulate_games` endpoint already runs Bayesian posterior sampling through the model chain (complexity → rating/users_rated → geek_rating). We add hurdle model loading for `predicted_hurdle_prob`, flatten simulation results into the existing `ml_predictions_landing` table schema (medians as point estimates + 90% interval columns), and switch the `score.py` CLI and GitHub Actions workflow to call `/simulate_games` instead of `/predict_games`. The Dataform layer passes through the new interval columns.

**Tech Stack:** Python/FastAPI (scoring service), BigQuery/Terraform (schema), Dataform/SQLX (data warehouse), GitHub Actions (CI/CD)

---

## Task 1: Add interval columns to Terraform BigQuery schema

**Files:**
- Modify: `terraform/bigquery.tf:154-164`

**Step 1: Add interval columns to ml_predictions_landing schema**

Insert 8 new FLOAT NULLABLE columns before the `score_ts` field in the `ml_predictions_landing` table schema:

```hcl
    {
      name = "complexity_lower_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "complexity_upper_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "rating_lower_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "rating_upper_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "users_rated_lower_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "users_rated_upper_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "geek_rating_lower_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "geek_rating_upper_90"
      type = "FLOAT"
      mode = "NULLABLE"
    },
```

Also add `n_samples` (INTEGER NULLABLE) to track how many simulation samples were used.

**Step 2: Commit**

```bash
git add terraform/bigquery.tf
git commit -m "feat: add prediction interval columns to ml_predictions_landing schema"
```

---

## Task 2: Add interval columns to Python BigQuery uploader schema

**Files:**
- Modify: `src/data/bigquery_uploader.py:19-46`

**Step 1: Add interval fields to PREDICTIONS_LANDING_SCHEMA**

Add these `bigquery.SchemaField` entries after the `predicted_geek_rating` field (line 28) and before the model metadata fields:

```python
    bigquery.SchemaField("complexity_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("n_samples", "INTEGER", mode="NULLABLE"),
```

Note: The uploader uses `ALLOW_FIELD_ADDITION` so existing rows will have NULL for these new columns. No migration needed.

**Step 2: Commit**

```bash
git add src/data/bigquery_uploader.py
git commit -m "feat: add prediction interval fields to BigQuery uploader schema"
```

---

## Task 3: Add hurdle model + BigQuery upload to simulate_games endpoint

**Files:**
- Modify: `scoring_service/main.py:111-134` (SimulateGamesRequest/Response models)
- Modify: `scoring_service/main.py:1020-1119` (simulate_games_endpoint)

This is the main task. The simulate endpoint currently:
- Loads 4 models (complexity, rating, users_rated, geek_rating) — no hurdle
- Runs `simulate_batch()` and returns JSON results
- Does NOT upload to BigQuery or GCS

We need to:
1. Add hurdle model loading and `predict_proba` call
2. Add `upload_to_data_warehouse` and `output_path` fields to request model
3. Flatten simulation results into a DataFrame compatible with `ml_predictions_landing`
4. Upload to GCS and BigQuery using the same pattern as `predict_games`

**Step 1: Update SimulateGamesRequest and SimulateGamesResponse**

In `scoring_service/main.py`, update the request model to match `PredictGamesRequest` fields:

```python
class SimulateGamesRequest(BaseModel):
    hurdle_model_name: str
    complexity_model_name: str
    rating_model_name: str
    users_rated_model_name: str
    geek_rating_model_name: str
    hurdle_model_version: Optional[int] = None
    complexity_model_version: Optional[int] = None
    rating_model_version: Optional[int] = None
    users_rated_model_version: Optional[int] = None
    geek_rating_model_version: Optional[int] = None
    game_ids: Optional[List[int]] = None
    start_year: Optional[int] = 2024
    end_year: Optional[int] = 2029
    n_samples: int = 500
    random_state: int = 42
    output_path: Optional[str] = "data/predictions/game_simulations.parquet"
    upload_to_data_warehouse: bool = True
    use_change_detection: bool = False
    max_games: Optional[int] = 50000
```

Update the response model to include upload info:

```python
class SimulateGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    n_samples: int
    games_simulated: int
    results: Optional[List[Dict[str, Any]]] = None
    output_location: Optional[str] = None
    data_warehouse_job_id: Optional[str] = None
    data_warehouse_table: Optional[str] = None
    skipped_reason: Optional[str] = None
```

**Step 2: Update simulate_games_endpoint**

Rewrite the endpoint to:

1. Load all 5 models (add hurdle):
```python
registered_hurdle = get_registered_model("hurdle")
hurdle_pipeline, hurdle_reg = registered_hurdle.load_registered_model(
    request.hurdle_model_name, request.hurdle_model_version
)
```

2. Load game data (same as current, but add change detection support):
```python
if request.game_ids:
    df_pandas = load_game_data(game_ids=request.game_ids)
elif request.use_change_detection:
    df_pandas = load_games_for_main_scoring(
        start_year=request.start_year,
        end_year=request.end_year,
        max_games=request.max_games,
        hurdle_model_version=hurdle_reg["version"],
        complexity_model_version=complexity_reg["version"],
        rating_model_version=rating_reg["version"],
        users_rated_model_version=users_rated_reg["version"],
    )
    if len(df_pandas) == 0:
        return SimulateGamesResponse(
            job_id=job_id, model_details={}, n_samples=request.n_samples,
            games_simulated=0, skipped_reason="no_changes",
        )
else:
    df_pandas = load_game_data(request.start_year, request.end_year)
```

3. Predict hurdle probabilities (before simulation):
```python
predicted_hurdle_prob = predict_hurdle_probabilities(hurdle_pipeline, df_pandas)
```

4. Run simulation (existing code, keep as-is)

5. Flatten simulation results into a DataFrame for upload:
```python
def flatten_simulation_results(
    sim_results: List,
    df_pandas: pd.DataFrame,
    predicted_hurdle_prob: pd.Series,
) -> pd.DataFrame:
    """Flatten SimulationResult objects into a DataFrame for BigQuery upload."""
    rows = []
    for i, r in enumerate(sim_results):
        row = {
            "game_id": r.game_id,
            "name": r.game_name,
            "year_published": df_pandas.iloc[i]["year_published"],
            "predicted_hurdle_prob": float(predicted_hurdle_prob.iloc[i]),
            "predicted_complexity": float(np.median(r.complexity_samples)),
            "predicted_rating": float(np.median(r.rating_samples)),
            "predicted_users_rated": float(np.median(r.users_rated_count_samples)),
            "predicted_geek_rating": float(np.median(r.geek_rating_samples)),
            "complexity_lower_90": float(r.interval(r.complexity_samples, 0.90)[0]),
            "complexity_upper_90": float(r.interval(r.complexity_samples, 0.90)[1]),
            "rating_lower_90": float(r.interval(r.rating_samples, 0.90)[0]),
            "rating_upper_90": float(r.interval(r.rating_samples, 0.90)[1]),
            "users_rated_lower_90": float(r.interval(r.users_rated_count_samples, 0.90)[0]),
            "users_rated_upper_90": float(r.interval(r.users_rated_count_samples, 0.90)[1]),
            "geek_rating_lower_90": float(r.interval(r.geek_rating_samples, 0.90)[0]),
            "geek_rating_upper_90": float(r.interval(r.geek_rating_samples, 0.90)[1]),
        }
        rows.append(row)
    return pd.DataFrame(rows)
```

Note: `predicted_users_rated` and the `users_rated_*_90` interval columns use **count scale** (via `users_rated_count_samples`) not log scale, since that's what the dashboard and humans care about. The existing `predict_games` endpoint already stores users_rated in count scale.

6. Upload to GCS and BigQuery (mirror the `predict_games` pattern from lines 608-684):
```python
if request.game_ids:
    # Return results in response, skip uploads
    results_list = [r.summary() for r in sim_results]
    output_path = None
    data_warehouse_job_id = None
    data_warehouse_table = None
else:
    results_list = None

    # Save locally + upload to GCS
    local_output_path = request.output_path or f"/tmp/{job_id}_simulations.parquet"
    flat_results.to_parquet(local_output_path, index=False)

    storage_client = authenticator.get_storage_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    gcs_output_path = f"{ENVIRONMENT_PREFIX}/predictions/{job_id}_simulations.parquet"
    blob = bucket.blob(gcs_output_path)
    blob.upload_from_filename(local_output_path)
    output_path = f"gs://{BUCKET_NAME}/{gcs_output_path}"

    # Upload to BigQuery
    data_warehouse_job_id = None
    data_warehouse_table = None
    if request.upload_to_data_warehouse:
        model_versions = {
            "hurdle": hurdle_reg["name"],
            "hurdle_version": hurdle_reg["version"],
            "hurdle_experiment": hurdle_reg["original_experiment"]["name"],
            "complexity": complexity_reg["name"],
            "complexity_version": complexity_reg["version"],
            "complexity_experiment": complexity_reg["original_experiment"]["name"],
            "rating": rating_reg["name"],
            "rating_version": rating_reg["version"],
            "rating_experiment": rating_reg["original_experiment"]["name"],
            "users_rated": users_rated_reg["name"],
            "users_rated_version": users_rated_reg["version"],
            "users_rated_experiment": users_rated_reg["original_experiment"]["name"],
            "geek_rating": geek_rating_reg["name"],
            "geek_rating_version": geek_rating_reg["version"],
            "geek_rating_experiment": geek_rating_reg["original_experiment"]["name"],
        }

        dw_uploader = DataWarehousePredictionUploader()
        data_warehouse_job_id = dw_uploader.upload_predictions(
            flat_results, job_id, model_versions=model_versions
        )
        data_warehouse_table = dw_uploader.table_id
```

7. Add hurdle to model_details and return response:
```python
model_details = {
    "hurdle": {
        "name": hurdle_reg["name"],
        "version": hurdle_reg["version"],
        "experiment": hurdle_reg["original_experiment"]["name"],
    },
    "complexity": { ... },  # existing
    "rating": { ... },      # existing
    "users_rated": { ... }, # existing
    "geek_rating": { ... }, # existing
}
```

**Step 3: Commit**

```bash
git add scoring_service/main.py
git commit -m "feat: add hurdle model and BigQuery upload to simulate_games endpoint"
```

---

## Task 4: Switch score.py CLI from predict_games to simulate_games

**Files:**
- Modify: `scoring_service/score.py`

**Step 1: Update submit_scoring_request to call /simulate_games**

Change the payload structure and endpoint URL:

1. Change URL from `/predict_games` to `/simulate_games` (line 109):
```python
response = requests.post(
    f"{service_url}/simulate_games",
    json=payload,
    timeout=1800,  # 30-minute timeout (simulations take longer)
)
```

2. Update the payload to match `SimulateGamesRequest`:
- Remove `prior_rating` and `prior_weight` (not used by simulation)
- Add `n_samples` parameter
- Add `use_change_detection` and `max_games`
- Keep all model name fields

3. Add `--n-samples` argument to the CLI parser:
```python
parser.add_argument(
    "--n-samples",
    type=int,
    default=500,
    help="Number of posterior samples for simulation (default: 500)",
)
```

4. Remove `--prior-rating` and `--prior-weight` arguments (simulation doesn't use Bayesian averaging)

5. Update the function signature:
```python
def submit_scoring_request(
    service_url: str,
    start_year: int,
    end_year: int,
    hurdle_model: Optional[str] = None,
    complexity_model: Optional[str] = None,
    rating_model: Optional[str] = None,
    users_rated_model: Optional[str] = None,
    geek_rating_model: Optional[str] = None,
    output_path: Optional[str] = None,
    n_samples: int = 500,
    upload_to_data_warehouse: bool = True,
    use_change_detection: bool = False,
) -> dict:
```

**Step 2: Commit**

```bash
git add scoring_service/score.py
git commit -m "feat: switch score.py CLI from predict_games to simulate_games"
```

---

## Task 5: Update GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/run-scoring-service.yml`

**Step 1: Add n_samples input and update the scoring command**

Add `n_samples` to `workflow_dispatch.inputs`:
```yaml
      n_samples:
        description: 'Number of posterior samples for simulation'
        required: false
        default: '500'
```

Update the "Trigger Predictions" step command to remove `--prior-rating`/`--prior-weight` references and add `--n-samples`:
```yaml
    - name: Trigger Simulations using score.py
      run: |
        USE_CHANGE_DETECTION="${{ github.event.inputs.use_change_detection || 'true' }}"

        CHANGE_DETECTION_FLAG=""
        if [ "$USE_CHANGE_DETECTION" = "true" ]; then
          CHANGE_DETECTION_FLAG="--use-change-detection"
        fi

        uv run -m scoring_service.score \
          --service-url "${{ steps.get-url.outputs.service_url }}" \
          --start-year ${{ github.event.inputs.start_year || 2024 }} \
          --end-year ${{ github.event.inputs.end_year || 2029 }} \
          --n-samples ${{ github.event.inputs.n_samples || 500 }} \
          --upload-to-bigquery \
          $CHANGE_DETECTION_FLAG \
          ${{ github.event.inputs.output_path != '' && format('--output-path "{0}"', github.event.inputs.output_path) || '' }}
```

**Step 2: Commit**

```bash
git add .github/workflows/run-scoring-service.yml
git commit -m "feat: update scoring workflow to use simulate_games with n_samples"
```

---

## Task 6: Add interval columns to Dataform bgg_predictions.sqlx

**Files:**
- Modify: `/Users/phenrickson/Documents/projects/bgg-data-warehouse/definitions/bgg_predictions.sqlx`

**Step 1: Add interval columns to the inner SELECT**

Add the 8 interval columns after `predicted_geek_rating` (line 22) and before the model metadata columns:

```sql
    complexity_lower_90,
    complexity_upper_90,
    rating_lower_90,
    rating_upper_90,
    users_rated_lower_90,
    users_rated_upper_90,
    geek_rating_lower_90,
    geek_rating_upper_90,
    n_samples,
```

The full inner SELECT becomes:
```sql
  SELECT
    job_id,
    game_id,
    name,
    year_published,
    predicted_hurdle_prob,
    predicted_complexity,
    predicted_rating,
    predicted_users_rated,
    predicted_geek_rating,
    complexity_lower_90,
    complexity_upper_90,
    rating_lower_90,
    rating_upper_90,
    users_rated_lower_90,
    users_rated_upper_90,
    geek_rating_lower_90,
    geek_rating_upper_90,
    n_samples,
    geek_rating_model_name,
    geek_rating_model_version,
    geek_rating_experiment,
    hurdle_model_name,
    hurdle_model_version,
    hurdle_experiment,
    complexity_model_name,
    complexity_model_version,
    complexity_experiment,
    rating_model_name,
    rating_model_version,
    rating_experiment,
    users_rated_model_name,
    users_rated_model_version,
    users_rated_experiment,
    score_ts,
    source_environment,
    ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC, job_id DESC) as rn
  FROM ${ref("bgg-predictive-models", "raw", "ml_predictions_landing")}
```

Old rows (from `predict_games`) will have NULL for the interval columns, which is fine.

**Step 2: Commit (in bgg-data-warehouse repo)**

```bash
cd /Users/phenrickson/Documents/projects/bgg-data-warehouse
git add definitions/bgg_predictions.sqlx
git commit -m "feat: add prediction interval columns from simulate_games"
```

---

## Task 7: Add geek_rating to Dataform deployed_models.sqlx

**Files:**
- Modify: `/Users/phenrickson/Documents/projects/bgg-data-warehouse/definitions/deployed_models.sqlx`

**Step 1: Add geek_rating UNION ALL block to prediction_models CTE**

After the `users_rated` block (line 70), add:

```sql
  UNION ALL

  SELECT
    'prediction' AS model_category,
    'geek_rating' AS model_type,
    geek_rating_model_name AS model_name,
    geek_rating_model_version AS model_version,
    geek_rating_experiment AS experiment,
    CAST(NULL AS STRING) AS algorithm,
    CAST(NULL AS INT64) AS embedding_dim,
    CAST(NULL AS STRING) AS document_method,
    COUNT(DISTINCT game_id) AS games_count,
    MAX(score_ts) AS last_updated
  FROM ${ref("bgg-predictive-models", "raw", "ml_predictions_landing")}
  GROUP BY geek_rating_model_name, geek_rating_model_version, geek_rating_experiment
```

**Step 2: Commit (in bgg-data-warehouse repo)**

```bash
cd /Users/phenrickson/Documents/projects/bgg-data-warehouse
git add definitions/deployed_models.sqlx
git commit -m "feat: add geek_rating model to deployed_models monitoring view"
```

---

## Verification

### Local Testing

1. **Rebuild Docker image and start:**
```bash
cd /Users/phenrickson/Documents/projects/bgg-predictive-models
make docker-scoring && make start-scoring
```

2. **Test simulate_games with specific game_ids (quick smoke test):**
```bash
curl -s -X POST http://localhost:8087/simulate_games \
  -H 'Content-Type: application/json' \
  -d '{
    "hurdle_model_name":"hurdle-v2026",
    "complexity_model_name":"complexity-v2026",
    "rating_model_name":"rating-v2026",
    "users_rated_model_name":"users_rated-v2026",
    "geek_rating_model_name":"geek_rating-v2026",
    "game_ids":[174430,167791,224517],
    "n_samples":100,
    "upload_to_data_warehouse":false
  }' | python3 -m json.tool
```

Verify response includes:
- `predicted_hurdle_prob` in results (was missing before)
- All interval fields present
- `model_details` includes `hurdle` key

3. **Test batch simulation with upload (2025 only, small batch):**
```bash
curl -s -X POST http://localhost:8087/simulate_games \
  -H 'Content-Type: application/json' \
  -d '{
    "hurdle_model_name":"hurdle-v2026",
    "complexity_model_name":"complexity-v2026",
    "rating_model_name":"rating-v2026",
    "users_rated_model_name":"users_rated-v2026",
    "geek_rating_model_name":"geek_rating-v2026",
    "start_year":2025,
    "end_year":2026,
    "n_samples":100,
    "upload_to_data_warehouse":false
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Games: {d[\"games_simulated\"]}')"
```

4. **Test score.py CLI locally:**
```bash
uv run -m scoring_service.score \
  --service-url http://localhost:8087 \
  --start-year 2025 --end-year 2026 \
  --n-samples 100 \
  --no-upload
```

### Deployment Checklist

1. Push scoring service changes → GitHub Actions builds Docker image and deploys to Cloud Run
2. Run `ENVIRONMENT=prod make register` to register models to prod GCS
3. Trigger scoring workflow (manual dispatch or wait for `dataform_complexity_ready` event)
4. Verify in BigQuery: `SELECT * FROM bgg-predictive-models.raw.ml_predictions_landing WHERE complexity_lower_90 IS NOT NULL LIMIT 10`
5. Run Dataform in bgg-data-warehouse to materialize updated `bgg_predictions`
6. Apply Terraform to update the table schema (or rely on `ALLOW_FIELD_ADDITION`)
