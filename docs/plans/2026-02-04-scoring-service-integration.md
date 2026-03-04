# Scoring Service Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate refactored Bayesian outcome models with the scoring service, replacing the broken geek_rating calculation with the direct mode pipeline, and adding a new `/simulate_games` endpoint for uncertainty-aware batch predictions.

**Architecture:** The scoring service (`scoring_service/main.py`) is a FastAPI app that loads registered sklearn pipelines from GCS and runs predictions. We add geek_rating as a registered model type (direct mode pipeline), fix the broken `calculate_geek_rating` import, and add a new `/simulate_games` endpoint that wraps the existing `simulate_batch()` function from `src/models/outcomes/simulation.py`. Simulation results (point estimates + credible intervals) persist to BigQuery and GCS alongside existing point predictions.

**Tech Stack:** Python 3.12, FastAPI, scikit-learn (BayesianRidge/ARDRegression), numpy, pandas, polars, Google Cloud Storage, BigQuery, Docker

---

## Task 1: Add `geek_rating` to `register_model.py` choices

**Files:**
- Modify: `scoring_service/register_model.py:128-130` (argparse choices)

**Step 1: Add `geek_rating` to the `--model-type` choices**

In `scoring_service/register_model.py`, the argparse `choices` parameter currently restricts model types to `["hurdle", "rating", "complexity", "users_rated"]`. Add `"geek_rating"`.

```python
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["hurdle", "rating", "complexity", "users_rated", "geek_rating"],
        help="Type of model to register",
    )
```

**Step 2: Verify registration works**

Run:
```bash
uv run -m scoring_service.register_model --model-type geek_rating --experiment ard-geek_rating --name geek_rating-v2026 --description "Production (v2026) direct model for predicting geek rating" --dry-run
```

Note: If `--dry-run` is not supported, this step verifies the argument parsing accepts `geek_rating`. Full registration requires trained model artifacts.

**Step 3: Commit**

```bash
git add scoring_service/register_model.py
git commit -m "feat: add geek_rating to register_model.py model type choices"
```

---

## Task 2: Add `geek_rating` to `register.py` and config

**Files:**
- Modify: `register.py`
- Modify: `config.yaml`

**Step 1: Add geek_rating to `config.yaml` scoring models**

In `config.yaml`, under `scoring.models`, add the geek_rating registered model name:

```yaml
scoring:
  models:
    hurdle: hurdle-v2026
    complexity: complexity-v2026
    rating: rating-v2026
    users_rated: users_rated-v2026
    geek_rating: geek_rating-v2026
    embeddings: embeddings-v2026
```

**Step 2: Add geek_rating to `load_registration_config()` in `register.py`**

Update the `experiments` dict in `load_registration_config()` to include geek_rating:

```python
def load_registration_config() -> Dict[str, Any]:
    """Load configuration for registration from config.yaml"""
    from src.utils.config import load_config

    config = load_config()

    return {
        "current_year": config.years.current,
        "experiments": {
            "hurdle": config.models["hurdle"].experiment_name,
            "complexity": config.models["complexity"].experiment_name,
            "rating": config.models["rating"].experiment_name,
            "users_rated": config.models["users_rated"].experiment_name,
            "geek_rating": config.models["geek_rating"].experiment_name,
        },
    }
```

**Step 3: Add `register_geek_rating()` function to `register.py`**

Add after `register_hurdle()`:

```python
def register_geek_rating(config: Dict[str, Any]) -> None:
    """Register geek_rating model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "geek_rating",
        "--experiment",
        config["experiments"]["geek_rating"],
        "--name",
        f"geek_rating-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) direct model for predicting geek rating",
    ]
    run_command(cmd, "Registering geek_rating model")
```

**Step 4: Call `register_geek_rating()` in `main()`**

Add after the `register_hurdle(config)` call in `main()`:

```python
        # Register geek_rating model
        logger.info(
            f"\n📝 Registering geek_rating model ({config['experiments']['geek_rating']})"
        )
        register_geek_rating(config)
```

**Step 5: Commit**

```bash
git add register.py config.yaml
git commit -m "feat: add geek_rating to registration pipeline and config"
```

---

## Task 3: Add `register_geek_rating` Makefile target

**Files:**
- Modify: `Makefile`

**Step 1: Add Makefile variables and targets**

Add `GEEK_RATING_CANDIDATE` variable near the other candidate variables (around line 216):

```makefile
GEEK_RATING_CANDIDATE ?= ard-geek_rating
```

Add `register_geek_rating` to the `.PHONY` list and `register` dependency:

```makefile
.PHONY: register_complexity register_rating register_users_rated register_hurdle register_geek_rating register_embeddings register_text_embeddings register
register: register_complexity register_rating register_users_rated register_hurdle register_geek_rating register_embeddings register_text_embeddings
```

Add the target itself after `register_hurdle`:

```makefile
register_geek_rating:
	uv run -m scoring_service.register_model \
	--model-type geek_rating \
	--experiment $(GEEK_RATING_CANDIDATE) \
	--name geek_rating-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) direct model for predicting geek rating"
```

**Step 2: Add geek-rating-model flag to scoring-service targets**

Update `scoring-service` and `scoring-service-upload` targets to include the geek rating model:

```makefile
scoring-service:
	uv run -m scoring_service.score \
    --service-url http://localhost:8080 \
    --start-year $(SCORE_START_YEAR) \
    --end-year $(SCORE_END_YEAR) \
    --hurdle-model hurdle-v$(CURRENT_YEAR) \
    --complexity-model complexity-v$(CURRENT_YEAR) \
    --rating-model rating-v$(CURRENT_YEAR) \
    --users-rated-model users_rated-v$(CURRENT_YEAR) \
    --geek-rating-model geek_rating-v$(CURRENT_YEAR) \
    --download
```

(Same for `scoring-service-upload`.)

**Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add geek_rating registration and scoring targets to Makefile"
```

---

## Task 4: Fix `/predict_games` - Replace broken geek_rating with direct model

**Files:**
- Modify: `scoring_service/main.py`

This is the largest single task. It replaces the broken `from src.models.geek_rating import calculate_geek_rating` import with loading and using a registered geek_rating direct mode pipeline.

**Step 1: Fix imports**

Remove:
```python
from src.models.geek_rating import calculate_geek_rating  # noqa: E402
```

No replacement import needed - geek_rating is loaded as a registered model pipeline, same as the others.

**Step 2: Add `geek_rating_model_name` to `PredictGamesRequest`**

```python
class PredictGamesRequest(BaseModel):
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
    start_year: Optional[int] = 2024
    end_year: Optional[int] = 2029
    prior_rating: float = 5.5
    prior_weight: float = 2000
    output_path: Optional[str] = "data/predictions/game_predictions.parquet"
    upload_to_data_warehouse: bool = True
    game_ids: Optional[List[int]] = None
    use_change_detection: bool = False
    max_games: Optional[int] = 50000
```

**Step 3: Load geek_rating pipeline in `predict_games_endpoint`**

After the existing model loading block (lines 481-514), add:

```python
        registered_geek_rating_model = RegisteredModel(
            "geek_rating", BUCKET_NAME, project_id=GCP_PROJECT_ID
        )
        geek_rating_pipeline, geek_rating_registration = (
            registered_geek_rating_model.load_registered_model(
                request.geek_rating_model_name, request.geek_rating_model_version
            )
        )
```

**Step 4: Modify `predict_game_characteristics` to keep log-scale users_rated**

The direct geek_rating model expects `predicted_users_rated_log` as a feature. Update the function to return it:

```python
def predict_game_characteristics(
    features: pd.DataFrame,
    complexity_model: Any,
    rating_model: Any,
    users_rated_model: Any,
    likely_games_mask: pd.Series,
) -> pd.DataFrame:
    """Predict game complexity, rating, and users rated for all games."""
    results = pd.DataFrame(index=features.index)

    # Predict complexity for all games
    results["predicted_complexity"] = complexity_model.predict(features)

    # Add predicted complexity to features
    features_with_complexity = features.copy()
    features_with_complexity["predicted_complexity"] = results["predicted_complexity"]

    # Predict rating and users rated for all games
    results["predicted_rating"] = rating_model.predict(features_with_complexity)

    # Keep log-scale prediction for geek_rating direct model
    predicted_users_rated_log = users_rated_model.predict(features_with_complexity)
    results["predicted_users_rated_log"] = predicted_users_rated_log

    # Transform to count scale for display/output
    results["predicted_users_rated"] = np.maximum(
        np.round(np.expm1(predicted_users_rated_log) / 50) * 50,
        25,
    )

    return results
```

**Step 5: Replace `calculate_geek_rating` call with pipeline predict**

Replace lines 590-595 (the `calculate_geek_rating` call) with:

```python
        # Prepare features for geek_rating direct model
        geek_rating_features = df_pandas.copy()
        geek_rating_features["predicted_complexity"] = characteristics["predicted_complexity"]
        geek_rating_features["predicted_rating"] = characteristics["predicted_rating"]
        geek_rating_features["predicted_users_rated_log"] = characteristics["predicted_users_rated_log"]

        # Predict geek rating using direct model
        results["predicted_geek_rating"] = geek_rating_pipeline.predict(geek_rating_features)
```

**Step 6: Add geek_rating to model_details and experiment tracking**

Add geek_rating registration info to the experiment columns:

```python
        results["geek_rating_experiment"] = geek_rating_registration[
            "original_experiment"
        ]["name"]
```

And to the model_details dict:

```python
            "geek_rating_model": {
                "name": geek_rating_registration["name"],
                "version": geek_rating_registration["version"],
                "experiment": geek_rating_registration["original_experiment"]["name"],
            },
```

**Step 7: Update data warehouse model_versions**

In the data warehouse upload section, add geek_rating to `model_versions`:

```python
                    model_versions = {
                        # ... existing entries ...
                        "geek_rating": geek_rating_registration["name"],
                        "geek_rating_version": geek_rating_registration["version"],
                        "geek_rating_experiment": geek_rating_registration["original_experiment"]["name"],
                    }
```

**Step 8: Add `"geek_rating"` to `/models` endpoint**

Update the `model_types` list in `list_available_models()`:

```python
    model_types = ["hurdle", "rating", "complexity", "users_rated", "geek_rating"]
```

**Step 9: Test manually**

Start the scoring service and verify `/health` works without the broken import:

```bash
cd /Users/phenrickson/Documents/projects/bgg-predictive-models
uv run uvicorn scoring_service.main:app --host 0.0.0.0 --port 8080
```

Then test:
```bash
curl http://localhost:8080/health
curl http://localhost:8080/models
```

**Step 10: Commit**

```bash
git add scoring_service/main.py
git commit -m "feat: replace broken geek_rating import with registered direct model pipeline"
```

---

## Task 5: Update `scoring_service/score.py` CLI client

**Files:**
- Modify: `scoring_service/score.py`

**Step 1: Add `--geek-rating-model` argument**

In the `main()` function, after the `--users-rated-model` argument:

```python
    parser.add_argument(
        "--geek-rating-model",
        help="Override geek rating model name from config",
    )
```

**Step 2: Update `submit_scoring_request` signature and payload**

Add `geek_rating_model` parameter:

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
    prior_rating: Optional[float] = None,
    prior_weight: Optional[float] = None,
    upload_to_data_warehouse: bool = True,
    use_change_detection: bool = False,
) -> dict:
```

Add to the payload in both config and fallback branches:

```python
            "geek_rating_model_name": geek_rating_model or model_config.get("geek_rating"),
```

**Step 3: Pass the new argument in `main()`**

```python
        response = submit_scoring_request(
            # ... existing args ...
            geek_rating_model=args.geek_rating_model,
        )
```

**Step 4: Commit**

```bash
git add scoring_service/score.py
git commit -m "feat: add geek-rating-model argument to scoring CLI client"
```

---

## Task 6: Add `/simulate_games` endpoint - Request/Response schemas

**Files:**
- Modify: `scoring_service/main.py`

**Step 1: Add simulation import**

At the top of `main.py`, alongside existing `src` imports:

```python
from src.models.outcomes.simulation import (  # noqa: E402
    simulate_batch,
    precompute_cholesky,
    SimulationResult,
)
```

**Step 2: Add request schema**

```python
class SimulateGamesRequest(BaseModel):
    complexity_model_name: str
    rating_model_name: str
    users_rated_model_name: str
    geek_rating_model_name: str
    complexity_model_version: Optional[int] = None
    rating_model_version: Optional[int] = None
    users_rated_model_version: Optional[int] = None
    geek_rating_model_version: Optional[int] = None
    n_samples: int = 500
    prior_rating: float = 5.5
    prior_weight: float = 2000
    random_state: int = 42
    game_ids: Optional[List[int]] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    max_games: Optional[int] = 50000
    upload_to_data_warehouse: bool = True
```

**Step 3: Add response schema**

```python
class SimulateGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    simulation_parameters: Dict[str, Any]
    games_simulated: int
    output_location: Optional[str] = None
    data_warehouse_table: Optional[str] = None
    data_warehouse_job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None
```

**Step 4: Commit**

```bash
git add scoring_service/main.py
git commit -m "feat: add SimulateGamesRequest/Response schemas"
```

---

## Task 7: Add `/simulate_games` endpoint - Core logic

**Files:**
- Modify: `scoring_service/main.py`

**Step 1: Add helper to build simulation predictions DataFrame**

```python
def build_simulation_predictions_df(
    results: List[SimulationResult],
) -> pd.DataFrame:
    """Build a DataFrame from SimulationResult list with point estimates and intervals."""
    predictions_data = []
    for r in results:
        s = r.summary()
        row = {
            "game_id": r.game_id,
            "name": r.game_name,
            # Point predictions
            "predicted_complexity": r.complexity_point,
            "predicted_rating": r.rating_point,
            "predicted_users_rated": float(max(np.expm1(r.users_rated_point), 25)),
            "predicted_geek_rating": r.geek_rating_point,
        }
        # Add simulation summaries per outcome
        for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
            row[f"{outcome}_median"] = s[outcome]["median"]
            row[f"{outcome}_std"] = s[outcome]["std"]
            row[f"{outcome}_lower_90"] = s[outcome]["interval_90"][0]
            row[f"{outcome}_upper_90"] = s[outcome]["interval_90"][1]
            row[f"{outcome}_lower_50"] = s[outcome]["interval_50"][0]
            row[f"{outcome}_upper_50"] = s[outcome]["interval_50"][1]
        predictions_data.append(row)

    return pd.DataFrame(predictions_data)
```

**Step 2: Add the endpoint**

```python
@app.post("/simulate_games", response_model=SimulateGamesResponse)
async def simulate_games_endpoint(request: SimulateGamesRequest):
    """
    Simulate game predictions with uncertainty estimation using Bayesian posteriors.

    Runs the full simulation chain:
    1. Sample complexity from posterior
    2. Sample rating/users_rated conditional on complexity samples
    3. Sample geek_rating from direct model posterior

    Returns point estimates, medians, credible intervals (50% and 90%) per game per outcome.
    """
    try:
        job_id = str(uuid.uuid4())

        # Load registered model pipelines
        registered_complexity = RegisteredModel("complexity", BUCKET_NAME, project_id=GCP_PROJECT_ID)
        registered_rating = RegisteredModel("rating", BUCKET_NAME, project_id=GCP_PROJECT_ID)
        registered_users_rated = RegisteredModel("users_rated", BUCKET_NAME, project_id=GCP_PROJECT_ID)
        registered_geek_rating = RegisteredModel("geek_rating", BUCKET_NAME, project_id=GCP_PROJECT_ID)

        complexity_pipeline, complexity_registration = registered_complexity.load_registered_model(
            request.complexity_model_name, request.complexity_model_version
        )
        rating_pipeline, rating_registration = registered_rating.load_registered_model(
            request.rating_model_name, request.rating_model_version
        )
        users_rated_pipeline, users_rated_registration = registered_users_rated.load_registered_model(
            request.users_rated_model_name, request.users_rated_model_version
        )
        geek_rating_pipeline, geek_rating_registration = registered_geek_rating.load_registered_model(
            request.geek_rating_model_name, request.geek_rating_model_version
        )

        # Validate all pipelines support Bayesian simulation
        for name, pipeline in [
            ("complexity", complexity_pipeline),
            ("rating", rating_pipeline),
            ("users_rated", users_rated_pipeline),
            ("geek_rating", geek_rating_pipeline),
        ]:
            model = pipeline.named_steps.get("model")
            if not (hasattr(model, "coef_") and hasattr(model, "sigma_")):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} model ({type(model).__name__}) does not support "
                    f"posterior sampling. Requires Bayesian model with coef_ and sigma_.",
                )

        # Pre-compute Cholesky decompositions
        logger.info("Pre-computing Cholesky decompositions...")
        cholesky_cache = precompute_cholesky(
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Load game data
        if request.game_ids:
            logger.info(f"Loading {len(request.game_ids)} specific games for simulation")
            df_pandas = load_game_data(game_ids=request.game_ids)
        elif request.start_year and request.end_year:
            logger.info(f"Loading games for years {request.start_year}-{request.end_year}")
            df_pandas = load_game_data(request.start_year, request.end_year)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either game_ids or start_year + end_year",
            )

        n_games = len(df_pandas)
        if n_games == 0:
            return SimulateGamesResponse(
                job_id=job_id,
                model_details={},
                simulation_parameters={"n_samples": request.n_samples},
                games_simulated=0,
            )

        if request.max_games and n_games > request.max_games:
            df_pandas = df_pandas.head(request.max_games)
            n_games = len(df_pandas)

        logger.info(f"Running simulation ({request.n_samples} samples, {n_games} games)...")

        # Run simulation
        results = simulate_batch(
            df_pandas,
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            n_samples=request.n_samples,
            prior_rating=request.prior_rating,
            prior_weight=request.prior_weight,
            random_state=request.random_state,
            cholesky_cache=cholesky_cache,
            geek_rating_mode="direct",
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Build predictions DataFrame
        predictions_df = build_simulation_predictions_df(results)

        # Add metadata columns
        predictions_df["score_ts"] = datetime.now(timezone.utc)

        # Persistence
        output_location = None
        data_warehouse_table = None
        data_warehouse_job_id = None
        predictions_list = None

        if request.game_ids:
            # Return inline for game_ids mode
            predictions_list = predictions_df.to_dict(orient="records")
        else:
            # Save to GCS
            local_output_path = f"/tmp/{job_id}_simulation.parquet"
            predictions_df.to_parquet(local_output_path, index=False)

            storage_client = authenticator.get_storage_client()
            bucket = storage_client.bucket(BUCKET_NAME)
            gcs_output_path = f"{ENVIRONMENT_PREFIX}/predictions/{job_id}_simulation.parquet"
            blob = bucket.blob(gcs_output_path)
            blob.upload_from_filename(local_output_path)
            output_location = f"gs://{BUCKET_NAME}/{gcs_output_path}"

            # Upload to BigQuery
            if request.upload_to_data_warehouse:
                try:
                    logger.info("Uploading simulation predictions to data warehouse")

                    model_versions = {
                        "geek_rating": geek_rating_registration["name"],
                        "geek_rating_version": geek_rating_registration["version"],
                        "geek_rating_experiment": geek_rating_registration["original_experiment"]["name"],
                        "complexity": complexity_registration["name"],
                        "complexity_version": complexity_registration["version"],
                        "complexity_experiment": complexity_registration["original_experiment"]["name"],
                        "rating": rating_registration["name"],
                        "rating_version": rating_registration["version"],
                        "rating_experiment": rating_registration["original_experiment"]["name"],
                        "users_rated": users_rated_registration["name"],
                        "users_rated_version": users_rated_registration["version"],
                        "users_rated_experiment": users_rated_registration["original_experiment"]["name"],
                    }

                    dw_uploader = DataWarehousePredictionUploader()
                    data_warehouse_job_id = dw_uploader.upload_predictions(
                        predictions_df, job_id, model_versions=model_versions
                    )
                    data_warehouse_table = dw_uploader.table_id

                    logger.info(f"Uploaded to data warehouse: {data_warehouse_table}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to upload to data warehouse: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload simulation predictions to data warehouse: {str(e)}",
                    )

        # Build model details
        model_details = {
            "complexity_model": {
                "name": complexity_registration["name"],
                "version": complexity_registration["version"],
                "experiment": complexity_registration["original_experiment"]["name"],
            },
            "rating_model": {
                "name": rating_registration["name"],
                "version": rating_registration["version"],
                "experiment": rating_registration["original_experiment"]["name"],
            },
            "users_rated_model": {
                "name": users_rated_registration["name"],
                "version": users_rated_registration["version"],
                "experiment": users_rated_registration["original_experiment"]["name"],
            },
            "geek_rating_model": {
                "name": geek_rating_registration["name"],
                "version": geek_rating_registration["version"],
                "experiment": geek_rating_registration["original_experiment"]["name"],
            },
        }

        return SimulateGamesResponse(
            job_id=job_id,
            model_details=model_details,
            simulation_parameters={
                "n_samples": request.n_samples,
                "prior_rating": request.prior_rating,
                "prior_weight": request.prior_weight,
                "random_state": request.random_state,
                "geek_rating_mode": "direct",
            },
            games_simulated=n_games,
            output_location=output_location,
            data_warehouse_table=data_warehouse_table,
            data_warehouse_job_id=data_warehouse_job_id,
            predictions=predictions_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error during simulation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Commit**

```bash
git add scoring_service/main.py
git commit -m "feat: add /simulate_games endpoint with full posterior sampling chain"
```

---

## Task 8: Update BigQuery schema for simulation columns

**Files:**
- Modify: `src/data/bigquery_uploader.py`

The existing `PREDICTIONS_LANDING_SCHEMA` needs additional columns for simulation outputs. The `DataWarehousePredictionUploader` uses `ALLOW_FIELD_ADDITION` in the job config, so new columns will be auto-added to BigQuery. However, we should define them in the schema for documentation and type safety.

**Step 1: Add simulation columns to schema**

After the existing schema fields, add:

```python
PREDICTIONS_LANDING_SCHEMA = [
    # ... existing fields ...
    # Simulation uncertainty columns (nullable - only populated by /simulate_games)
    bigquery.SchemaField("complexity_median", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_std", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_lower_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("complexity_upper_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_median", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_std", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_lower_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rating_upper_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_median", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_std", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_lower_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("users_rated_upper_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_median", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_std", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_lower_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_upper_90", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_lower_50", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("geek_rating_upper_50", "FLOAT", mode="NULLABLE"),
]
```

**Step 2: Commit**

```bash
git add src/data/bigquery_uploader.py
git commit -m "feat: add simulation uncertainty columns to BigQuery predictions schema"
```

---

## Task 9: Update Docker build

**Files:**
- Modify: `docker/scoring.Dockerfile`

The Dockerfile already copies `src/` into the image. The simulation module lives at `src/models/outcomes/simulation.py` and will be available. No changes needed to the COPY commands.

However, verify that `scipy` is available in the Docker image since `simulation.py` uses numpy but the base `TrainableModel` class imports `scipy.stats` for confidence intervals.

**Step 1: Verify scipy is in dependencies**

Check `pyproject.toml` for scipy:

```bash
grep scipy pyproject.toml
```

If scipy is listed as a dependency (it is - statsmodels depends on it), no Dockerfile changes are needed.

**Step 2: Commit (if any changes)**

If no changes are needed, skip this commit.

---

## Task 10: End-to-end verification

**Step 1: Start the scoring service locally**

```bash
cd /Users/phenrickson/Documents/projects/bgg-predictive-models
uv run uvicorn scoring_service.main:app --host 0.0.0.0 --port 8080
```

Verify it starts without import errors.

**Step 2: Test health and models endpoints**

```bash
curl http://localhost:8080/health
curl http://localhost:8080/models
```

Verify `geek_rating` appears in the models list.

**Step 3: Test `/predict_games` with game_ids (requires registered models)**

```bash
curl -X POST http://localhost:8080/predict_games \
  -H "Content-Type: application/json" \
  -d '{
    "hurdle_model_name": "hurdle-v2026",
    "complexity_model_name": "complexity-v2026",
    "rating_model_name": "rating-v2026",
    "users_rated_model_name": "users_rated-v2026",
    "geek_rating_model_name": "geek_rating-v2026",
    "game_ids": [174430, 167791, 224517]
  }'
```

Verify response includes `predicted_geek_rating` computed by the direct model pipeline.

**Step 4: Test `/simulate_games` with game_ids**

```bash
curl -X POST http://localhost:8080/simulate_games \
  -H "Content-Type: application/json" \
  -d '{
    "complexity_model_name": "complexity-v2026",
    "rating_model_name": "rating-v2026",
    "users_rated_model_name": "users_rated-v2026",
    "geek_rating_model_name": "geek_rating-v2026",
    "game_ids": [174430, 167791, 224517],
    "n_samples": 100
  }'
```

Verify response includes credible intervals for all four outcomes.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete scoring service integration with direct geek_rating and simulation"
```

---

## Summary of all files changed

| File | Change |
|------|--------|
| `scoring_service/register_model.py` | Add `geek_rating` to `--model-type` choices |
| `register.py` | Add `register_geek_rating()`, update `load_registration_config()` |
| `config.yaml` | Add `geek_rating: geek_rating-v2026` to `scoring.models` |
| `Makefile` | Add `register_geek_rating` target, update `register` deps, update scoring targets |
| `scoring_service/main.py` | Remove broken import, add geek_rating model loading to `/predict_games`, add `SimulateGamesRequest`/`Response`, add `/simulate_games` endpoint, add `geek_rating` to `/models` |
| `scoring_service/score.py` | Add `--geek-rating-model` CLI argument |
| `src/data/bigquery_uploader.py` | Add simulation uncertainty columns to schema |

## Dependencies

- Tasks 1-3 (registration) can be done in parallel
- Task 4 (fix predict_games) depends on Task 1 (register_model.py accepting geek_rating)
- Task 5 (score.py CLI) depends on Task 4
- Tasks 6-7 (simulate endpoint) depend on Task 4
- Task 8 (BigQuery schema) can be done in parallel with Tasks 6-7
- Task 9 (Docker verification) depends on all others
- Task 10 (e2e test) depends on all others

## Prerequisites

Before executing this plan:
1. The geek_rating direct model must be trained and have a `pipeline.pkl` in `models/experiments/geek_rating/ard-geek_rating/v{N}/`
2. All four Bayesian models (complexity, rating, users_rated, geek_rating) must use ARDRegression or BayesianRidge (have `coef_` and `sigma_` attributes) for simulation to work
3. GCP credentials must be configured for model registration
