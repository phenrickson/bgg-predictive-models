"""FastAPI service for embedding generation and similarity search."""

import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from google.cloud import bigquery

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.config import load_config  # noqa: E402
from src.data.loader import BGGDataLoader  # noqa: E402
from embeddings_service.registered_model import RegisteredEmbeddingModel  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGG Embeddings Service",
    description="Service for generating game embeddings and similarity search",
    version="1.0.0",
)

# Global config
config = load_config()

# Constants
DEFAULT_MAX_GAMES = 25000


# Request/Response models
class GenerateEmbeddingsRequest(BaseModel):
    """Request for generating embeddings."""

    model_name: str = Field(..., description="Name of registered embedding model")
    model_version: Optional[int] = Field(None, description="Specific model version")
    max_games: int = Field(DEFAULT_MAX_GAMES, description="Maximum games to process")
    game_ids: Optional[List[int]] = Field(None, description="Specific game IDs to embed")


class GenerateEmbeddingsResponse(BaseModel):
    """Response for embedding generation."""

    job_id: str
    model_details: Dict[str, Any]
    games_embedded: int
    table_id: str
    bq_job_id: Optional[str] = None


class SimilarGamesRequest(BaseModel):
    """Request for finding similar games."""

    game_id: Optional[int] = Field(None, description="Source game ID")
    game_ids: Optional[List[int]] = Field(None, description="Multiple game IDs to combine")
    weights: Optional[List[float]] = Field(None, description="Weights for combining games")
    top_k: Optional[int] = Field(None, description="Number of results")
    distance_type: Optional[str] = Field(None, description="cosine, euclidean, dot_product")
    model_version: Optional[int] = Field(None, description="Specific model version")
    # Filters
    min_year: Optional[int] = Field(None, description="Minimum year published")
    max_year: Optional[int] = Field(None, description="Maximum year published")
    min_users_rated: Optional[int] = Field(None, description="Minimum number of ratings")
    max_users_rated: Optional[int] = Field(None, description="Maximum number of ratings")
    min_rating: Optional[float] = Field(None, description="Minimum average rating")
    max_rating: Optional[float] = Field(None, description="Maximum average rating")
    min_geek_rating: Optional[float] = Field(None, description="Minimum BGG geek rating (Bayesian average)")
    max_geek_rating: Optional[float] = Field(None, description="Maximum BGG geek rating (Bayesian average)")
    min_complexity: Optional[float] = Field(None, description="Minimum complexity/weight (1-5)")
    max_complexity: Optional[float] = Field(None, description="Maximum complexity/weight (1-5)")


class SimilarGame(BaseModel):
    """A similar game result."""

    game_id: int
    name: str
    year_published: Optional[int]
    users_rated: Optional[int] = None
    average_rating: Optional[float] = None
    geek_rating: Optional[float] = None
    complexity: Optional[float] = None
    thumbnail: Optional[str] = None
    distance: float


class SimilarGamesResponse(BaseModel):
    """Response for similar games query."""

    query: Dict[str, Any]
    results: List[SimilarGame]
    distance_type: str


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str
    version: int
    status: str
    description: str
    registered_at: str
    algorithm: Optional[str] = None
    embedding_dim: Optional[int] = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    models: List[ModelInfo]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "embeddings", "timestamp": datetime.now().isoformat()}


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all registered embedding models."""
    try:
        registered_model = RegisteredEmbeddingModel()
        models = registered_model.list_registered_models()

        model_infos = []
        for m in models:
            model_info = m.get("model_info", {})
            model_infos.append(
                ModelInfo(
                    name=m["name"],
                    version=m["version"],
                    status=m["status"],
                    description=m.get("description", ""),
                    registered_at=m.get("registered_at", ""),
                    algorithm=model_info.get("algorithm"),
                    embedding_dim=model_info.get("embedding_dim"),
                )
            )

        return ModelsResponse(models=model_infos)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_games_for_embedding(
    game_ids: Optional[List[int]] = None,
    max_games: int = DEFAULT_MAX_GAMES,
) -> pd.DataFrame:
    """Load games that need embeddings.

    Args:
        game_ids: Specific game IDs to load.
        max_games: Maximum number of games to load.

    Returns:
        DataFrame with game features.
    """
    loader = BGGDataLoader(config.data_warehouse)

    if game_ids:
        logger.info(f"Loading {len(game_ids)} specific games for embeddings...")
        return loader.load_prediction_data(game_ids=game_ids).to_pandas()

    # Change detection: find games needing embeddings
    # Games that either don't have embeddings or have updated features
    logger.info("Loading games needing embeddings via change detection...")

    ml_project = config.ml_project_id
    dw_project = config.data_warehouse.project_id
    emb_config = config.embeddings
    # Use upload config for raw table (where we write embeddings)
    table_id = f"{ml_project}.{emb_config.upload.dataset}.{emb_config.upload.table}"

    # Score ALL games regardless of ratings - embeddings should work for everything
    query = f"""
    SELECT gf.game_id
    FROM `{dw_project}.analytics.games_features` gf
    LEFT JOIN `{dw_project}.staging.game_features_hash` fh
      ON gf.game_id = fh.game_id
    LEFT JOIN (
      SELECT game_id, created_ts,
             ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY created_ts DESC) as rn
      FROM `{table_id}`
    ) le ON gf.game_id = le.game_id AND le.rn = 1
    WHERE gf.year_published IS NOT NULL
      AND (
        le.game_id IS NULL
        OR fh.last_updated > le.created_ts
      )
    LIMIT {max_games}
    """

    try:
        client = bigquery.Client(project=ml_project)
        result = client.query(query).to_dataframe()
        game_ids_to_load = result["game_id"].tolist()

        if not game_ids_to_load:
            logger.info("No games need embedding updates")
            return pd.DataFrame()

        logger.info(f"Found {len(game_ids_to_load)} games needing embeddings")
        return loader.load_prediction_data(game_ids=game_ids_to_load).to_pandas()

    except Exception as e:
        logger.warning(f"Change detection query failed, falling back to year filter: {e}")
        # Fallback: load games from scoring years (all games, no min_ratings filter)
        where_clause = (
            f"year_published >= {config.years.score_start} "
            f"AND year_published <= {config.years.score_end}"
        )
        return loader.load_data(where_clause).to_pandas()


def upload_embeddings_to_bigquery(
    embeddings_df: pd.DataFrame,
    job_id: str,
    model_name: str,
    model_version: int,
    algorithm: str,
    embedding_dim: int,
) -> str:
    """Upload embeddings to BigQuery.

    Args:
        embeddings_df: DataFrame with game_id, name, year_published, embedding columns.
        job_id: Unique job identifier.
        model_name: Name of the model used.
        model_version: Version of the model.
        algorithm: Algorithm used (svd, pca, etc.).
        embedding_dim: Dimension of embeddings.

    Returns:
        BigQuery job ID.
    """
    emb_config = config.embeddings
    # Use upload config for raw table (where we write embeddings)
    table_id = f"{config.ml_project_id}.{emb_config.upload.dataset}.{emb_config.upload.table}"

    # Prepare data for upload
    upload_df = embeddings_df.copy()
    upload_df["embedding_model"] = model_name
    upload_df["embedding_version"] = model_version
    upload_df["embedding_dim"] = embedding_dim
    upload_df["algorithm"] = algorithm
    upload_df["created_ts"] = datetime.now()
    upload_df["job_id"] = job_id

    # Convert embedding arrays to list format for BigQuery
    if "embedding" in upload_df.columns:
        upload_df["embedding"] = upload_df["embedding"].apply(
            lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
        )

    client = bigquery.Client(project=config.ml_project_id)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )

    load_job = client.load_table_from_dataframe(
        upload_df, table_id, job_config=job_config
    )
    load_job.result()

    logger.info(f"Uploaded {len(upload_df)} embeddings to {table_id}")
    return load_job.job_id


@app.post("/generate_embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    """Generate embeddings for games and upload to BigQuery."""
    job_id = str(uuid.uuid4())
    logger.info(f"Starting embedding generation job {job_id}")

    try:
        # Load registered model
        registered_model = RegisteredEmbeddingModel()
        pipeline, registration = registered_model.load_registered_model(
            request.model_name, request.model_version
        )

        model_info = registration.get("model_info", {})
        algorithm = model_info.get("algorithm", "unknown")
        embedding_dim = model_info.get("embedding_dim", config.embeddings.embedding_dim)

        logger.info(
            f"Loaded model {request.model_name} v{registration['version']} "
            f"(algorithm={algorithm}, dim={embedding_dim})"
        )

        # Load games
        games_df = load_games_for_embedding(
            game_ids=request.game_ids,
            max_games=request.max_games,
        )

        if games_df.empty:
            return GenerateEmbeddingsResponse(
                job_id=job_id,
                model_details={
                    "name": request.model_name,
                    "version": registration["version"],
                    "algorithm": algorithm,
                },
                games_embedded=0,
                table_id=f"{config.ml_project_id}.{config.embeddings.upload.dataset}.{config.embeddings.upload.table}",
            )

        logger.info(f"Generating embeddings for {len(games_df)} games...")

        # Generate embeddings using pipeline transform
        # Returns DataFrame with shape (n_games, embedding_dim)
        embeddings = pipeline.transform(games_df)

        # Build results DataFrame
        # embeddings.values converts to numpy array, iterate rows to get list of vectors
        results_df = pd.DataFrame({
            "game_id": games_df["game_id"].values,
            "name": games_df["name"].values if "name" in games_df.columns else None,
            "year_published": games_df["year_published"].values if "year_published" in games_df.columns else None,
            "embedding": [row for row in embeddings.values],
        })

        # Upload to BigQuery (skip if specific game_ids provided - return in response)
        bq_job_id = None
        if request.game_ids is None:
            bq_job_id = upload_embeddings_to_bigquery(
                results_df,
                job_id=job_id,
                model_name=request.model_name,
                model_version=registration["version"],
                algorithm=algorithm,
                embedding_dim=embedding_dim,
            )

        return GenerateEmbeddingsResponse(
            job_id=job_id,
            model_details={
                "name": request.model_name,
                "version": registration["version"],
                "algorithm": algorithm,
                "embedding_dim": embedding_dim,
            },
            games_embedded=len(results_df),
            table_id=f"{config.ml_project_id}.{config.embeddings.upload.dataset}.{config.embeddings.upload.table}",
            bq_job_id=bq_job_id,
        )

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarGamesResponse)
async def find_similar_games(request: SimilarGamesRequest):
    """Find similar games using vector similarity search."""
    from src.models.embeddings.search import NearestNeighborSearch, SearchFilters

    # Get defaults from config
    search_config = config.embeddings.search
    top_k = request.top_k or search_config.default_top_k
    distance_type = (request.distance_type or search_config.default_distance_type).upper()

    # Build filters from request
    filters = SearchFilters(
        min_year=request.min_year,
        max_year=request.max_year,
        min_users_rated=request.min_users_rated,
        max_users_rated=request.max_users_rated,
        min_rating=request.min_rating,
        max_rating=request.max_rating,
        min_geek_rating=request.min_geek_rating,
        max_geek_rating=request.max_geek_rating,
        min_complexity=request.min_complexity,
        max_complexity=request.max_complexity,
    )

    try:
        search = NearestNeighborSearch()

        if request.game_id and not request.game_ids:
            # Single game query
            results = search.find_similar_games(
                game_id=request.game_id,
                top_k=top_k,
                distance_type=distance_type,
                model_version=request.model_version,
                filters=filters,
            )
            query_info = {"game_id": request.game_id}

        elif request.game_ids:
            # Multiple games - combine embeddings
            results = search.find_games_like(
                game_ids=request.game_ids,
                top_k=top_k,
                distance_type=distance_type,
                model_version=request.model_version,
                filters=filters,
            )
            query_info = {"game_ids": request.game_ids}
            if request.weights:
                query_info["weights"] = request.weights

        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either game_id or game_ids"
            )

        # Add applied filters to query info
        if filters.has_filters():
            query_info["filters"] = {
                k: v for k, v in {
                    "min_year": request.min_year,
                    "max_year": request.max_year,
                    "min_users_rated": request.min_users_rated,
                    "max_users_rated": request.max_users_rated,
                    "min_rating": request.min_rating,
                    "max_rating": request.max_rating,
                    "min_geek_rating": request.min_geek_rating,
                    "max_geek_rating": request.max_geek_rating,
                    "min_complexity": request.min_complexity,
                    "max_complexity": request.max_complexity,
                }.items() if v is not None
            }

        # Convert results to response format
        similar_games = []
        for row in results.to_dicts():
            similar_games.append(
                SimilarGame(
                    game_id=row["game_id"],
                    name=row.get("name", ""),
                    year_published=row.get("year_published"),
                    users_rated=row.get("users_rated"),
                    average_rating=row.get("average_rating"),
                    geek_rating=row.get("geek_rating"),
                    complexity=row.get("complexity"),
                    thumbnail=row.get("thumbnail"),
                    distance=row["distance"],
                )
            )

        return SimilarGamesResponse(
            query=query_info,
            results=similar_games,
            distance_type=distance_type.lower(),
        )

    except Exception as e:
        logger.error(f"Error finding similar games: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embedding_stats")
async def get_embedding_stats(model_version: Optional[int] = None):
    """Get statistics about stored embeddings."""
    from src.models.embeddings.search import NearestNeighborSearch

    try:
        search = NearestNeighborSearch()
        stats = search.get_embedding_stats(model_version=model_version)
        return stats

    except Exception as e:
        logger.error(f"Error getting embedding stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
