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
from src.models.embeddings.data import EmbeddingDataLoader  # noqa: E402
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
    embedding_dims: int = Field(64, description="Embedding dimensions to use (8, 16, 32, or 64)")

    # Absolute filters
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

    # Relative complexity filtering (relative to query game)
    complexity_mode: Optional[str] = Field(
        None,
        description="Complexity filter mode: 'within_band' (±band), 'less_complex', 'more_complex'. "
                    "If set, overrides min/max_complexity with values relative to query game."
    )
    complexity_band: Optional[float] = Field(
        0.5,
        description="Complexity band for relative modes (±value for within_band, max difference for directional)"
    )

    # Visualization options
    include_embeddings: bool = Field(
        False,
        description="Include embedding vectors in response for visualization (component profiles, etc.)"
    )
    include_umap: bool = Field(
        False,
        description="Include UMAP 2D coordinates in response (requires UMAP model to be registered)"
    )


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
    embedding: Optional[List[float]] = Field(None, description="Embedding vector (if include_embeddings=true)")
    umap_1: Optional[float] = Field(None, description="UMAP x-coordinate (if include_umap=true)")
    umap_2: Optional[float] = Field(None, description="UMAP y-coordinate (if include_umap=true)")


class SimilarGamesResponse(BaseModel):
    """Response for similar games query."""

    query: Dict[str, Any]
    results: List[SimilarGame]
    distance_type: str
    query_embedding: Optional[List[float]] = Field(None, description="Query game embedding (if include_embeddings=true)")
    query_umap: Optional[List[float]] = Field(None, description="Query game UMAP coordinates [x, y] (if include_umap=true)")


class EmbeddingProfileRequest(BaseModel):
    """Request for embedding profiles."""

    game_ids: List[int] = Field(..., description="List of game IDs to get embeddings for")
    model_version: Optional[int] = Field(None, description="Specific model version")
    embedding_dims: int = Field(64, description="Embedding dimensions (8, 16, 32, or 64)")
    include_umap: bool = Field(False, description="Include UMAP 2D coordinates")


class GameEmbedding(BaseModel):
    """Embedding data for a single game."""

    game_id: int
    name: Optional[str] = None
    year_published: Optional[int] = None
    complexity: Optional[float] = None
    embedding: List[float]
    umap_1: Optional[float] = Field(None, description="UMAP x-coordinate")
    umap_2: Optional[float] = Field(None, description="UMAP y-coordinate")


class EmbeddingProfileResponse(BaseModel):
    """Response containing embedding profiles."""

    games: List[GameEmbedding]
    embedding_dim: int
    model_version: int


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


class GenerateCoordinatesRequest(BaseModel):
    """Request for generating 2D coordinates from embeddings."""

    model_name: str = Field("bgg-embeddings", description="Name of registered embedding model")
    model_version: Optional[int] = Field(None, description="Specific model version")
    game_ids: Optional[List[int]] = Field(None, description="Specific game IDs to process")
    max_games: int = Field(DEFAULT_MAX_GAMES, description="Maximum games to process")
    include_umap: bool = Field(True, description="Generate UMAP coordinates")
    include_pca: bool = Field(True, description="Generate PCA coordinates")


class GameCoordinates(BaseModel):
    """2D coordinates for a single game."""

    game_id: int
    umap_1: Optional[float] = None
    umap_2: Optional[float] = None
    pca_1: Optional[float] = None
    pca_2: Optional[float] = None


class GenerateCoordinatesResponse(BaseModel):
    """Response for coordinate generation."""

    job_id: str
    model_details: Dict[str, Any]
    games_processed: int
    coordinates: Optional[List[GameCoordinates]] = Field(
        None, description="Coordinates (only returned for specific game_ids)"
    )
    table_id: Optional[str] = Field(None, description="BigQuery table (if uploaded)")
    bq_job_id: Optional[str] = None


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
    model_name: str,
    model_version: int,
    game_ids: Optional[List[int]] = None,
    max_games: int = DEFAULT_MAX_GAMES,
) -> pd.DataFrame:
    """Load games that need embeddings.

    Args:
        model_name: Name of the embedding model (used for change detection).
        model_version: Version of the model (games with older versions will be re-embedded).
        game_ids: Specific game IDs to load.
        max_games: Maximum number of games to load.

    Returns:
        DataFrame with game features and predicted_complexity.
    """
    # Use EmbeddingDataLoader to get predicted_complexity joined with features
    emb_loader = EmbeddingDataLoader(config)

    if game_ids:
        logger.info(f"Loading {len(game_ids)} specific games for embeddings...")
        return emb_loader.load_scoring_data(game_ids=game_ids).to_pandas()

    # Change detection: find games needing embeddings for THIS model
    # Games that either don't have embeddings from this model or have updated features
    logger.info(f"Loading games needing embeddings via change detection (model: {model_name})...")

    ml_project = config.ml_project_id
    dw_project = config.data_warehouse.project_id
    emb_config = config.embeddings
    # Use upload config for raw table (where we write embeddings)
    table_id = f"{ml_project}.{emb_config.upload.dataset}.{emb_config.upload.table}"

    # Score ALL games regardless of ratings - embeddings should work for everything
    # Only consider embeddings from the specified model for change detection
    # Re-embed if: no embedding exists, features changed, OR version is outdated
    query = f"""
    SELECT gf.game_id
    FROM `{dw_project}.analytics.games_features` gf
    LEFT JOIN `{dw_project}.staging.game_features_hash` fh
      ON gf.game_id = fh.game_id
    LEFT JOIN (
      SELECT game_id, created_ts, embedding_version,
             ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY created_ts DESC) as rn
      FROM `{table_id}`
      WHERE embedding_model = '{model_name}'
    ) le ON gf.game_id = le.game_id AND le.rn = 1
    WHERE gf.year_published IS NOT NULL
      AND (
        le.game_id IS NULL
        OR fh.last_updated > le.created_ts
        OR le.embedding_version != {model_version}
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
        return emb_loader.load_scoring_data(game_ids=game_ids_to_load).to_pandas()

    except Exception as e:
        logger.error(f"Change detection query failed: {e}")
        raise


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
    # Create truncated versions for efficient similarity search at different dimensions
    if "embedding" in upload_df.columns:
        def to_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        upload_df["embedding"] = upload_df["embedding"].apply(to_list)
        upload_df["embedding_8"] = upload_df["embedding"].apply(lambda x: x[:8])
        upload_df["embedding_16"] = upload_df["embedding"].apply(lambda x: x[:16])
        upload_df["embedding_32"] = upload_df["embedding"].apply(lambda x: x[:32])

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

        # Load games (change detection filters by model_name and model_version)
        games_df = load_games_for_embedding(
            model_name=request.model_name,
            model_version=registration["version"],
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


def get_query_game_complexity(game_id: int) -> Optional[float]:
    """Fetch the complexity of a game from the embeddings table.

    Args:
        game_id: The game ID to look up.

    Returns:
        The game's complexity value, or None if not found.
    """
    emb_config = config.embeddings
    project = emb_config.vector_search.project or config.ml_project_id
    dataset = emb_config.vector_search.dataset
    table = emb_config.vector_search.table
    table_id = f"{project}.{dataset}.{table}"

    query = f"""
    SELECT complexity
    FROM `{table_id}`
    WHERE game_id = @game_id
    LIMIT 1
    """

    client = bigquery.Client(project=config.ml_project_id)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("game_id", "INT64", game_id),
        ]
    )

    try:
        result = client.query(query, job_config=job_config).to_dataframe()
        if len(result) > 0 and result["complexity"].iloc[0] is not None:
            return float(result["complexity"].iloc[0])
        return None
    except Exception as e:
        logger.warning(f"Could not fetch complexity for game {game_id}: {e}")
        return None


def compute_complexity_bounds(
    query_complexity: float,
    mode: str,
    band: float,
) -> tuple[Optional[float], Optional[float]]:
    """Compute min/max complexity based on relative mode.

    Args:
        query_complexity: The query game's complexity.
        mode: One of 'within_band', 'less_complex', 'more_complex'.
        band: The complexity band value (± for within_band, max difference for directional).

    Returns:
        Tuple of (min_complexity, max_complexity).
    """
    if mode == "within_band":
        return (
            max(1.0, query_complexity - band),
            min(5.0, query_complexity + band),
        )
    elif mode == "less_complex":
        return (
            max(1.0, query_complexity - band),
            query_complexity,
        )
    elif mode == "more_complex":
        return (
            query_complexity,
            min(5.0, query_complexity + band),
        )
    else:
        raise ValueError(f"Invalid complexity_mode: {mode}")


def fetch_game_embeddings(
    game_ids: List[int],
    embedding_dims: Optional[int] = None,
    model_version: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fetch embeddings and metadata for multiple games.

    Args:
        game_ids: List of game IDs to fetch.
        embedding_dims: Embedding dimensions (8, 16, 32, or 64/None).
        model_version: Specific model version, or None for latest.

    Returns:
        Dict mapping game_id to {embedding, name, year_published, complexity}.
    """
    from src.models.embeddings.search import VALID_EMBEDDING_DIMS

    emb_config = config.embeddings
    project = emb_config.vector_search.project or config.ml_project_id
    dataset = emb_config.vector_search.dataset
    table = emb_config.vector_search.table
    table_id = f"{project}.{dataset}.{table}"

    # Determine embedding column
    if embedding_dims is None or embedding_dims == 64:
        emb_col = "embedding"
    elif embedding_dims in VALID_EMBEDDING_DIMS:
        emb_col = f"embedding_{embedding_dims}"
    else:
        raise ValueError(f"Invalid embedding_dims: {embedding_dims}")

    # Build version filter
    if model_version:
        version_filter = f"embedding_version = {model_version}"
    else:
        version_filter = f"embedding_version = (SELECT MAX(embedding_version) FROM `{table_id}`)"

    game_ids_str = ",".join(str(g) for g in game_ids)

    query = f"""
    SELECT game_id, name, year_published, complexity, {emb_col} as embedding, embedding_version
    FROM `{table_id}`
    WHERE game_id IN ({game_ids_str}) AND {version_filter}
    """

    client = bigquery.Client(project=config.ml_project_id)

    try:
        result = client.query(query).to_dataframe()
        embeddings_map = {}
        for _, row in result.iterrows():
            embeddings_map[row["game_id"]] = {
                "embedding": list(row["embedding"]),
                "name": row.get("name"),
                "year_published": row.get("year_published"),
                "complexity": row.get("complexity"),
                "embedding_version": row.get("embedding_version"),
            }
        return embeddings_map
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        raise


# Cache for loaded UMAP and PCA models (keyed by model_name:version)
_umap_model_cache: Dict[str, Any] = {}
_pca_model_cache: Dict[str, Any] = {}


def get_umap_model(model_name: str = "bgg-embeddings", model_version: Optional[int] = None) -> Optional[Any]:
    """Load and cache the UMAP model for transforming embeddings to 2D.

    Args:
        model_name: Name of the registered embedding model.
        model_version: Specific version, or None for latest.

    Returns:
        Fitted UMAP model, or None if not available.
    """
    # Build cache key
    cache_key = f"{model_name}:{model_version or 'latest'}"

    if cache_key in _umap_model_cache:
        return _umap_model_cache[cache_key]

    try:
        registered_model = RegisteredEmbeddingModel()
        umap_model = registered_model.load_umap_model(model_name, model_version)

        if umap_model is not None:
            _umap_model_cache[cache_key] = umap_model
            logger.info(f"Loaded UMAP model for {cache_key}")

        return umap_model
    except Exception as e:
        logger.warning(f"Could not load UMAP model for {cache_key}: {e}")
        return None


def transform_to_umap(
    embeddings: List[List[float]],
    model_name: str = "bgg-embeddings",
    model_version: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """Transform embeddings to 2D UMAP coordinates.

    Args:
        embeddings: List of embedding vectors (must be full 64-dim for UMAP).
        model_name: Name of the registered embedding model.
        model_version: Specific version, or None for latest.

    Returns:
        List of [umap_1, umap_2] coordinates, or None if UMAP not available.
    """
    import numpy as np

    umap_model = get_umap_model(model_name, model_version)
    if umap_model is None:
        return None

    try:
        embeddings_array = np.array(embeddings)
        umap_coords = umap_model.transform(embeddings_array)
        return [[float(x), float(y)] for x, y in umap_coords]
    except Exception as e:
        logger.warning(f"UMAP transform failed: {e}")
        return None


def get_pca_model(model_name: str = "bgg-embeddings", model_version: Optional[int] = None) -> Optional[Any]:
    """Load and cache the PCA model for transforming embeddings to 2D.

    Args:
        model_name: Name of the registered embedding model.
        model_version: Specific version, or None for latest.

    Returns:
        Fitted PCA model, or None if not available.
    """
    cache_key = f"{model_name}:{model_version or 'latest'}"

    if cache_key in _pca_model_cache:
        return _pca_model_cache[cache_key]

    try:
        registered_model = RegisteredEmbeddingModel()
        pca_model = registered_model.load_pca_model(model_name, model_version)

        if pca_model is not None:
            _pca_model_cache[cache_key] = pca_model
            logger.info(f"Loaded PCA model for {cache_key}")

        return pca_model
    except Exception as e:
        logger.warning(f"Could not load PCA model for {cache_key}: {e}")
        return None


def transform_to_pca(
    embeddings: List[List[float]],
    model_name: str = "bgg-embeddings",
    model_version: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """Transform embeddings to 2D PCA coordinates.

    Args:
        embeddings: List of embedding vectors (must be full 64-dim for PCA).
        model_name: Name of the registered embedding model.
        model_version: Specific version, or None for latest.

    Returns:
        List of [pca_1, pca_2] coordinates, or None if PCA not available.
    """
    import numpy as np

    pca_model = get_pca_model(model_name, model_version)
    if pca_model is None:
        return None

    try:
        embeddings_array = np.array(embeddings)
        pca_coords = pca_model.transform(embeddings_array)
        return [[float(x), float(y)] for x, y in pca_coords]
    except Exception as e:
        logger.warning(f"PCA transform failed: {e}")
        return None


@app.post("/similar", response_model=SimilarGamesResponse)
async def find_similar_games(request: SimilarGamesRequest):
    """Find similar games using vector similarity search.

    Supports both absolute filters (min_year, max_complexity, etc.) and
    relative complexity filtering (within_band, less_complex, more_complex)
    based on the query game's complexity.
    """
    from src.models.embeddings.search import NearestNeighborSearch, SearchFilters

    # Get defaults from config
    search_config = config.embeddings.search
    top_k = request.top_k or search_config.default_top_k
    distance_type = (request.distance_type or search_config.default_distance_type).upper()

    # Determine complexity bounds
    min_complexity = request.min_complexity
    max_complexity = request.max_complexity
    query_complexity = None
    complexity_mode_applied = None

    # Handle relative complexity mode
    if request.complexity_mode and request.game_id:
        query_complexity = get_query_game_complexity(request.game_id)
        if query_complexity is not None:
            band = request.complexity_band if request.complexity_band is not None else 0.5
            min_complexity, max_complexity = compute_complexity_bounds(
                query_complexity, request.complexity_mode, band
            )
            complexity_mode_applied = request.complexity_mode
            logger.info(
                f"Applied complexity_mode={request.complexity_mode} "
                f"(query={query_complexity:.2f}, band={band}) -> "
                f"[{min_complexity:.2f}, {max_complexity:.2f}]"
            )
        else:
            logger.warning(
                f"complexity_mode requested but query game {request.game_id} "
                "has no complexity value - ignoring"
            )

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
        min_complexity=min_complexity,
        max_complexity=max_complexity,
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
                embedding_dims=request.embedding_dims,
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
                embedding_dims=request.embedding_dims,
            )
            query_info = {"game_ids": request.game_ids}
            if request.weights:
                query_info["weights"] = request.weights

        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either game_id or game_ids"
            )

        # Add embedding_dims to query info
        query_info["embedding_dims"] = request.embedding_dims

        # Build applied filters dict for response
        applied_filters = {}
        if request.min_year is not None:
            applied_filters["min_year"] = request.min_year
        if request.max_year is not None:
            applied_filters["max_year"] = request.max_year
        if request.min_users_rated is not None:
            applied_filters["min_users_rated"] = request.min_users_rated
        if request.max_users_rated is not None:
            applied_filters["max_users_rated"] = request.max_users_rated
        if request.min_rating is not None:
            applied_filters["min_rating"] = request.min_rating
        if request.max_rating is not None:
            applied_filters["max_rating"] = request.max_rating
        if request.min_geek_rating is not None:
            applied_filters["min_geek_rating"] = request.min_geek_rating
        if request.max_geek_rating is not None:
            applied_filters["max_geek_rating"] = request.max_geek_rating

        # Add complexity info (either absolute or computed from mode)
        if complexity_mode_applied:
            applied_filters["complexity_mode"] = complexity_mode_applied
            applied_filters["complexity_band"] = request.complexity_band or 0.5
            applied_filters["query_complexity"] = query_complexity
            applied_filters["min_complexity"] = min_complexity
            applied_filters["max_complexity"] = max_complexity
        else:
            if request.min_complexity is not None:
                applied_filters["min_complexity"] = request.min_complexity
            if request.max_complexity is not None:
                applied_filters["max_complexity"] = request.max_complexity

        if applied_filters:
            query_info["filters"] = applied_filters

        # Fetch embeddings if requested for visualization (or for UMAP)
        embeddings_map = {}
        umap_coords_map = {}
        query_embedding = None
        query_umap = None

        if request.include_embeddings or request.include_umap:
            # Get all game IDs we need embeddings for
            result_game_ids = [row["game_id"] for row in results.to_dicts()]
            all_game_ids = result_game_ids.copy()
            if request.game_id:
                all_game_ids.append(request.game_id)

            # For UMAP, we need full 64-dim embeddings regardless of embedding_dims setting
            fetch_dims = None if request.include_umap else request.embedding_dims

            embeddings_map = fetch_game_embeddings(
                game_ids=all_game_ids,
                embedding_dims=fetch_dims,
                model_version=request.model_version,
            )

            # Extract query embedding
            if request.game_id and request.game_id in embeddings_map:
                query_embedding = embeddings_map[request.game_id]["embedding"]

            # Compute UMAP coordinates if requested
            if request.include_umap and embeddings_map:
                # Build list of embeddings in same order as game_ids
                ordered_embeddings = [
                    embeddings_map[gid]["embedding"]
                    for gid in all_game_ids
                    if gid in embeddings_map
                ]
                ordered_game_ids = [
                    gid for gid in all_game_ids if gid in embeddings_map
                ]

                umap_result = transform_to_umap(
                    ordered_embeddings,
                    model_version=request.model_version,
                )

                if umap_result is not None:
                    for gid, coords in zip(ordered_game_ids, umap_result):
                        umap_coords_map[gid] = coords

                    # Extract query UMAP
                    if request.game_id and request.game_id in umap_coords_map:
                        query_umap = umap_coords_map[request.game_id]

        # Convert results to response format
        similar_games = []
        for row in results.to_dicts():
            game_id = row["game_id"]
            embedding = None
            umap_1 = None
            umap_2 = None

            if request.include_embeddings and game_id in embeddings_map:
                embedding = embeddings_map[game_id]["embedding"]

            if request.include_umap and game_id in umap_coords_map:
                umap_1, umap_2 = umap_coords_map[game_id]

            similar_games.append(
                SimilarGame(
                    game_id=game_id,
                    name=row.get("name", ""),
                    year_published=row.get("year_published"),
                    users_rated=row.get("users_rated"),
                    average_rating=row.get("average_rating"),
                    geek_rating=row.get("geek_rating"),
                    complexity=row.get("complexity"),
                    thumbnail=row.get("thumbnail"),
                    distance=row["distance"],
                    embedding=embedding,
                    umap_1=umap_1,
                    umap_2=umap_2,
                )
            )

        return SimilarGamesResponse(
            query=query_info,
            results=similar_games,
            distance_type=distance_type.lower(),
            query_embedding=query_embedding,
            query_umap=query_umap,
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


@app.post("/embedding_profile", response_model=EmbeddingProfileResponse)
async def get_embedding_profile(request: EmbeddingProfileRequest):
    """Get embedding profiles for multiple games.

    Returns the embedding vectors and metadata for the requested games,
    suitable for creating component profile visualizations.

    Example use cases:
    - Plot embedding values across all components for multiple games
    - Compare how games are positioned in embedding space
    - Analyze which components differentiate similar games
    - Visualize games in 2D UMAP space (with include_umap=true)
    """
    try:
        # For UMAP, we need full 64-dim embeddings
        fetch_dims = None if request.include_umap else request.embedding_dims

        embeddings_map = fetch_game_embeddings(
            game_ids=request.game_ids,
            embedding_dims=fetch_dims,
            model_version=request.model_version,
        )

        if not embeddings_map:
            raise HTTPException(
                status_code=404,
                detail=f"No embeddings found for game_ids: {request.game_ids}"
            )

        # Determine embedding dimension and version from first result
        first_game = next(iter(embeddings_map.values()))
        embedding_dim = len(first_game["embedding"])
        model_version_found = first_game.get("embedding_version", 0)

        # Compute UMAP coordinates if requested
        umap_coords_map = {}
        if request.include_umap:
            ordered_game_ids = [gid for gid in request.game_ids if gid in embeddings_map]
            ordered_embeddings = [embeddings_map[gid]["embedding"] for gid in ordered_game_ids]

            umap_result = transform_to_umap(
                ordered_embeddings,
                model_version=request.model_version,
            )

            if umap_result is not None:
                for gid, coords in zip(ordered_game_ids, umap_result):
                    umap_coords_map[gid] = coords

        # Build response
        games = []
        for game_id in request.game_ids:
            if game_id in embeddings_map:
                data = embeddings_map[game_id]
                umap_1 = None
                umap_2 = None

                if game_id in umap_coords_map:
                    umap_1, umap_2 = umap_coords_map[game_id]

                games.append(
                    GameEmbedding(
                        game_id=game_id,
                        name=data.get("name"),
                        year_published=data.get("year_published"),
                        complexity=data.get("complexity"),
                        embedding=data["embedding"],
                        umap_1=umap_1,
                        umap_2=umap_2,
                    )
                )

        return EmbeddingProfileResponse(
            games=games,
            embedding_dim=embedding_dim,
            model_version=model_version_found,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedding profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_coordinates", response_model=GenerateCoordinatesResponse)
async def generate_coordinates(request: GenerateCoordinatesRequest):
    """Generate 2D coordinates (UMAP and/or PCA) for game embeddings.

    This endpoint fetches existing embeddings from BigQuery and projects them
    to 2D space using the registered UMAP and PCA models. Use this to:
    - Get coordinates for specific games for visualization
    - Batch process games for the coordinate table

    If game_ids is provided, returns coordinates in the response.
    Otherwise, processes games needing updates and uploads to BigQuery.
    """
    import numpy as np

    job_id = str(uuid.uuid4())
    logger.info(f"Starting coordinate generation job {job_id}")

    try:
        # Get model info
        registered_model = RegisteredEmbeddingModel()
        versions = registered_model.list_model_versions(request.model_name)
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"No registered model found: {request.model_name}"
            )

        model_version = request.model_version or max(v["version"] for v in versions)
        model_details = {
            "name": request.model_name,
            "version": model_version,
            "include_umap": request.include_umap,
            "include_pca": request.include_pca,
        }

        # Determine which games to process
        if request.game_ids:
            game_ids = request.game_ids
            logger.info(f"Processing {len(game_ids)} specific games")
        else:
            # For batch processing, get games from embeddings table
            # that need coordinate updates
            emb_config = config.embeddings
            project = emb_config.vector_search.project or config.ml_project_id
            dataset = emb_config.vector_search.dataset
            table = emb_config.vector_search.table
            table_id = f"{project}.{dataset}.{table}"

            query = f"""
            SELECT DISTINCT game_id
            FROM `{table_id}`
            WHERE embedding_version = {model_version}
            LIMIT {request.max_games}
            """

            client = bigquery.Client(project=config.ml_project_id)
            result = client.query(query).to_dataframe()
            game_ids = result["game_id"].tolist()
            logger.info(f"Found {len(game_ids)} games to process")

        if not game_ids:
            return GenerateCoordinatesResponse(
                job_id=job_id,
                model_details=model_details,
                games_processed=0,
            )

        # Fetch embeddings (need full 64-dim for projection)
        embeddings_map = fetch_game_embeddings(
            game_ids=game_ids,
            embedding_dims=None,  # Full embeddings
            model_version=model_version,
        )

        if not embeddings_map:
            return GenerateCoordinatesResponse(
                job_id=job_id,
                model_details=model_details,
                games_processed=0,
            )

        # Build ordered lists for batch transform
        ordered_game_ids = [gid for gid in game_ids if gid in embeddings_map]
        ordered_embeddings = [embeddings_map[gid]["embedding"] for gid in ordered_game_ids]

        # Transform to UMAP coordinates
        umap_coords = None
        if request.include_umap:
            umap_coords = transform_to_umap(
                ordered_embeddings,
                model_name=request.model_name,
                model_version=model_version,
            )
            if umap_coords:
                logger.info(f"Generated UMAP coordinates for {len(umap_coords)} games")
            else:
                logger.warning("UMAP model not available")

        # Transform to PCA coordinates
        pca_coords = None
        if request.include_pca:
            pca_coords = transform_to_pca(
                ordered_embeddings,
                model_name=request.model_name,
                model_version=model_version,
            )
            if pca_coords:
                logger.info(f"Generated PCA coordinates for {len(pca_coords)} games")
            else:
                logger.warning("PCA model not available")

        # Build coordinate results
        coordinates = []
        for i, game_id in enumerate(ordered_game_ids):
            coord = GameCoordinates(game_id=game_id)
            if umap_coords and i < len(umap_coords):
                coord.umap_1 = umap_coords[i][0]
                coord.umap_2 = umap_coords[i][1]
            if pca_coords and i < len(pca_coords):
                coord.pca_1 = pca_coords[i][0]
                coord.pca_2 = pca_coords[i][1]
            coordinates.append(coord)

        # If specific game_ids were requested, return in response
        if request.game_ids:
            return GenerateCoordinatesResponse(
                job_id=job_id,
                model_details=model_details,
                games_processed=len(coordinates),
                coordinates=coordinates,
            )

        # Otherwise, upload to BigQuery
        # Build DataFrame for upload
        upload_data = []
        for coord in coordinates:
            row = {
                "game_id": coord.game_id,
                "embedding_model": request.model_name,
                "embedding_version": model_version,
                "umap_1": coord.umap_1,
                "umap_2": coord.umap_2,
                "pca_1": coord.pca_1,
                "pca_2": coord.pca_2,
                "created_ts": datetime.now(),
                "job_id": job_id,
            }
            upload_data.append(row)

        upload_df = pd.DataFrame(upload_data)

        # Upload to coordinates table
        emb_config = config.embeddings
        coords_table_id = f"{config.ml_project_id}.{emb_config.upload.dataset}.game_coordinates"

        client = bigquery.Client(project=config.ml_project_id)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )

        load_job = client.load_table_from_dataframe(
            upload_df, coords_table_id, job_config=job_config
        )
        load_job.result()

        logger.info(f"Uploaded {len(upload_df)} coordinates to {coords_table_id}")

        return GenerateCoordinatesResponse(
            job_id=job_id,
            model_details=model_details,
            games_processed=len(coordinates),
            table_id=coords_table_id,
            bq_job_id=load_job.job_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating coordinates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
