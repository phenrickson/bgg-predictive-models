"""FastAPI service for text embedding generation."""

import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import bigquery

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.config import load_config  # noqa: E402
from src.models.training import load_data  # noqa: E402
from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGG Text Embeddings Service",
    description="Service for generating text embeddings from game descriptions",
    version="1.0.0",
)

# Global config
config = load_config()

# Constants
DEFAULT_MAX_GAMES = 25000


# Request/Response models
class GenerateEmbeddingsRequest(BaseModel):
    """Request for generating text embeddings."""

    model_name: str = Field(..., description="Name of registered text embedding model")
    model_version: Optional[int] = Field(None, description="Specific model version")
    max_games: int = Field(DEFAULT_MAX_GAMES, description="Maximum games to process")
    game_ids: Optional[List[int]] = Field(None, description="Specific game IDs to embed")
    start_year: Optional[int] = Field(None, description="Start year for filtering games")
    end_year: Optional[int] = Field(None, description="End year for filtering games")


class GenerateEmbeddingsResponse(BaseModel):
    """Response for embedding generation."""

    job_id: str
    model_details: Dict[str, Any]
    games_embedded: int
    table_id: str
    bq_job_id: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str
    version: int
    status: str
    description: str
    registered_at: str
    algorithm: Optional[str] = None
    embedding_dim: Optional[int] = None
    document_method: Optional[str] = None
    vocab_size: Optional[int] = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    models: List[ModelInfo]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "text-embeddings",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all registered text embedding models."""
    try:
        registered_model = RegisteredTextEmbeddingModel()
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
                    document_method=model_info.get("document_method"),
                    vocab_size=model_info.get("vocab_size"),
                )
            )

        return ModelsResponse(models=model_infos)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_games_for_embedding(
    model_name: str,
    game_ids: Optional[List[int]] = None,
    max_games: int = DEFAULT_MAX_GAMES,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
):
    """Load games that need text embeddings.

    Args:
        model_name: Name of the embedding model (used for change detection).
        game_ids: Specific game IDs to load.
        max_games: Maximum number of games to load.
        start_year: Start year for filtering.
        end_year: End year for filtering.

    Returns:
        Polars DataFrame with game_id, name, description columns.
    """
    import polars as pl

    # Determine end year from config if not specified
    if end_year is None:
        end_year = config.years.score.end

    if game_ids:
        logger.info(f"Loading {len(game_ids)} specific games for text embeddings...")
        df = load_data(end_train_year=end_year)
        df = df.filter(pl.col("game_id").is_in(game_ids))
        return df

    # Change detection: find games needing embeddings for THIS model
    logger.info(f"Loading games needing text embeddings via change detection (model: {model_name})...")

    ml_project = config.ml_project_id
    dw_project = config.data_warehouse.project_id

    # Get upload config
    if config.text_embeddings and config.text_embeddings.upload:
        dataset = config.text_embeddings.upload.dataset
        table = config.text_embeddings.upload.table
    else:
        dataset = "raw"
        table = "description_embeddings"

    table_id = f"{ml_project}.{dataset}.{table}"

    # Build year filter
    year_filter = "WHERE gf.year_published IS NOT NULL"
    if start_year:
        year_filter += f" AND gf.year_published >= {start_year}"
    if end_year:
        year_filter += f" AND gf.year_published <= {end_year}"

    # Query for games needing embeddings
    # Games that either don't have embeddings from this model or have been updated
    query = f"""
    SELECT gf.game_id
    FROM `{dw_project}.analytics.games_features` gf
    LEFT JOIN (
      SELECT game_id, created_ts,
             ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY created_ts DESC) as rn
      FROM `{table_id}`
      WHERE embedding_model = '{model_name}'
    ) le ON gf.game_id = le.game_id AND le.rn = 1
    {year_filter}
      AND le.game_id IS NULL
    LIMIT {max_games}
    """

    try:
        client = bigquery.Client(project=ml_project)
        result = client.query(query).to_dataframe()
        game_ids_to_load = result["game_id"].tolist()

        if not game_ids_to_load:
            logger.info("No games need text embedding updates")
            return pl.DataFrame()

        logger.info(f"Found {len(game_ids_to_load)} games needing text embeddings")

        # Load full data for these games
        df = load_data(end_train_year=end_year)
        df = df.filter(pl.col("game_id").is_in(game_ids_to_load))
        return df

    except Exception as e:
        logger.error(f"Change detection query failed: {e}")
        # Fall back to loading all games if change detection fails
        logger.info("Falling back to loading all games...")
        df = load_data(end_train_year=end_year)
        if start_year:
            df = df.filter(pl.col("year_published") >= start_year)
        return df.head(max_games)


def upload_embeddings_to_bigquery(
    embeddings_df,
    job_id: str,
    model_name: str,
    model_version: int,
    algorithm: str,
    embedding_dim: int,
    document_method: Optional[str] = None,
) -> str:
    """Upload text embeddings to BigQuery.

    Args:
        embeddings_df: Polars DataFrame with game_id, name, embedding columns.
        job_id: Unique job identifier.
        model_name: Name of the model used.
        model_version: Version of the model.
        algorithm: Algorithm used (pmi, etc.).
        embedding_dim: Dimension of embeddings.
        document_method: Document aggregation method.

    Returns:
        BigQuery job ID.
    """
    import pandas as pd

    # Get upload config
    if config.text_embeddings and config.text_embeddings.upload:
        dataset = config.text_embeddings.upload.dataset
        table = config.text_embeddings.upload.table
    else:
        dataset = "raw"
        table = "description_embeddings"

    table_id = f"{config.ml_project_id}.{dataset}.{table}"

    # Convert to pandas for BigQuery upload
    upload_df = embeddings_df.to_pandas()
    upload_df["embedding_model"] = model_name
    upload_df["embedding_version"] = model_version
    upload_df["embedding_dim"] = embedding_dim
    upload_df["algorithm"] = algorithm
    upload_df["document_method"] = document_method
    upload_df["created_ts"] = datetime.now()
    upload_df["job_id"] = job_id

    # Convert embedding arrays to list format for BigQuery
    if "embedding" in upload_df.columns:
        def to_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)
        upload_df["embedding"] = upload_df["embedding"].apply(to_list)

    client = bigquery.Client(project=config.ml_project_id)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )

    load_job = client.load_table_from_dataframe(
        upload_df, table_id, job_config=job_config
    )
    load_job.result()

    logger.info(f"Uploaded {len(upload_df)} text embeddings to {table_id}")
    return load_job.job_id


@app.post("/generate_embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    """Generate text embeddings for games and upload to BigQuery."""
    import polars as pl

    job_id = str(uuid.uuid4())
    logger.info(f"Starting text embedding generation job {job_id}")

    try:
        # Load registered model
        registered_model = RegisteredTextEmbeddingModel()
        word_model, doc_model, registration = registered_model.load_registered_model(
            request.model_name, request.model_version
        )

        model_info = registration.get("model_info", {})
        algorithm = model_info.get("algorithm", "pmi")
        embedding_dim = model_info.get("embedding_dim", 100)
        document_method = model_info.get("document_method", "mean")
        version = registration["version"]

        logger.info(
            f"Loaded model {request.model_name} v{version} "
            f"(algorithm={algorithm}, dim={embedding_dim}, method={document_method})"
        )

        # Get upload config for table_id
        if config.text_embeddings and config.text_embeddings.upload:
            dataset = config.text_embeddings.upload.dataset
            table = config.text_embeddings.upload.table
        else:
            dataset = "raw"
            table = "description_embeddings"

        table_id = f"{config.ml_project_id}.{dataset}.{table}"

        # Load games
        games_df = load_games_for_embedding(
            model_name=request.model_name,
            game_ids=request.game_ids,
            max_games=request.max_games,
            start_year=request.start_year,
            end_year=request.end_year,
        )

        if len(games_df) == 0:
            return GenerateEmbeddingsResponse(
                job_id=job_id,
                model_details={
                    "name": request.model_name,
                    "version": version,
                    "algorithm": algorithm,
                    "embedding_dim": embedding_dim,
                    "document_method": document_method,
                },
                games_embedded=0,
                table_id=table_id,
            )

        # Check for description column
        if "description" not in games_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Data does not contain 'description' column"
            )

        logger.info(f"Generating text embeddings for {len(games_df)} games...")

        # Generate embeddings
        descriptions = games_df["description"].fill_null("").to_list()
        embeddings = doc_model.transform(descriptions)

        # Create output dataframe
        embeddings_df = pl.DataFrame({
            "game_id": games_df["game_id"],
            "name": games_df["name"] if "name" in games_df.columns else None,
            "embedding": [emb.tolist() for emb in embeddings],
        })

        logger.info(f"Generated {len(embeddings_df)} text embeddings")

        # Upload to BigQuery (skip if specific game_ids provided)
        bq_job_id = None
        if request.game_ids is None:
            bq_job_id = upload_embeddings_to_bigquery(
                embeddings_df,
                job_id=job_id,
                model_name=request.model_name,
                model_version=version,
                algorithm=algorithm,
                embedding_dim=embedding_dim,
                document_method=document_method,
            )

        return GenerateEmbeddingsResponse(
            job_id=job_id,
            model_details={
                "name": request.model_name,
                "version": version,
                "algorithm": algorithm,
                "embedding_dim": embedding_dim,
                "document_method": document_method,
            },
            games_embedded=len(embeddings_df),
            table_id=table_id,
            bq_job_id=bq_job_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
