# BigQuery resources for ML predictions

# Raw dataset for predictions landing
resource "google_bigquery_dataset" "raw" {
  dataset_id  = "raw"
  project     = var.project_id
  location    = var.location
  description = "Raw landing zone for ML predictions"

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

# Predictions landing table
resource "google_bigquery_table" "ml_predictions_landing" {
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "ml_predictions_landing"
  project             = var.project_id
  description         = "Landing table for ML predictions - consumed by Dataform in bgg-data-warehouse"
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "score_ts"
  }

  clustering = ["game_id"]

  schema = jsonencode([
    {
      name = "job_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "game_id"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "year_published"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "predicted_hurdle_prob"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "predicted_complexity"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "predicted_rating"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "predicted_users_rated"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "predicted_geek_rating"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "geek_rating_model_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "geek_rating_model_version"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "geek_rating_experiment"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "hurdle_model_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "hurdle_model_version"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "hurdle_experiment"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "complexity_model_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "complexity_model_version"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "complexity_experiment"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "rating_model_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "rating_model_version"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "rating_experiment"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "users_rated_model_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "users_rated_model_version"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "users_rated_experiment"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "score_ts"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "source_environment"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }
}

# Complexity predictions table
resource "google_bigquery_table" "complexity_predictions" {
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "complexity_predictions"
  project             = var.project_id
  description         = "Daily complexity predictions for board games"
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "score_ts"
  }

  clustering = ["game_id"]

  schema = jsonencode([
    {
      name        = "game_id"
      type        = "INTEGER"
      mode        = "REQUIRED"
      description = "BGG game ID"
    },
    {
      name        = "name"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Game name"
    },
    {
      name        = "year_published"
      type        = "FLOAT"
      mode        = "NULLABLE"
      description = "Year game was published"
    },
    {
      name        = "predicted_complexity"
      type        = "FLOAT"
      mode        = "REQUIRED"
      description = "Predicted complexity score (1-5 scale)"
    },
    {
      name        = "complexity_model_name"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Name of complexity model used"
    },
    {
      name        = "complexity_model_version"
      type        = "INTEGER"
      mode        = "REQUIRED"
      description = "Version number of the model"
    },
    {
      name        = "complexity_experiment"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Experiment name the model was trained from"
    },
    {
      name        = "score_ts"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When this prediction was scored"
    },
    {
      name        = "job_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique job ID for this scoring run"
    }
  ])

  labels = {
    environment = "production"
    managed_by  = "terraform"
    model_type  = "complexity"
  }
}

# Game embeddings table for vector search
resource "google_bigquery_table" "game_embeddings" {
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "game_embeddings"
  project             = var.project_id
  description         = "Game embeddings for vector search - nearest neighbor queries"
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "created_ts"
  }

  clustering = ["game_id"]

  schema = jsonencode([
    {
      name        = "game_id"
      type        = "INTEGER"
      mode        = "REQUIRED"
      description = "BGG game ID"
    },
    {
      name        = "name"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Game name"
    },
    {
      name        = "year_published"
      type        = "INTEGER"
      mode        = "NULLABLE"
      description = "Year game was published"
    },
    {
      name        = "embedding"
      type        = "FLOAT64"
      mode        = "REPEATED"
      description = "Game embedding vector for vector search"
    },
    {
      name        = "embedding_model"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Name of embedding model used"
    },
    {
      name        = "embedding_version"
      type        = "INTEGER"
      mode        = "REQUIRED"
      description = "Version number of the embedding model"
    },
    {
      name        = "embedding_dim"
      type        = "INTEGER"
      mode        = "REQUIRED"
      description = "Dimensionality of the embedding"
    },
    {
      name        = "algorithm"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Algorithm used (pca, svd, umap, autoencoder)"
    },
    {
      name        = "created_ts"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When this embedding was created"
    },
    {
      name        = "job_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique job ID for this embedding run"
    }
  ])

  labels = {
    environment = "production"
    managed_by  = "terraform"
    model_type  = "embeddings"
  }
}
