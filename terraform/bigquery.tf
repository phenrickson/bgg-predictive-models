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
      name = "game_name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "year_published"
      type = "INTEGER"
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
