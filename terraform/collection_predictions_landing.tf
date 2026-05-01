# Landing table for per-user collection predictions

resource "google_bigquery_table" "collection_predictions_landing" {
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "collection_predictions_landing"
  project             = var.project_id
  description         = "Append-only landing table for per-user collection predictions. Rows accumulate across model versions; downstream Dataform deduplicates."
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "score_ts"
  }

  clustering = ["username", "game_id"]

  schema = jsonencode([
    {
      name = "username"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "game_id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "outcome"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "predicted_prob"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "predicted_label"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "threshold"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "model_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "model_version"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "score_ts"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }
}
