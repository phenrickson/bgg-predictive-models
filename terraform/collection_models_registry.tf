# Registry table for deployed per-user collection models

resource "google_bigquery_table" "collection_models_registry" {
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "collection_models_registry"
  project             = var.project_id
  description         = "Registry of deployed per-user collection models. Active rows drive the daily scoring job."
  deletion_protection = true

  schema = jsonencode([
    {
      name = "username"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "outcome"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "model_version"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "finalize_through_year"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "gcs_path"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "registered_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    }
  ])

  labels = {
    environment = "production"
    managed_by  = "terraform"
  }
}
