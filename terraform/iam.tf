# Service account for ML workloads
resource "google_service_account" "ml_service" {
  account_id   = "bgg-ml-service"
  display_name = "BGG ML Service Account"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

# Storage access for model buckets
resource "google_storage_bucket_iam_member" "ml_storage" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ml_service.email}"
}

# BigQuery access within ML project (for any future ML-specific datasets)
resource "google_project_iam_member" "ml_bigquery_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ml_service.email}"
}

# CROSS-PROJECT: Read access to data warehouse BigQuery
resource "google_project_iam_member" "dw_bigquery_viewer" {
  project = var.data_warehouse_project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.ml_service.email}"
}

resource "google_project_iam_member" "dw_bigquery_job_user" {
  project = var.data_warehouse_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ml_service.email}"
}

# CROSS-PROJECT: Write access to raw dataset (for predictions landing table)
resource "google_bigquery_dataset_iam_member" "dw_raw_write" {
  project    = var.data_warehouse_project_id
  dataset_id = "raw"
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.ml_service.email}"
}

# Service account key (for local development and CI/CD)
resource "google_service_account_key" "ml_service_key" {
  service_account_id = google_service_account.ml_service.name
}

# Output the key for secure storage
output "service_account_key" {
  value     = google_service_account_key.ml_service_key.private_key
  sensitive = true
}
