output "models_bucket_name" {
  description = "Name of the GCS bucket for model storage"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "URL of the GCS bucket for model storage"
  value       = google_storage_bucket.models.url
}

output "service_account_email" {
  description = "Email of the ML service account"
  value       = google_service_account.ml_service.email
}

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}
