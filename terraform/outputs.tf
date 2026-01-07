output "models_bucket_name" {
  description = "Name of the GCS bucket for model storage"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "URL of the GCS bucket for model storage"
  value       = google_storage_bucket.models.url
}

output "workload_service_account_email" {
  description = "Email of the workload service account (used by Cloud Run)"
  value       = google_service_account.workload.email
}

output "terraform_admin_service_account_email" {
  description = "Email of the Terraform Admin service account (created manually)"
  value       = local.terraform_admin_sa
}

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}
