variable "project_id" {
  description = "GCP project ID for ML artifacts"
  type        = string
  default     = "bgg-predictive-models"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "GCS bucket location"
  type        = string
  default     = "US"
}

variable "environment" {
  description = "Environment (dev/prod/test)"
  type        = string
}

variable "data_warehouse_project_id" {
  description = "Project ID of the data warehouse for cross-project access"
  type        = string
  default     = "bgg-data-warehouse"
}
