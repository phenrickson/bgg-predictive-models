# Model storage bucket (per environment)
resource "google_storage_bucket" "models" {
  name     = "bgg-predictive-models-${var.environment}"
  location = var.location
  project  = var.project_id

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age        = 90
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    purpose     = "ml-models"
  }

  depends_on = [google_project_service.apis]
}

# Terraform state bucket (only needed once, not per environment)
# This should be created manually or in a bootstrap step before running terraform init
# resource "google_storage_bucket" "terraform_state" {
#   name     = "bgg-predictive-models-terraform-state"
#   location = var.location
#   project  = var.project_id
#
#   uniform_bucket_level_access = true
#   versioning {
#     enabled = true
#   }
# }
