# Import existing repository into terraform state
import {
  to = google_artifact_registry_repository.bgg_predictive_models
  id = "projects/bgg-predictive-models/locations/us-central1/repositories/bgg-predictive-models"
}

# Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "bgg_predictive_models" {
  location      = var.region
  repository_id = "bgg-predictive-models"
  description   = "Docker repository for BGG predictive models"
  format        = "DOCKER"

  # Keep only the 3 most recent images per package
  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = 3
    }
  }

  # Delete untagged images after 7 days
  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"

    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  # Delete old tagged images (non-prod) after 14 days
  cleanup_policies {
    id     = "delete-old-non-prod"
    action = "DELETE"

    condition {
      tag_state  = "TAGGED"
      older_than = "1209600s" # 14 days
    }
  }

  depends_on = [google_project_service.apis]
}
