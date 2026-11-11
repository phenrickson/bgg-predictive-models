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

  # Each image is ~3.7 GB, so retention is the repo's main storage cost.
  # KEEP rules override DELETE rules.

  # Keep the 2 most recent versions per package: the live image + one rollback.
  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = 2
    }
  }

  # Keep GitHub-release images forever (release-docker-images.yml tags them
  # with the bare release name, e.g. "0.5.0"). Commit-SHA tags contain no
  # dot, so they never match these prefixes.
  cleanup_policies {
    id     = "keep-releases"
    action = "KEEP"

    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
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

  # Delete any other tagged image after 14 days (anything not kept above)
  cleanup_policies {
    id     = "delete-old-tagged"
    action = "DELETE"

    condition {
      tag_state  = "TAGGED"
      older_than = "1209600s" # 14 days
    }
  }

  depends_on = [google_project_service.apis]
}
