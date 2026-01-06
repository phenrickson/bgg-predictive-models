# =============================================================================
# SERVICE ACCOUNTS
# =============================================================================
#
# Two service accounts are used:
#
# 1. terraform-admin (MANUAL) - Created manually, key stored as GCP_SA_KEY_BGG_ML
#    - Used by GitHub Actions and Terraform to authenticate and deploy
#    - Needs: run.admin, artifactregistry.writer, iam.serviceAccountUser, storage.admin
#
# 2. bgg-predictive-models (TERRAFORM) - Created here
#    - Cloud Run services run AS this identity (no key needed)
#    - Needs: GCS access, BigQuery access to data warehouse
#
# =============================================================================

# Workload service account - Cloud Run services run as this identity
resource "google_service_account" "workload" {
  account_id   = "bgg-predictive-models"
  display_name = "BGG Predictive Models Workload"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

# -----------------------------------------------------------------------------
# Workload SA permissions (what Cloud Run services can do at runtime)
# -----------------------------------------------------------------------------

# Storage access for model bucket
resource "google_storage_bucket_iam_member" "workload_storage" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workload.email}"
}

# CROSS-PROJECT: Read access to data warehouse BigQuery
resource "google_project_iam_member" "workload_dw_viewer" {
  project = var.data_warehouse_project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.workload.email}"
}

resource "google_project_iam_member" "workload_dw_job_user" {
  project = var.data_warehouse_project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.workload.email}"
}

resource "google_project_iam_member" "workload_dw_read_session" {
  project = var.data_warehouse_project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.workload.email}"
}

# CROSS-PROJECT: Write access to raw dataset (for predictions landing table)
resource "google_bigquery_dataset_iam_member" "workload_dw_raw_write" {
  project    = var.data_warehouse_project_id
  dataset_id = "raw"
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.workload.email}"
}

# -----------------------------------------------------------------------------
# Terraform Admin SA permissions (created manually, permissions managed here)
# -----------------------------------------------------------------------------

locals {
  terraform_admin_sa = "terraform-admin@${var.project_id}.iam.gserviceaccount.com"
}

# Deploy to Cloud Run
resource "google_project_iam_member" "terraform_admin_run" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${local.terraform_admin_sa}"
}

# Push to Artifact Registry
resource "google_project_iam_member" "terraform_admin_artifact" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${local.terraform_admin_sa}"
}

# Act as the workload service account when deploying
resource "google_service_account_iam_member" "terraform_admin_can_act_as_workload" {
  service_account_id = google_service_account.workload.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.terraform_admin_sa}"
}

# Manage storage (for terraform state, etc.)
resource "google_project_iam_member" "terraform_admin_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${local.terraform_admin_sa}"
}
