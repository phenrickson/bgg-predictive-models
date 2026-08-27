# Gating for bgg-collection-scoring (the Cloud Run service behind services/collections).
#
# The service itself is deployed imperatively by .github/workflows/docker-collections-build.yml
# (`gcloud run deploy ... --allow-unauthenticated`) — there is no `google_cloud_run_v2_service`
# Terraform resource for it, same as bgg-warehouse-api in bgg-data-warehouse. Terraform owns ONLY
# the inbound invoker IAM, as an AUTHORITATIVE binding so `allUsers` can never be (re)added out
# of band. Applied by .github/workflows/terraform.yml.
#
# SEQUENCING (do not skip): this binding is added FIRST, while the deploy workflow still passes
# `--allow-unauthenticated` — so it only *adds* specific invokers on top of still-public access,
# which is safe to merge on its own. Only after this is applied does a separate PR flip the
# deploy workflow to `--no-allow-unauthenticated`, removing the public grant. Reversing this
# order (or combining both into one PR/merge) risks a window where the daily
# run-collection-scoring.yml cron has no access to the service at all.
#
# See docs/superpowers/specs/2026-07-16-service-auth-pattern-design.md (bgg-data-warehouse) and
# docs/superpowers/plans/2026-08-27-collection-filter-phase2.md (bgg-viewer) Part B.

variable "collection_scoring_invoker_members" {
  description = <<-EOT
    Principals granted roles/run.invoker on bgg-collection-scoring (AUTHORITATIVE — this is
    the complete allow-list; anything not here, including allUsers, cannot invoke).

    To give a consumer access, add its identity here and merge (terraform.yml applies it):
      - a person:  "user:someone@example.com"
      - a service: "serviceAccount:caller@project.iam.gserviceaccount.com"

    Not listed here: the run-collection-scoring.yml GH Actions cron authenticates as
    terraform-admin (GCP_SA_KEY_BGG_ML), which already holds project-level roles/run.admin — a
    superset of run.invoker — so it does not need an explicit entry in this list. Verify this
    holds after the deploy-flag flip (Part B's second PR) rather than assuming it.
  EOT
  type        = list(string)
  default = [
    "user:phil.henrickson@gmail.com",
    # bgg-viewer's Cloud Run runtime SA — mints its own ID token to call /sync/{username}
    # (src/lib/server/collections/sync.ts, reusing the mintIdToken helper already built for
    # the warehouse-api pattern). Without this, every self-serve collection link 403s.
    "serviceAccount:bgg-viewer@bgg-data-warehouse.iam.gserviceaccount.com",
  ]
}

resource "google_cloud_run_v2_service_iam_binding" "collection_scoring_invokers" {
  project  = var.project_id
  location = var.region
  name     = "bgg-collection-scoring"
  role     = "roles/run.invoker"

  members = var.collection_scoring_invoker_members
}
