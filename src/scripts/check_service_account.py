"""Script to check what service account is attached and its permissions."""

import os
from google.auth import default
from google.cloud import bigquery, storage
import requests


def check_service_account_info():
    """Check what service account is currently being used."""
    print("🔍 Checking Service Account Information")
    print("=" * 50)

    try:
        # Get default credentials and project
        credentials, project_id = default()

        print(f"📋 Project ID: {project_id}")
        print(f"🔑 Credentials Type: {type(credentials).__name__}")

        # Check if we have service account email
        if hasattr(credentials, "service_account_email"):
            print(f"📧 Service Account Email: {credentials.service_account_email}")
        elif hasattr(credentials, "_service_account_email"):
            print(f"📧 Service Account Email: {credentials._service_account_email}")
        else:
            print("📧 Service Account Email: Not available (might be user credentials)")

        # Try to get more info from metadata service (works in Cloud Run/GCE)
        try:
            print("\n🌐 Checking Metadata Service...")
            metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
            headers = {"Metadata-Flavor": "Google"}
            response = requests.get(metadata_url, headers=headers, timeout=5)

            if response.status_code == 200:
                service_account_email = response.text
                print(f"✅ Metadata Service Account: {service_account_email}")

                # Get scopes
                scopes_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes"
                scopes_response = requests.get(scopes_url, headers=headers, timeout=5)
                if scopes_response.status_code == 200:
                    scopes = scopes_response.text.strip().split("\n")
                    print(f"🔐 Available Scopes:")
                    for scope in scopes:
                        print(f"   - {scope}")
            else:
                print(
                    "❌ Not running in Google Cloud environment (no metadata service)"
                )

        except Exception as e:
            print(f"❌ Metadata service not available: {str(e)}")
            print("   (This is normal for local development)")

        return credentials, project_id

    except Exception as e:
        print(f"❌ Error getting credentials: {str(e)}")
        return None, None


def test_bigquery_permissions(credentials, project_id):
    """Test BigQuery permissions."""
    print("\n🗄️  Testing BigQuery Permissions")
    print("=" * 40)

    try:
        client = bigquery.Client(project=project_id, credentials=credentials)

        # Test basic query
        query = "SELECT 1 as test_value"
        query_job = client.query(query)
        results = list(query_job.result())
        print("✅ Basic BigQuery access: WORKING")

        # Test dataset access
        try:
            dataset_id = "bgg_data_dev"
            dataset = client.get_dataset(dataset_id)
            print(f"✅ Dataset '{dataset_id}' access: WORKING")

            # List tables
            tables = list(client.list_tables(dataset))
            print(f"📊 Found {len(tables)} tables in dataset")
            for table in tables[:3]:  # Show first 3
                print(f"   - {table.table_id}")
            if len(tables) > 3:
                print(f"   ... and {len(tables) - 3} more")

        except Exception as e:
            print(f"❌ Dataset access failed: {str(e)}")

        return True

    except Exception as e:
        print(f"❌ BigQuery test failed: {str(e)}")
        return False


def test_gcs_permissions(credentials, project_id):
    """Test Google Cloud Storage permissions."""
    print("\n☁️  Testing GCS Permissions")
    print("=" * 40)

    try:
        client = storage.Client(project=project_id, credentials=credentials)

        # List buckets
        buckets = list(client.list_buckets())
        print(f"✅ GCS access: WORKING")
        print(f"🪣 Found {len(buckets)} accessible buckets")

        for bucket in buckets[:3]:  # Show first 3
            print(f"   - {bucket.name}")
        if len(buckets) > 3:
            print(f"   ... and {len(buckets) - 3} more")

        return True

    except Exception as e:
        print(f"❌ GCS test failed: {str(e)}")
        return False


def check_environment():
    """Check what environment we're running in."""
    print("\n🌍 Environment Detection")
    print("=" * 30)

    # Check for Cloud Run
    if os.getenv("K_SERVICE"):
        print(f"🚀 Running in Cloud Run")
        print(f"   Service: {os.getenv('K_SERVICE')}")
        print(f"   Revision: {os.getenv('K_REVISION', 'unknown')}")
        return "cloud_run"

    # Check for App Engine
    elif os.getenv("GAE_ENV"):
        print(f"🏗️  Running in App Engine")
        return "app_engine"

    # Check for Compute Engine
    elif os.getenv("GOOGLE_CLOUD_PROJECT"):
        print(f"💻 Running in Compute Engine or similar")
        return "compute_engine"

    # Local development
    else:
        print(f"🏠 Running in local development")
        return "local"


def main():
    """Main function to check service account and permissions."""
    print("🔐 Service Account & Permissions Checker")
    print("=" * 60)

    # Check environment
    env = check_environment()

    # Check service account
    credentials, project_id = check_service_account_info()

    if credentials and project_id:
        # Test permissions
        bq_success = test_bigquery_permissions(credentials, project_id)
        gcs_success = test_gcs_permissions(credentials, project_id)

        # Summary
        print("\n📋 SUMMARY")
        print("=" * 20)
        print(f"Environment: {env}")
        print(f"BigQuery:    {'✅ WORKING' if bq_success else '❌ FAILED'}")
        print(f"GCS:         {'✅ WORKING' if gcs_success else '❌ FAILED'}")

        if env == "cloud_run" and bq_success and gcs_success:
            print(
                "\n🎉 Service account is properly attached and has correct permissions!"
            )
        elif env == "local" and bq_success and gcs_success:
            print("\n🎉 Local authentication is working correctly!")
        else:
            print("\n⚠️  Some issues detected. Check permissions above.")

    else:
        print("\n❌ Could not determine service account information")


if __name__ == "__main__":
    main()
