"""Simple script to demonstrate BGG data loading and GCS access with proper authentication."""

import os
from google.cloud import storage
from src.data.config import load_config
from src.data.loader import BGGDataLoader


def test_gcs_authentication(config):
    """Test Google Cloud Storage authentication using the same credentials."""
    print("\n☁️  Testing Google Cloud Storage authentication...")

    try:
        # Create GCS client using same authentication approach as BigQuery
        if config.credentials_path and os.path.exists(config.credentials_path):
            # Use service account credentials
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                config.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            gcs_client = storage.Client(
                project=config.project_id, credentials=credentials
            )
        else:
            # Use default credentials (works for Cloud Run)
            gcs_client = storage.Client(project=config.project_id)

        print(f"✅ GCS client initialized for project: {gcs_client.project}")

        # List some buckets to test access (this requires minimal permissions)
        print("🪣 Testing bucket access...")
        buckets = list(gcs_client.list_buckets())

        if buckets:
            print(f"✅ Found {len(buckets)} accessible buckets:")
            for bucket in buckets[:3]:  # Show first 3 buckets
                print(f"   - {bucket.name}")
            if len(buckets) > 3:
                print(f"   ... and {len(buckets) - 3} more")
        else:
            print("ℹ️  No buckets found (this might be normal depending on permissions)")

        return True

    except Exception as e:
        print(f"❌ GCS authentication failed: {str(e)}")
        return False


def main():
    """Demonstrate authentication for both BigQuery and GCS."""
    print("🔐 BGG Data Loading & GCS Authentication Demonstration")
    print("=" * 60)

    bigquery_success = False
    gcs_success = False

    try:
        # Test different authentication modes
        print("📋 Testing authentication modes...")

        # Test 1: Auto-detect mode (default behavior)
        print("\n🔍 AUTO-DETECT MODE:")
        config_auto = load_config()
        print(f"   Project ID: {config_auto.project_id}")
        print(f"   Dataset: {config_auto.dataset}")
        if config_auto.credentials_path:
            print(f"   Using service account: {config_auto.credentials_path}")
        else:
            print("   Using default credentials (Cloud Run style)")

        # Test 2: Force default credentials (Cloud Run simulation)
        print("\n☁️  CLOUD RUN MODE (forced default credentials):")
        config_cloud = load_config(use_service_account=False)
        print(f"   Project ID: {config_cloud.project_id}")
        print(f"   Dataset: {config_cloud.dataset}")
        print(f"   Credentials path: {config_cloud.credentials_path} (should be None)")

        # Use the auto-detected config for actual testing
        config = config_auto

        # Test 1: BigQuery Authentication
        print("\n" + "=" * 50)
        print("🗄️  TESTING BIGQUERY AUTHENTICATION")
        print("=" * 50)

        try:
            # Initialize loader
            print("🚀 Initializing BGGDataLoader...")
            loader = BGGDataLoader(config)
            print("✅ BGGDataLoader initialized successfully")

            # Test with a small data load
            print("📊 Loading sample training data...")
            df = loader.load_training_data(
                end_train_year=2020,  # Smaller date range
                min_ratings=100,  # Higher threshold for smaller dataset
            )

            print(f"✅ Successfully loaded {len(df)} games")
            print(f"📋 Columns available: {len(df.columns)}")
            print(f"🎯 Sample columns: {list(df.columns)[:10]}...")

            # Show some basic stats
            if len(df) > 0:
                print(f"📈 Data summary:")
                print(
                    f"   Year range: {df['year_published'].min()} - {df['year_published'].max()}"
                )
                print(f"   Average rating: {df['rating'].mean():.2f}")
                print(f"   Average complexity: {df['complexity'].mean():.2f}")

            print("✅ BigQuery authentication and data loading successful!")
            bigquery_success = True

        except Exception as e:
            print(f"❌ BigQuery test failed: {str(e)}")

        # Test 2: GCS Authentication
        print("\n" + "=" * 50)
        print("☁️  TESTING GOOGLE CLOUD STORAGE AUTHENTICATION")
        print("=" * 50)

        gcs_success = test_gcs_authentication(config)

        # Test 3: Cloud Run Simulation (force default credentials)
        print("\n" + "=" * 50)
        print("🌐 TESTING CLOUD RUN SIMULATION (Default Credentials Only)")
        print("=" * 50)

        cloud_run_success = False
        try:
            print("🔄 Testing with forced default credentials...")
            cloud_config = load_config(use_service_account=False)

            # Quick BigQuery test with default credentials
            cloud_loader = BGGDataLoader(cloud_config)
            print("✅ Cloud Run style authentication successful!")
            cloud_run_success = True

        except Exception as e:
            print(f"❌ Cloud Run simulation failed: {str(e)}")
            print("   This might be expected if you don't have gcloud auth set up")

        # Final Summary
        print("\n" + "=" * 60)
        print("📋 AUTHENTICATION SUMMARY")
        print("=" * 60)
        print(f"BigQuery:         {'✅ PASS' if bigquery_success else '❌ FAIL'}")
        print(f"GCS:              {'✅ PASS' if gcs_success else '❌ FAIL'}")
        print(f"Cloud Run Style:  {'✅ PASS' if cloud_run_success else '❌ FAIL'}")

        if bigquery_success and gcs_success:
            print("\n🎉 All authentication tests passed!")
            if cloud_run_success:
                print("🚀 Ready for Cloud Run deployment!")
            else:
                print(
                    "⚠️  Cloud Run simulation failed - but this is normal for local development"
                )
                print(
                    "   In actual Cloud Run, default credentials will work automatically"
                )
        else:
            print("\n⚠️  Some authentication tests failed.")

    except Exception as e:
        print(f"\n❌ Configuration error: {str(e)}")

    # Always show troubleshooting tips
    print("\n🔧 Troubleshooting tips:")
    print("1. For local development:")
    print("   - Run: gcloud auth application-default login")
    print("   - Or place service account key at: credentials/service-account-key.json")
    print("2. For Cloud Run:")
    print("   - Ensure the service account has BigQuery Data Viewer permissions")
    print("   - Ensure the service account has Storage Object Viewer permissions")
    print("   - Set GCP_PROJECT_ID environment variable if needed")
    print("3. Check that the dataset 'bgg_data_dev' exists and is accessible")


if __name__ == "__main__":
    main()
