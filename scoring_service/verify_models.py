from google.cloud import storage
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def verify_model_registration(bucket_name, model_types, project_id=None):
    """
    Verify model registration in the specified bucket.

    Args:
        bucket_name (str): Name of the GCS bucket
        model_types (list): List of model types to verify
        project_id (str, optional): Google Cloud Project ID
    """
    # Use project_id if provided, otherwise try to get from environment
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT_ID")

    if not project_id:
        raise ValueError("No Google Cloud Project ID found. Set GCP_PROJECT_ID in .env")

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    print(f"Verifying models in bucket: {bucket_name}")
    print(f"Using Project ID: {project_id}")

    for model_type in model_types:
        print(f"\nChecking {model_type} models:")

        # Construct the prefix for the model type
        prefix = f"models/registered/{model_type}/"

        # List blobs with this prefix
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"  No models found for {model_type}")
            continue

        # Group blobs by registered name and version
        registered_models = {}
        for blob in blobs:
            # Split the blob path to extract registered name and version
            parts = blob.name.split("/")
            if len(parts) >= 5 and blob.name.endswith("registration.json"):
                registered_name = parts[3]
                version = parts[4]

                if registered_name not in registered_models:
                    registered_models[registered_name] = []
                registered_models[registered_name].append(version)

        # Print model details
        for name, versions in registered_models.items():
            print(f"  Registered Model: {name}")
            print(f"  Versions: {', '.join(versions)}")

            # Try to read registration details for the latest version
            latest_version = sorted(versions)[-1]
            registration_path = f"{prefix}{name}/{latest_version}/registration.json"

            try:
                registration_blob = bucket.blob(registration_path)
                registration_content = registration_blob.download_as_text()
                registration = json.loads(registration_content)

                print("  Registration Details:")
                print(
                    f"    Experiment: {registration.get('original_experiment', {}).get('name', 'N/A')}"
                )
                print(f"    Description: {registration.get('description', 'N/A')}")
                print(f"    Status: {registration.get('status', 'N/A')}")
                print(f"    Registered At: {registration.get('registered_at', 'N/A')}")

            except Exception as e:
                print(f"  Could not read registration details: {e}")


def main():
    # Get bucket name from environment variable
    bucket_name = os.getenv("GCS_BUCKET_NAME", "bgg-predictive-models-dev")

    # Model types to verify
    model_types = ["hurdle", "complexity", "rating", "users_rated"]

    verify_model_registration(bucket_name, model_types)


if __name__ == "__main__":
    main()
