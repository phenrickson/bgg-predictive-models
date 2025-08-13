import os
import json
from pathlib import Path
from typing import Optional, List

from google.cloud import storage
from src.models.experiments import Experiment, ExperimentTracker

def upload_experiment_to_gcs(
    experiment: Experiment, 
    bucket_name: str, 
    base_prefix: str = "models/experiments"
):
    """
    Upload a finalized experiment to Google Cloud Storage
    
    Args:
        experiment: Experiment object to upload
        bucket_name: GCS bucket name
        base_prefix: Base prefix for storing experiments
    
    Returns:
        GCS path of uploaded experiment
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Determine experiment details
    model_type = experiment.metadata.get('model_type', 'unknown')
    experiment_name = experiment.name
    
    # Find the finalized model directory
    finalized_dir = experiment.exp_dir / "finalized"
    
    if not finalized_dir.exists():
        raise ValueError(f"No finalized model found for experiment {experiment_name}")
    
    # Construct GCS path
    gcs_prefix = f"{base_prefix}/{model_type}/{experiment_name}/{finalized_dir.parent.name}"
    
    # Files to upload
    files_to_upload = [
        'pipeline.pkl',
        'info.json',
        '../metadata.json',
        '../parameters.json',
        '../model_info.json'
    ]
    
    # Upload each file
    uploaded_files = {}
    for relative_path in files_to_upload:
        file_path = finalized_dir.parent / relative_path
        
        if file_path.exists():
            # Determine GCS blob name
            blob_name = f"{gcs_prefix}/{file_path.name}"
            blob = bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_filename(str(file_path))
            uploaded_files[file_path.name] = blob_name
    
    return {
        'gcs_prefix': gcs_prefix,
        'uploaded_files': uploaded_files
    }

def upload_latest_experiments(
    model_types: Optional[List[str]] = None, 
    bucket_name: Optional[str] = None,
    base_dir: str = "models/experiments"
):
    """
    Upload the latest version of each experiment type to GCS
    
    Args:
        model_types: List of model types to upload (default: all)
        bucket_name: GCS bucket name (required)
        base_dir: Base directory for local experiments
    
    Returns:
        Dictionary of uploaded experiments
    """
    # Validate bucket name
    if not bucket_name:
        raise ValueError("Bucket name must be provided")
    
    # Default model types if not specified
    if model_types is None:
        model_types = ['hurdle', 'complexity', 'rating', 'users_rated']
    
    # Track uploaded experiments
    uploaded_experiments = {}
    
    for model_type in model_types:
        try:
            # Create experiment tracker
            tracker = ExperimentTracker(model_type, base_dir=base_dir)
            
            # List experiments
            experiments = tracker.list_experiments()
            
            if not experiments:
                print(f"No experiments found for {model_type}")
                continue
            
            # Get the latest experiment
            latest_experiment = max(
                experiments, 
                key=lambda x: x['version']
            )
            
            # Load the experiment
            experiment = tracker.load_experiment(
                latest_experiment['name'], 
                latest_experiment['version']
            )
            
            # Upload to GCS
            upload_result = upload_experiment_to_gcs(
                experiment, 
                bucket_name
            )
            
            uploaded_experiments[model_type] = {
                'experiment_name': latest_experiment['name'],
                'version': latest_experiment['version'],
                'gcs_path': upload_result['gcs_prefix']
            }
            
            print(f"Uploaded {model_type} experiment: {latest_experiment['name']} v{latest_experiment['version']}")
        
        except Exception as e:
            print(f"Error uploading {model_type} experiment: {e}")
    
    return uploaded_experiments

def main():
    """
    CLI entry point for uploading experiments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model experiments to Google Cloud Storage")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--model-types", nargs='+', help="Model types to upload")
    
    args = parser.parse_args()
    
    # Upload experiments
    uploaded = upload_latest_experiments(
        model_types=args.model_types,
        bucket_name=args.bucket
    )
    
    # Print results
    print("\nUpload Summary:")
    for model_type, details in uploaded.items():
        print(f"{model_type}: {details['experiment_name']} v{details['version']} -> {details['gcs_path']}")

if __name__ == "__main__":
    main()
