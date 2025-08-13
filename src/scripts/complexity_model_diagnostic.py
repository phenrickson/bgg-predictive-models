"""Diagnostic script for complexity model preprocessing and feature preservation."""
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.experiments import ExperimentTracker

def setup_logging():
    """Configure logging for the diagnostic script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)

def load_latest_complexity_model():
    """Load the latest complexity experiment model."""
    logger = logging.getLogger(__name__)
    
    # Explicitly set the base directory
    base_dir = Path('models/experiments/complexity/test_complexity')
    logger.info(f"Base Directory: {base_dir.absolute()}")
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    # List all versions
    versions = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    
    if not versions:
        raise ValueError("No versions found in test_complexity directory")
    
    # Select the latest version
    latest_version = max(versions, key=lambda x: int(x.name[1:]))
    
    logger.info(f"Selected Version: {latest_version}")
    
    # List directory contents
    logger.info("Directory Contents:")
    for item in latest_version.iterdir():
        logger.info(f"  {item} ({'directory' if item.is_dir() else 'file'})")
    
    # Try multiple potential paths for the model
    potential_paths = [
        latest_version / 'experiment_model.pkl',
        latest_version / 'finalized' / 'pipeline.pkl',
        latest_version / 'pipeline.pkl'
    ]
    
    for model_path in potential_paths:
        logger.info(f"Trying path: {model_path}")
        if model_path.exists():
            logger.info(f"Found model at: {model_path}")
            return joblib.load(model_path)
    
    # If no path works, provide detailed error
    raise FileNotFoundError(f"""
    Could not find experiment model in {latest_version}
    Tried paths:
    {chr(10).join(str(path) for path in potential_paths)}
    """)

def inspect_preprocessing_pipeline(pipeline):
    """Inspect and log details about the preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("Preprocessing Pipeline Steps:")
    for name, step in pipeline.named_steps.items():
        logger.info(f"  Step: {name}")
        logger.info(f"    Type: {type(step)}")
        
        # Try to get feature names
        try:
            feature_names = step.get_feature_names_out()
            logger.info(f"    Feature Names Count: {len(feature_names)}")
            logger.info(f"    First 10 Feature Names: {feature_names[:10]}")
        except Exception as e:
            logger.info(f"    Could not get feature names: {e}")

def diagnostic_prediction(pipeline):
    """Perform diagnostic predictions to verify model behavior."""
    logger = logging.getLogger(__name__)
    
    # Load a sample dataset for testing
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Load recent data for testing
    df = loader.load_training_data(
        end_train_year=2025, 
        min_weights=10,
        min_ratings=0
    )
    
    # Convert to pandas
    df_pandas = df.to_pandas()
    
    # Remove target column for prediction
    X = df_pandas.drop('complexity', axis=1)
    
    logger.info("\nDiagnostic Prediction:")
    logger.info(f"Input Features Shape: {X.shape}")
    
    # Predict
    raw_predictions = pipeline.predict(X)
    
    logger.info("Raw Prediction Diagnostics:")
    logger.info(f"  Range: min={raw_predictions.min():.4f}, max={raw_predictions.max():.4f}")
    logger.info(f"  Mean: {raw_predictions.mean():.4f}")
    logger.info(f"  Std Dev: {raw_predictions.std():.4f}")
    
    # First 10 predictions
    logger.info("First 10 Raw Predictions:")
    for i, pred in enumerate(raw_predictions[:10], 1):
        logger.info(f"  Prediction {i}: {pred:.4f}")

def extract_coefficients(pipeline):
    """Extract feature names and coefficients from a pipeline."""
    logger = logging.getLogger(__name__)
    
    # Get preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    # Try to get feature names from the final step of the preprocessor
    try:
        # Get the last step of the preprocessor
        last_step_name = list(preprocessor.named_steps.keys())[-1]
        last_step = preprocessor.named_steps[last_step_name]
        
        logger.info(f"Extracting feature names from final preprocessor step: {last_step_name}")
        
        # Get feature names from the final step
        feature_names = last_step.get_feature_names_out()
        
        # Get coefficients
        coefficients = regressor.coef_
        
        # Diagnostic logging
        logger.info(f"Feature names count: {len(feature_names)}")
        logger.info(f"Coefficients count: {len(coefficients)}")
        
        # If slight mismatch, truncate to the smaller length
        if len(feature_names) != len(coefficients):
            logger.warning(f"Mismatch in feature names ({len(feature_names)}) and coefficients ({len(coefficients)})")
            min_length = min(len(feature_names), len(coefficients))
            feature_names = feature_names[:min_length]
            coefficients = coefficients[:min_length]
        
        # Create DataFrame with coefficients
        import pandas as pd
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    except Exception as e:
        logger.error(f"Error extracting coefficients: {e}")
        raise

def main():
    logger = setup_logging()
    
    try:
        # Load the test_complexity experiment model
        test_complexity_model = load_latest_complexity_model()
        
        # Load the baseline_complexity finalized model
        import joblib
        finalized_model_path = Path('models/experiments/complexity/baseline_complexity/v3/finalized/pipeline.pkl')
        logger.info(f"Loading finalized model from: {finalized_model_path}")
        finalized_model = joblib.load(finalized_model_path)
        
        # Extract coefficients
        logger.info("\nTest Complexity Model Coefficients:")
        test_complexity_coef = extract_coefficients(test_complexity_model)
        logger.info(test_complexity_coef.head(10).to_string())
        
        logger.info("\nFinalized Model Coefficients:")
        finalized_coef = extract_coefficients(finalized_model)
        logger.info(finalized_coef.head(10).to_string())
        
        # Compare top features
        logger.info("\nTop Feature Comparison:")
        top_test = set(test_complexity_coef.head(10)['feature'])
        top_finalized = set(finalized_coef.head(10)['feature'])
        
        logger.info("Top 10 Test Complexity Features:")
        logger.info(", ".join(top_test))
        
        logger.info("\nTop 10 Finalized Features:")
        logger.info(", ".join(top_finalized))
        
        logger.info("\nCommon Top Features:")
        logger.info(", ".join(top_test.intersection(top_finalized)))
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
