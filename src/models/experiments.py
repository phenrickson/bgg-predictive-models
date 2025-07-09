"""Experiment tracking and management for model training."""
import json
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

def compute_hash(data: Dict[str, Any]) -> str:
    """Compute a hash of the experiment configuration and data.
    
    Args:
        data: Dictionary containing experiment configuration
        
    Returns:
        Hash string
    """
    # Convert data to a stable string representation
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()[:8]

class ExperimentTracker:
    """Tracks and manages machine learning experiments."""
    
    def __init__(
        self, 
        model_type: str,
        base_dir: Union[str, Path] = "models/experiments"
    ):
        """Initialize experiment tracker.
        
        Args:
            model_type: Type of model being tracked (e.g., 'hurdle', 'rating')
            base_dir: Base directory for storing experiments
        """
        self.base_dir = Path(base_dir)
        self.model_type = model_type
        self.model_dir = self.base_dir / model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def create_experiment(
        self, 
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> "Experiment":
        """Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            metadata: Optional metadata dictionary
            version: Optional version number. If None, auto-increments from existing versions.
            config: Optional configuration dictionary for tracking experiment details
            
        Returns:
            Experiment object
        """
        # Compute hash of experiment configuration
        exp_config = {
            'name': name,
            'description': description,
            'metadata': metadata or {},
            'config': config or {},
            'model_type': self.model_type
        }
        exp_hash = compute_hash(exp_config)
        
        # Find existing experiment directories
        experiment_dir = self.model_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing versions and determine next version
        existing_versions = [
            int(p.name[1:])
            for p in experiment_dir.iterdir() 
            if p.is_dir() and p.name.startswith('v') and p.name[1:].isdigit()
        ]
        
        if version is None:
            version = max(existing_versions, default=0) + 1
        elif version in existing_versions:
            raise ValueError(f"Version {version} already exists for experiment {name}")
        
        # Create version subdirectory
        versioned_dir = experiment_dir / f"v{version}"
        versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # Add hash to metadata
        if metadata is None:
            metadata = {}
        metadata['hash'] = exp_hash
        metadata['config_hash'] = exp_hash  # For backwards compatibility
        metadata['config'] = config or {}
        
        return Experiment(name, versioned_dir, description, metadata)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments for this model type with their versions."""
        experiments = []
        for exp_dir in self.model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Get all version subdirectories
            versions = [
                (int(v.name[1:]), v) 
                for v in exp_dir.iterdir() 
                if v.is_dir() and v.name.startswith('v') and v.name[1:].isdigit()
            ]
            
            # If no versions, skip
            if not versions:
                continue
            
            # Sort versions
            versions.sort(key=lambda x: x[0])
            
            for version, version_dir in versions:
                # Load metadata if available
                metadata_file = version_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                
                experiments.append({
                    'name': exp_dir.name,
                    'version': version,
                    'full_name': f"{exp_dir.name}/v{version}",
                    'description': metadata.get('description'),
                    'timestamp': metadata.get('timestamp')
                })
        
        return sorted(experiments, key=lambda x: (x['name'], x['version']))
    
    def load_experiment(self, name: str, version: Optional[int] = None) -> "Experiment":
        """Load an existing experiment.
        
        Args:
            name: Name of the experiment to load
            version: Optional specific version to load. If None, loads latest version.
            
        Returns:
            Experiment object
        """
        # Find the experiment directory
        experiment_dir = self.model_dir / name
        
        if not experiment_dir.exists():
            raise ValueError(f"No experiment found matching '{name}'")
        
        # Find version subdirectories
        versions = [
            int(v.name[1:]) 
            for v in experiment_dir.iterdir() 
            if v.is_dir() and v.name.startswith('v') and v.name[1:].isdigit()
        ]
        
        if not versions:
            raise ValueError(f"No versions found for experiment '{name}'")
        
        # If version is not specified, find the latest
        if version is None:
            version = max(versions)
        elif version not in versions:
            raise ValueError(f"Version {version} not found for experiment '{name}'")
        
        # Select the version directory
        version_dir = experiment_dir / f"v{version}"
        
        # Load metadata
        metadata_file = version_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Create and return experiment
        return Experiment(
            name=name,  # Use base experiment name
            base_dir=version_dir,  # Use specific version directory
            description=metadata.get("description"),
            metadata=metadata.get("metadata", {})
        )

class Experiment:
    """Represents a single experiment run."""
    
    def __init__(
        self,
        name: str,
        base_dir: Path,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize experiment.
        
        Args:
            name: Name of the experiment
            base_dir: Base directory for all experiments
            description: Optional description
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.exp_dir = base_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.description = description
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        
        # Save initial metadata
        self._save_metadata()
        
    def _save_metadata(self):
        """Save experiment metadata to file."""
        metadata = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
        
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Dict[str, float]], dataset: str):
        """Log metrics for a specific dataset.
        
        Args:
            metrics: Dictionary of metrics
            dataset: Dataset name (e.g., 'train', 'tune', 'test')
        """
        metrics_file = self.exp_dir / f"{dataset}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters.
        
        Args:
            params: Dictionary of parameters
        """
        params_file = self.exp_dir / "parameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information.
        
        Args:
            model_info: Dictionary containing model information
        """
        info_file = self.exp_dir / "model_info.json"
        with open(info_file, "w") as f:
            json.dump(model_info, f, indent=2)
            
    def save_pipeline(self, pipeline: Pipeline) -> None:
        """Save the complete sklearn pipeline.
        
        Args:
            pipeline: Complete sklearn pipeline including preprocessing and model
        """
        pipeline_path = self.exp_dir / "pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
    
    def load_pipeline(self) -> Pipeline:
        """Load the complete sklearn pipeline.
        
        Returns:
            Complete sklearn pipeline including preprocessing and model
        """
        pipeline_path = self.exp_dir / "pipeline.pkl"
        if not pipeline_path.exists():
            raise ValueError(f"No pipeline found for experiment {self.name}")
            
        with open(pipeline_path, "rb") as f:
            return pickle.load(f)
    
    def finalize_model(
        self,
        X: Any,
        y: Any,
        description: str = "Finalized model for production use",
        final_end_year: Optional[int] = None,
        sample_weight: Optional[Any] = None
    ) -> Path:
        """Create production version by fitting pipeline on full dataset.
        
        Args:
            X: Features to fit on
            y: Target to fit on
            description: Description of finalized model
            final_end_year: Final year of data used in model training
            sample_weight: Optional sample weights for fitting
            
        Returns:
            Path to finalized model directory
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Load and clone pipeline
        original_pipeline = self.load_pipeline()
        pipeline = clone(original_pipeline)
        
        # Diagnostic logging for original pipeline
        logger.info("Original Pipeline Steps:")
        for name, step in original_pipeline.named_steps.items():
            logger.info(f"  Step: {name}, Type: {type(step)}")
            
            # Try to get feature names and scaling details
            try:
                feature_names = step.get_feature_names_out()
                logger.info(f"    Feature Names: {feature_names[:10]}")
                logger.info(f"    Total Feature Names Count: {len(feature_names)}")
            except Exception as e:
                logger.info(f"    Could not extract feature names: {e}")
            
            # Check for scaling-related attributes
            try:
                if hasattr(step, 'scale_'):
                    logger.info(f"    Scale: {step.scale_}")
                if hasattr(step, 'mean_'):
                    logger.info(f"    Mean: {step.mean_}")
                if hasattr(step, 'var_'):
                    logger.info(f"    Variance: {step.var_}")
            except Exception as e:
                logger.info(f"    Could not extract scaling details: {e}")
            
            # Additional diagnostic for preprocessor steps
            if hasattr(step, 'named_steps'):
                logger.info("    Preprocessor Sub-Steps:")
                for sub_name, sub_step in step.named_steps.items():
                    logger.info(f"      Sub-Step: {sub_name}, Type: {type(sub_step)}")
                    try:
                        sub_feature_names = sub_step.get_feature_names_out()
                        logger.info(f"        Sub-Step Feature Names: {sub_feature_names[:10]}")
                        logger.info(f"        Sub-Step Total Feature Names Count: {len(sub_feature_names)}")
                    except Exception as e:
                        logger.info(f"        Could not extract sub-step feature names: {e}")
        
        # Detailed input data diagnostics
        logger.info("\nInput Data Diagnostics:")
        logger.info(f"  Input Features Shape: {X.shape}")
        logger.info(f"  Target Shape: {y.shape}")
        logger.info(f"  Target Type: {type(y)}")
        logger.info(f"  Target Range: min={y.min()}, max={y.max()}")
        logger.info(f"  Target Mean: {y.mean()}")
        logger.info(f"  Target Std Dev: {y.std()}")
        
        # Fit on full dataset with optional sample weights
        if sample_weight is not None:
            logger.info("\nFitting with Sample Weights:")
            logger.info(f"  Sample Weight Shape: {sample_weight.shape}")
            logger.info(f"  Sample Weight Range: min={sample_weight.min()}, max={sample_weight.max()}")
            logger.info(f"  Sample Weight Mean: {sample_weight.mean()}")
            pipeline.fit(X, y, **{'sample_weight': sample_weight})
        else:
            pipeline.fit(X, y)
        
        # Diagnostic logging for fitted pipeline
        logger.info("\nFitted Pipeline Steps:")
        for name, step in pipeline.named_steps.items():
            logger.info(f"  Step: {name}, Type: {type(step)}")
            
            # Try to get feature names and scaling details
            try:
                if hasattr(step, 'get_feature_names_out'):
                    feature_names = step.get_feature_names_out()
                    logger.info(f"    Feature Names: {feature_names[:10]}")
                
                # Check for scaling-related attributes
                if hasattr(step, 'scale_'):
                    logger.info(f"    Scale: {step.scale_}")
                if hasattr(step, 'mean_'):
                    logger.info(f"    Mean: {step.mean_}")
                if hasattr(step, 'var_'):
                    logger.info(f"    Variance: {step.var_}")
                
                # Additional diagnostic for preprocessor steps
                if hasattr(step, 'named_steps'):
                    logger.info("    Preprocessor Sub-Steps:")
                    for sub_name, sub_step in step.named_steps.items():
                        logger.info(f"      Sub-Step: {sub_name}, Type: {type(sub_step)}")
                        try:
                            if hasattr(sub_step, 'get_feature_names_out'):
                                sub_feature_names = sub_step.get_feature_names_out()
                                logger.info(f"        Sub-Step Feature Names: {sub_feature_names[:10]}")
                        except Exception as e:
                            logger.info(f"        Could not extract sub-step feature names: {e}")
            except Exception as e:
                logger.info(f"    Could not extract details: {e}")
        
        # Save to finalized directory
        finalized_dir = self.exp_dir / "finalized"
        finalized_dir.mkdir(exist_ok=True)
        
        # Save fitted pipeline
        pipeline_path = finalized_dir / "pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
            
        # Save info
        info = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "experiment_name": self.name,
            "experiment_hash": self.metadata.get("hash"),
            "model_type": self.metadata.get("model_type"),
            "target": self.metadata.get("target"),
            "final_end_year": final_end_year
        }
        
        info_path = finalized_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
            
        return finalized_dir
    
    def load_finalized_model(self) -> Pipeline:
        """Load the finalized pipeline.
        
        Returns:
            Complete sklearn pipeline ready for predictions
            
        Raises:
            ValueError: If no finalized model exists
        """
        finalized_dir = self.exp_dir / "finalized"
        if not finalized_dir.exists():
            raise ValueError(f"No finalized model found for experiment {self.name}")
            
        pipeline_path = finalized_dir / "pipeline.pkl"
        if not pipeline_path.exists():
            raise ValueError(f"Pipeline not found in finalized directory")
            
        with open(pipeline_path, "rb") as f:
            return pickle.load(f)
    
    def log_coefficients(self, coefficients_df: pl.DataFrame):
        """Log model coefficients.
        
        Args:
            coefficients_df: DataFrame containing coefficient information
        """
        coef_file = self.exp_dir / "coefficients.csv"
        coefficients_df.write_csv(coef_file)
    
    def get_metrics(self, dataset: str) -> Dict[str, float]:
        """Get metrics for a specific dataset.
        
        Args:
            dataset: Dataset name (e.g., 'train', 'tune', 'test')
            
        Returns:
            Dictionary of metrics
        """
        metrics_file = self.exp_dir / f"{dataset}_metrics.json"
        if not metrics_file.exists():
            raise ValueError(f"No metrics found for dataset '{dataset}'")
        
        with open(metrics_file) as f:
            return json.load(f)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        params_file = self.exp_dir / "parameters.json"
        if not params_file.exists():
            raise ValueError("No parameters found")
        
        with open(params_file) as f:
            return json.load(f)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        info_file = self.exp_dir / "model_info.json"
        if not info_file.exists():
            raise ValueError("No model info found")
        
        with open(info_file) as f:
            return json.load(f)
    
    def get_coefficients(self) -> pl.DataFrame:
        """Get model coefficients.
        
        Returns:
            DataFrame containing coefficient information
        """
        coef_file = self.exp_dir / "coefficients.csv"
        if not coef_file.exists():
            raise ValueError("No coefficients found")
        
        return pl.read_csv(coef_file)
