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
        
        # Check for existing experiments with same hash
        for exp_dir in self.model_dir.glob(f"{name}_*"):
            if not exp_dir.is_dir():
                continue
            
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    existing_metadata = json.load(f)
                    if existing_metadata.get('hash') == exp_hash:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Found existing experiment with same configuration: {exp_dir.name}")
        
        # Find existing versions and determine next version
        existing_versions = [
            int(p.name.split('_v')[-1].split('_')[0])  # Handle hash suffix
            for p in self.model_dir.glob(f"{name}_v*")
            if p.name.split('_v')[-1].split('_')[0].isdigit()
        ]
        
        if version is None:
            version = max(existing_versions, default=0) + 1
        elif version in existing_versions:
            raise ValueError(f"Version {version} already exists for experiment {name}")
        
        versioned_name = f"{name}_v{version}_{exp_hash}"
        
        # Add hash to metadata
        if metadata is None:
            metadata = {}
        metadata['hash'] = exp_hash
        metadata['config_hash'] = exp_hash  # For backwards compatibility
        return Experiment(versioned_name, self.model_dir, description, metadata)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments for this model type with their versions."""
        experiments = []
        for exp_dir in self.model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # More flexible parsing of experiment names
            name_parts = exp_dir.name.split('_')
            
            # Try to find a version and hash
            version = None
            base_name = exp_dir.name
            
            for i, part in enumerate(name_parts):
                if part.startswith('v') and part[1:].isdigit():
                    version = int(part[1:])
                    base_name = '_'.join(name_parts[:i] + name_parts[i+1:])
                    break
            
            # If no version found, skip
            if version is None:
                continue
            
            # Load metadata if available
            metadata_file = exp_dir / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            
            experiments.append({
                'name': base_name,
                'version': version,
                'full_name': exp_dir.name,
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
        # More flexible experiment name matching
        matching_experiments = [
            p for p in self.model_dir.iterdir() 
            if p.is_dir() and name in p.name
        ]
        
        if not matching_experiments:
            raise ValueError(f"No experiments found matching '{name}'")
        
        # If version is not specified, find the latest
        if version is None:
            # Extract versions from matching experiments
            versions = []
            for exp_path in matching_experiments:
                try:
                    # More flexible version extraction
                    name_parts = exp_path.name.split('_')
                    for part in name_parts:
                        if part.startswith('v') and part[1:].isdigit():
                            versions.append(int(part[1:]))
                            break
                except (IndexError, ValueError):
                    continue
            
            if not versions:
                raise ValueError(f"No versions found for experiment '{name}'")
            version = max(versions)
        
        # Find the exact experiment matching version
        matching_version_exps = [
            p for p in matching_experiments 
            if f'_v{version}_' in p.name
        ]
        
        if not matching_version_exps:
            raise ValueError(f"Version {version} not found for experiment '{name}'")
        
        # Use the first matching experiment
        exp_dir = matching_version_exps[0]
        versioned_name = exp_dir.name
        
        # Load metadata
        metadata_file = exp_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Create and return experiment
        return Experiment(
            name=versioned_name,  # Use full name including version and hash
            base_dir=self.model_dir,  # Use model-specific directory
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
        self.exp_dir = base_dir / name
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
        description: str = "Finalized model for production use"
    ) -> Path:
        """Create production version by fitting pipeline on full dataset.
        
        Args:
            X: Features to fit on
            y: Target to fit on
            description: Description of finalized model
            
        Returns:
            Path to finalized model directory
        """
        # Load and clone pipeline
        pipeline = clone(self.load_pipeline())
        
        # Fit on full dataset
        pipeline.fit(X, y)
        
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
            "target": self.metadata.get("target")
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
