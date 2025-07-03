"""Compare results across different experiment runs."""
import polars as pl
from src.models.experiments import ExperimentTracker
from typing import List, Dict
from pathlib import Path

def load_experiment_metrics(experiment_name: str) -> Dict[str, Dict[str, float]]:
    """Load metrics for all datasets from an experiment."""
    tracker = ExperimentTracker()
    experiment = tracker.load_experiment(experiment_name)
    
    return {
        'train': experiment.get_metrics('train'),
        'tune': experiment.get_metrics('tune'),
        'test': experiment.get_metrics('test')
    }

def compare_experiments(experiment_names: List[str]):
    """Compare metrics across multiple experiments."""
    all_metrics = {}
    all_params = {}
    
    tracker = ExperimentTracker()
    
    for name in experiment_names:
        experiment = tracker.load_experiment(name)
        
        # Load metrics
        all_metrics[name] = load_experiment_metrics(name)
        
        # Load parameters
        all_params[name] = experiment.get_parameters()
    
    # Create comparison tables for each dataset
    for dataset in ['train', 'tune', 'test']:
        print(f"\n{dataset.upper()} Set Comparison:")
        print("-" * 80)
        
        # Header with experiment names
        header = f"{'Metric':20}"
        for name in experiment_names:
            header += f"{name:>15}"
        print(header)
        print("-" * 80)
        
        # Get all unique metrics across experiments
        all_metric_names = set()
        for exp_metrics in all_metrics.values():
            all_metric_names.update(exp_metrics[dataset].keys())
        
        # Print each metric
        for metric in sorted(all_metric_names):
            row = f"{metric:20}"
            for name in experiment_names:
                value = all_metrics[name][dataset].get(metric, float('nan'))
                row += f"{value:>15.4f}"
            print(row)
    
    # Compare parameters
    print("\nModel Parameters:")
    print("-" * 80)
    
    # Get all unique parameters
    all_param_names = set()
    for params in all_params.values():
        all_param_names.update(params.keys())
    
    # Print parameter comparison
    header = f"{'Parameter':20}"
    for name in experiment_names:
        header += f"{name:>15}"
    print(header)
    print("-" * 80)
    
    for param in sorted(all_param_names):
        row = f"{param:20}"
        for name in experiment_names:
            value = all_params[name].get(param, "N/A")
            row += f"{str(value):>15}"
        print(row)

def main():
    """Compare experiment results."""
    # Compare experiments with different forced regularization strengths
    experiments_to_compare = ['high_reg_force', 'low_reg_force']
    
    print("\nComparing experiments:", ", ".join(experiments_to_compare))
    compare_experiments(experiments_to_compare)

if __name__ == "__main__":
    main()
