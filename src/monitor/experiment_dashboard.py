"""Streamlit-based Experiment Tracking Dashboard."""
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objs as go
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.experiments import ExperimentTracker

def load_all_experiment_details(model_type: str):
    """Load details for all experiments of a given model type."""
    tracker = ExperimentTracker(model_type)
    experiments_list = tracker.list_experiments()
    
    all_experiments = []
    for exp in experiments_list:
        try:
            # Extract name and version more flexibly
            full_name = exp['full_name'] if 'full_name' in exp else exp['name']
            version = exp['version']
            
            # Split full name to get base experiment name
            base_name = full_name.split('/')[0]
            
            # Try loading experiment with base name
            experiment = tracker.load_experiment(base_name, version)
            
            # Load metadata
            with open(Path("models/experiments") / model_type / base_name / f"v{version}" / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Debug: Print full metadata
            st.write(f"Metadata for {full_name}:", metadata)
            
            # Collect comprehensive experiment details
            details = {
                'name': full_name,
                'version': version,
                'metrics': {},
                'parameters': experiment.get_parameters(),
                'model_info': experiment.get_model_info(),
                'metadata': metadata
            }
            
            # Load metrics for different datasets
            for dataset in ['train', 'tune', 'test']:
                try:
                    details['metrics'][dataset] = experiment.get_metrics(dataset)
                except ValueError:
                    details['metrics'][dataset] = {}
            
            # Try to load coefficients if available
            try:
                details['coefficients'] = experiment.get_coefficients()
            except ValueError:
                details['coefficients'] = None
            
            all_experiments.append(details)
        except Exception as e:
            st.warning(f"Could not load experiment {exp.get('name', 'Unknown')} (v{exp.get('version', 'Unknown')}): {e}")
    
    return all_experiments

def create_metrics_overview(experiments, selected_dataset):
    """Create a comprehensive metrics overview for a specific dataset."""
    # Collect all unique metrics for the selected dataset
    all_metrics = set()
    for exp in experiments:
        all_metrics.update(exp['metrics'].get(selected_dataset, {}).keys())
    
    # Prepare metrics DataFrame
    metrics_data = []
    for exp in experiments:
        # Debug: print full metadata for each experiment
        st.write(f"Full metadata for {exp['name']}:", exp['metadata'])
        
        row = {
            'Experiment': exp['name'],
            'Version': exp['version']
        }
        
        # Try multiple ways to extract years and other metadata
        metadata_keys = [
            f'{selected_dataset}_years', 
            'training_years', 
            'years', 
            'year_range'
        ]
        
        for key in metadata_keys:
            years = exp['metadata'].get(key)
            if years:
                row[f'{selected_dataset.capitalize()} Years'] = years
                break
        else:
            row[f'{selected_dataset.capitalize()} Years'] = 'N/A'
        
        # Add other metadata
        metadata_mappings = {
            'Model Type': ['model_type', 'model', 'type'],
            'Preprocessor': ['preprocessor', 'preprocessing', 'prep']
        }
        
        for display_key, possible_keys in metadata_mappings.items():
            for key in possible_keys:
                value = exp['metadata'].get(key)
                if value:
                    row[display_key] = value
                    break
            else:
                row[display_key] = 'N/A'
        
        # Add metrics for the selected dataset
        for metric in all_metrics:
            row[metric] = exp['metrics'].get(selected_dataset, {}).get(metric, 'N/A')
        
        metrics_data.append(row)
    
    return pl.DataFrame(metrics_data)

def create_parameters_overview(experiments):
    """Create a comprehensive parameters overview."""
    # Collect all unique parameters
    all_params = set()
    for exp in experiments:
        all_params.update(exp['parameters'].keys())
    
    # Prepare parameters DataFrame
    params_data = []
    for exp in experiments:
        row = {
            'Experiment': exp['name'],
            'Version': exp['version']
        }
        
        # Add parameters
        for param in all_params:
            row[param] = exp['parameters'].get(param, 'N/A')
        
        params_data.append(row)
    
    return pl.DataFrame(params_data)

def compare_experiments(experiments):
    """Compare experiments and highlight similarities/differences."""
    if len(experiments) < 2:
        return None
    
    # Compare metadata
    metadata_comparison = {}
    for exp in experiments:
        metadata_comparison[exp['name']] = {}
        for key, value in exp['metadata'].items():
            metadata_comparison[exp['name']][key] = value
    
    # Compare parameters
    parameter_comparison = {}
    for exp in experiments:
        parameter_comparison[exp['name']] = exp['parameters']
    
    # Compare metrics
    metrics_comparison = {}
    for exp in experiments:
        metrics_comparison[exp['name']] = {}
        for dataset in ['train', 'tune', 'test']:
            metrics_comparison[exp['name']][dataset] = exp['metrics'].get(dataset, {})
    
    return {
        'metadata': metadata_comparison,
        'parameters': parameter_comparison,
        'metrics': metrics_comparison
    }

def create_feature_importance_plot(experiments):
    """Create feature importance visualization."""
    feature_data = []
    for exp in experiments:
        if exp['coefficients'] is not None:
            coef_df = exp['coefficients']
            for row in coef_df.iter_rows(named=True):
                feature_data.append({
                    'Experiment': exp['name'],
                    'Feature': row['feature'],
                    'Coefficient': row['coefficient']
                })
    
    if feature_data:
        df = pl.DataFrame(feature_data)
        fig = px.bar(
            df.to_pandas(), 
            x='Feature', 
            y='Coefficient', 
            color='Experiment', 
            barmode='group',
            title='Feature Importance Across Experiments'
        )
        return fig
    return None

def main():
    """Streamlit app main function."""
    st.title('Experiment Tracking Dashboard')
    
    # Model Type Selection
    model_types = [d.name for d in Path("models/experiments").iterdir() if d.is_dir()]
    selected_model_type = st.sidebar.selectbox('Select Model Type', model_types)
    
    # Load all experiments
    experiments = load_all_experiment_details(selected_model_type)
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Metrics Overview', 
        'Parameters', 
        'Feature Importance', 
        'Model Information',
        'Experiment Metadata'
    ])
    
    with tab1:
        st.header('Metrics Overview')
        
        # Dataset selector
        selected_dataset = st.selectbox(
            'Select Dataset', 
            ['train', 'tune', 'test']
        )
        
        # Metrics overview for selected dataset
        metrics_df = create_metrics_overview(experiments, selected_dataset)
        st.dataframe(metrics_df)
    
    with tab2:
        st.header('Model Parameters')
        params_df = create_parameters_overview(experiments)
        st.dataframe(params_df)
    
    with tab3:
        st.header('Feature Importance')
        feature_fig = create_feature_importance_plot(experiments)
        if feature_fig:
            st.plotly_chart(feature_fig)
        else:
            st.write("No feature importance data available")
    
    with tab4:
        st.header('Model Information')
        for exp in experiments:
            with st.expander(f"{exp['name']} Details"):
                st.subheader('Model Info')
                st.json(exp['model_info'])
    
    with tab5:
        st.header('Experiment Metadata')
        for exp in experiments:
            with st.expander(f"{exp['name']} Metadata"):
                st.json(exp['metadata'])

if __name__ == '__main__':
    main()
