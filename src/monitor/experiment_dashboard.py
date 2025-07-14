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

def load_experiments(model_type: str) -> List[Dict[str, Any]]:
    """
    Load experiments for a specific model type using ExperimentTracker.
    
    Args:
        model_type: Type of model to load experiments for
    
    Returns:
        List of experiment details
    """
    try:
        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()
        
        # Minimal diagnostic logging
        st.sidebar.info(f"Loaded {len(experiments)} experiments for {model_type}")
        
        # Enrich experiments with additional details
        enriched_experiments = []
        for exp in experiments:
            try:
                # Try to load additional experiment details
                full_exp = tracker.load_experiment(exp['name'], exp['version'])
                
                # Combine listed and loaded experiment details
                enriched_exp = {
                    **exp,
                    'metrics': {},
                    'parameters': {},
                    'model_info': {}
                }
                
                # Load metrics
                for dataset in ['train', 'tune', 'test']:
                    try:
                        enriched_exp['metrics'][dataset] = full_exp.get_metrics(dataset)
                    except ValueError:
                        pass
                
                # Load parameters and model info
                try:
                    enriched_exp['parameters'] = full_exp.get_parameters()
                    enriched_exp['model_info'] = full_exp.get_model_info()
                except ValueError:
                    pass
                
                enriched_experiments.append(enriched_exp)
            
            except Exception as e:
                st.sidebar.warning(f"Could not fully load experiment {exp['name']}: {e}")
                # Fallback to original experiment details
                enriched_experiments.append(exp)
        
        return enriched_experiments
    
    except Exception as e:
        st.sidebar.error(f"Error loading experiments: {e}")
        return []

def get_experiment_details(model_type: str, experiment_name: str, version: int):
    """
    Retrieve detailed information for a specific experiment.
    
    Args:
        model_type: Type of model
        experiment_name: Name of the experiment
        version: Version of the experiment
    
    Returns:
        Detailed experiment information
    """
    tracker = ExperimentTracker(model_type)
    experiment = tracker.load_experiment(experiment_name, version)
    
    # Collect comprehensive experiment details
    details = {
        'name': experiment.name,
        'version': version,
        'description': experiment.description,
        'timestamp': experiment.timestamp,
        'metadata': experiment.metadata,
        'metrics': {},
        'parameters': {},
        'model_info': {}
    }
    
    # Load metrics
    for dataset in ['train', 'tune', 'test']:
        try:
            details['metrics'][dataset] = experiment.get_metrics(dataset)
        except ValueError:
            st.warning(f"No metrics found for {dataset} dataset")
    
    # Load parameters
    try:
        details['parameters'] = experiment.get_parameters()
    except ValueError:
        st.warning("No parameters found for experiment")
    
    # Load model info
    try:
        details['model_info'] = experiment.get_model_info()
    except ValueError:
        st.warning("No model info found for experiment")
    
    return details

def create_metrics_overview(experiments: List[Dict[str, Any]], selected_dataset: str):
    """
    Create a comprehensive metrics overview for a specific dataset.
    
    Args:
        experiments: List of experiments
        selected_dataset: Dataset to analyze (train/tune/test)
    
    Returns:
        Polars DataFrame with metrics
    """
    # Collect all unique metrics
    all_metrics = set()
    for exp in experiments:
        metrics = exp.get('metrics', {}).get(selected_dataset, {})
        all_metrics.update(metrics.keys())
    
    # Prepare metrics data
    metrics_data = []
    for exp in experiments:
        row = {
            'Experiment': exp['full_name'],
            'Timestamp': exp.get('timestamp', 'Unknown')
        }
        
        # Add metrics for the selected dataset
        dataset_metrics = exp.get('metrics', {}).get(selected_dataset, {})
        for metric in all_metrics:
            row[metric] = dataset_metrics.get(metric, 'N/A')
        
        metrics_data.append(row)
    
    return pl.DataFrame(metrics_data)

def create_parameters_overview(experiments: List[Dict[str, Any]]):
    """
    Create a comprehensive parameters overview.
    
    Args:
        experiments: List of experiments
    
    Returns:
        Polars DataFrame with parameters
    """
    # Prepare parameters data
    params_data = []
    for exp in experiments:
        row = {
            'Experiment': exp['full_name'],
            'Timestamp': exp.get('timestamp', 'Unknown')
        }
        
        # Add parameters
        parameters = exp.get('parameters', {})
        for key, value in parameters.items():
            row[key] = str(value)
        
        # Add model info parameters
        model_info = exp.get('model_info', {})
        for key in ['n_features', 'intercept', 'threshold']:
            if key in model_info:
                row[f"model_info.{key}"] = str(model_info[key])
        
        params_data.append(row)
    
    return pl.DataFrame(params_data)

def create_feature_importance_plot(experiments: List[Dict[str, Any]]):
    """
    Create feature importance visualization.
    
    Args:
        experiments: List of experiments
    
    Returns:
        Plotly figure or None
    """
    feature_data = []
    
    for exp in experiments:
        try:
            # Load coefficients
            coef_path = Path(f"models/experiments/{exp['name']}/v{exp['version']}/coefficients.csv")
            if coef_path.exists():
                coef_df = pl.read_csv(coef_path)
                
                # Extract feature and coefficient data
                for row in coef_df.iter_rows(named=True):
                    feature_data.append({
                        'Experiment': exp['full_name'],
                        'Feature': row['feature'],
                        'Coefficient': float(row['coefficient'])
                    })
        except Exception as e:
            st.warning(f"Could not process coefficients for {exp['full_name']}: {e}")
    
    # Create visualization
    if feature_data:
        df = pl.DataFrame(feature_data)
        
        # Aggregate coefficients
        df_agg = (
            df.group_by(['Experiment', 'Feature'])
            .agg(pl.col('Coefficient').mean())
            .sort('Coefficient', descending=True)
        )
        
        # Convert to pandas for Plotly
        df_pandas = df_agg.to_pandas()
        
        # Create interactive bar plot
        fig = px.bar(
            df_pandas, 
            x='Feature', 
            y='Coefficient', 
            color='Experiment', 
            barmode='group',
            title='Feature Importance Across Experiments',
            labels={'Coefficient': 'Mean Coefficient'}
        )
        
        # Enhanced layout
        fig.update_layout(
            height=600,
            width=1000,
            xaxis_tickangle=-45,
            xaxis_tickfont=dict(size=8),
            legend_title_text='Experiment',
            title_x=0.5,
            hovermode='closest'
        )
        
        return fig
    
    return None

def main():
    """Streamlit app main function."""
    st.title('Experiment Tracking Dashboard')
    
    # Model Type Selection
    model_types = [d.name for d in Path("models/experiments").iterdir() if d.is_dir()]
    selected_model_type = st.sidebar.selectbox('Select Model Type', model_types)
    
    # Load experiments
    experiments = load_experiments(selected_model_type)
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Metrics Overview', 
        'Parameters', 
        'Feature Importance', 
        'Experiment Details',
        'Experiment Metadata'
    ])
    
    with tab1:
        st.header('Metrics Overview')
        
        # Dataset selector
        selected_dataset = st.selectbox(
            'Select Dataset', 
            ['train', 'tune', 'test']
        )
        
        # Metrics overview
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
        st.header('Experiment Details')
        
        # Select specific experiment
        selected_experiment = st.selectbox(
            'Select Experiment', 
            [exp['full_name'] for exp in experiments]
        )
        
        # Parse experiment name and version
        exp_name, exp_version = selected_experiment.split('/')
        exp_version = int(exp_version.replace('v', ''))
        
        # Load and display experiment details
        try:
            details = get_experiment_details(selected_model_type, exp_name, exp_version)
            st.json(details)
        except Exception as e:
            st.error(f"Could not load experiment details: {e}")
    
    with tab5:
        st.header('Experiment Metadata')
        for exp in experiments:
            with st.expander(f"{exp['full_name']} Metadata"):
                st.json(exp)

if __name__ == '__main__':
    main()
