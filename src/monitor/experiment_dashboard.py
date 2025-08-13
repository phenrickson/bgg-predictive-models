"""Streamlit-based Experiment Tracking Dashboard."""
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objs as go
import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.experiments import ExperimentTracker

def load_all_experiment_details(model_type: str):
    """
    Load details for all experiments of a given model type with improved robustness.
    
    Handles the new directory structure:
    models/experiments/{model_type}/{experiment_variant}/{version}/
    """
    all_experiments = []
    
    # Path to the specific model type experiments
    model_experiments_path = Path("models/experiments") / model_type
    
    # Iterate through experiment variants
    for experiment_variant in model_experiments_path.iterdir():
        if not experiment_variant.is_dir():
            continue
        
        # Iterate through versions
        for version_dir in sorted(experiment_variant.iterdir(), key=lambda x: x.name):
            if not version_dir.is_dir():
                continue
            
            try:
                # Construct experiment details
                details = {
                    'name': f"{model_type}/{experiment_variant.name}",
                    'version': version_dir.name.replace('v', ''),
                    'metrics': {},
                    'parameters': {},
                    'model_info': {},
                    'metadata': {}
                }
                
                # Load metadata
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        details['metadata'] = json.load(f)
                
                # Load parameters
                params_path = version_dir / "parameters.json"
                if params_path.exists():
                    with open(params_path, "r") as f:
                        details['parameters'] = json.load(f)
                
                # Load model info (if exists)
                model_info_path = version_dir / "model_info.json"
                if model_info_path.exists():
                    with open(model_info_path, "r") as f:
                        details['model_info'] = json.load(f)
                
                # Load metrics for different datasets
                for dataset in ['train', 'tune', 'test']:
                    metrics_path = version_dir / f"{dataset}_metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path, "r") as f:
                            details['metrics'][dataset] = json.load(f)
                
                # Load coefficients if available
                coef_path = version_dir / "coefficients.csv"
                if coef_path.exists():
                    details['coefficients'] = pl.read_csv(coef_path)
                
                all_experiments.append(details)
            
            except Exception as e:
                st.warning(f"Could not load experiment {experiment_variant.name} (v{version_dir.name}): {e}")
    
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
        row = {
            'Experiment': exp['name'],
            'Version': exp['version']
        }
        
        # Add metrics for the selected dataset
        for metric in all_metrics:
            row[metric] = exp['metrics'].get(selected_dataset, {}).get(metric, 'N/A')
        
        metrics_data.append(row)
    
    return pl.DataFrame(metrics_data)

def create_parameters_overview(experiments):
    """Create a comprehensive parameters overview."""
    # Collect all unique parameters
    all_params = set()
    
    # Prepare parameters DataFrame
    params_data = []
    for exp in experiments:
        # Robust handling of experiment parameters
        try:
            # Ensure parameters is a dictionary
            params = exp.get('parameters', {})
            if not isinstance(params, dict):
                params = {}
            
            # Create base row with experiment info
            row = {
                'Experiment': str(exp.get('name', 'Unknown Experiment')),
                'Version': str(exp.get('version', 'Unknown Version'))
            }
            
            # Collect unique parameters
            all_params.update(params.keys())
            
            # Add parameters to row
            for param in all_params:
                row[param] = str(params.get(param, 'N/A'))
            
            params_data.append(row)
        
        except Exception as e:
            # Log error and skip problematic experiment
            st.warning(f"Could not process experiment parameters: {e}")
            continue
    
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
        # Check if coefficients exist and is a valid Polars DataFrame
        if 'coefficients' in exp and exp['coefficients'] is not None and not exp['coefficients'].is_empty():
            coef_df = exp['coefficients']
            
            # Flexible column name handling
            feature_col = None
            coef_col = None
            
            # Try common column names
            possible_feature_cols = ['feature', 'Feature', 'features', 'Features', 'name', 'Name']
            possible_coef_cols = ['coefficient', 'Coefficient', 'coef', 'Coef', 'value', 'Value']
            
            for col in coef_df.columns:
                if col in possible_feature_cols:
                    feature_col = col
                elif col in possible_coef_cols:
                    coef_col = col
            
            # If we found both columns, process the data
            if feature_col and coef_col:
                for row in coef_df.iter_rows(named=True):
                    feature_data.append({
                        'Experiment': exp['name'],
                        'Feature': row[feature_col],
                        'Coefficient': row[coef_col]
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
        
        # Customize layout
        fig.update_layout(
            height=600,  # Increase height
            xaxis_tickfont=dict(size=8)  # Reduce x-axis text size
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
        
        # Metrics Visualization
        st.header('Metrics Comparison')
        
        # Convert metrics DataFrame to long format for plotting
        metrics_long = metrics_df.melt(
            id_vars=['Experiment', 'Version'], 
            variable_name='Metric', 
            value_name='Value'
        )
        
        # Convert to pandas for Plotly
        metrics_pandas = metrics_long.to_pandas()
        
        # Remove 'N/A' values and convert to numeric
        metrics_pandas = metrics_pandas[metrics_pandas['Value'] != 'N/A']
        metrics_pandas['Value'] = pd.to_numeric(metrics_pandas['Value'], errors='coerce')
        
        # Drop rows with NaN values after conversion
        metrics_pandas = metrics_pandas.dropna(subset=['Value'])
        
        if not metrics_pandas.empty:
            # Get unique metrics
            unique_metrics = metrics_pandas['Metric'].unique()
            
            # Create subplot figure
            from plotly.subplots import make_subplots
            
            # Calculate number of rows needed
            num_rows = (len(unique_metrics) + 2) // 3  # 3 metrics per row
            
            # Create subplot figure
            fig = make_subplots(
                rows=num_rows, 
                cols=3, 
                subplot_titles=[m.upper() for m in unique_metrics],
                vertical_spacing=0.1
            )
            
            # Color palette
            color_palette = px.colors.qualitative.Plotly
            
            # Add box plots for each metric
            for i, metric in enumerate(unique_metrics):
                # Filter data for this metric
                metric_data = metrics_pandas[metrics_pandas['Metric'] == metric]
                
                # Determine row and column
                row = i // 3 + 1
                col = i % 3 + 1
                
                # Create box plot traces for each experiment
                experiments = metric_data['Experiment'].unique()
                for j, exp in enumerate(experiments):
                    exp_data = metric_data[metric_data['Experiment'] == exp]['Value']
                    
                    fig.add_trace(
                        go.Box(
                            y=exp_data, 
                            name=exp, 
                            marker_color=color_palette[j % len(color_palette)],
                            legendgroup=exp,
                            showlegend=i == 0  # Only show legend for first subplot
                        ),
                        row=row, 
                        col=col
                    )
            
            # Update layout
            fig.update_layout(
                #title_text=f'Metrics Comparison for {selected_dataset.capitalize()} Dataset',
                height=300 * num_rows,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(showticklabels=False)  # Remove x-axis labels
            fig.update_yaxes(title_text='')  # Remove y-axis title
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numeric data to create visualization")
    
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
            # Robust handling of experiment name
            try:
                exp_name = str(
                    exp.get('name') or 
                    exp.get('full_name') or 
                    exp.get('experiment_name') or 
                    'Unknown Experiment'
                )
                
                # Robust handling of model info
                model_info = exp.get('model_info', {})
                if not isinstance(model_info, dict):
                    model_info = {}
                
                with st.expander(f"{exp_name} Details"):
                    st.subheader('Model Info')
                    st.json(model_info)
            
            except Exception as e:
                st.warning(f"Could not process model information: {e}")
    
    with tab5:
        st.header('Experiment Metadata')
        for exp in experiments:
            # Robust handling of experiment name
            try:
                exp_name = str(
                    exp.get('name') or 
                    exp.get('full_name') or 
                    exp.get('experiment_name') or 
                    'Unknown Experiment'
                )
                
                # Robust handling of metadata
                metadata = exp.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                with st.expander(f"{exp_name} Metadata"):
                    st.json(metadata)
            
            except Exception as e:
                st.warning(f"Could not process experiment metadata: {e}")

if __name__ == '__main__':
    main()
