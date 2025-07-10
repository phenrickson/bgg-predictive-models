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
    
    # Validate model type directory
    if not model_experiments_path.exists():
        st.error(f"Model type directory does not exist: {model_experiments_path}")
        return []
    
    # Iterate through experiment variants
    for experiment_variant in model_experiments_path.iterdir():
        if not experiment_variant.is_dir():
            st.warning(f"Skipping non-directory: {experiment_variant}")
            continue
        
        # Iterate through versions
        for version_dir in sorted(experiment_variant.iterdir(), key=lambda x: x.name):
            if not version_dir.is_dir():
                st.warning(f"Skipping non-directory version: {version_dir}")
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
                
                # Comprehensive file loading with detailed logging
                files_to_load = [
                    ('metadata', "metadata.json"),
                    ('parameters', "parameters.json"),
                    ('model_info', "model_info.json")
                ]
                
                for key, filename in files_to_load:
                    file_path = version_dir / filename
                    if file_path.exists():
                        try:
                            with open(file_path, "r") as f:
                                details[key] = json.load(f)
                        except json.JSONDecodeError as e:
                            st.warning(f"JSON decoding error in {filename} for {details['name']}: {e}")
                    else:
                        st.warning(f"No {key} found for {details['name']}")
                
                # Load metrics for different datasets
                for dataset in ['train', 'tune', 'test']:
                    metrics_path = version_dir / f"{dataset}_metrics.json"
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, "r") as f:
                                details['metrics'][dataset] = json.load(f)
                        except json.JSONDecodeError as e:
                            st.warning(f"JSON decoding error in {dataset}_metrics.json for {details['name']}: {e}")
                    else:
                        st.warning(f"No {dataset} metrics found for {details['name']}")
                
                # Load coefficients if available
                coef_path = version_dir / "coefficients.csv"
                if coef_path.exists():
                    try:
                        details['coefficients'] = pl.read_csv(coef_path)
                    except Exception as e:
                        st.warning(f"Error reading coefficients for {details['name']}: {e}")
                else:
                    st.warning(f"No coefficients found for {details['name']}")
                
                # Validate experiment details
                if not details['metrics'] and not details['parameters'] and not details['model_info']:
                    st.warning(f"Skipping experiment {details['name']} due to lack of data")
                    continue
                
                all_experiments.append(details)
            
            except Exception as e:
                st.warning(f"Could not load experiment {experiment_variant.name} (v{version_dir.name}): {e}")
    
    # Add comprehensive logging
    if not all_experiments:
        st.error(f"No experiments found for model type: {model_type}")
        st.info(f"Checked directory: {model_experiments_path}")
        st.info(f"Directory contents: {list(model_experiments_path.iterdir())}")
        st.info(f"Experiment variant contents: {[d.name for d in model_experiments_path.iterdir() if d.is_dir()]}")
    else:
        st.info(f"Loaded {len(all_experiments)} experiments")
        for exp in all_experiments:
            st.info(f"Experiment: {exp['name']}")
            st.info(f"Experiment details: {list(exp.keys())}")
    
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
    # Validate input
    if not hasattr(experiments, '__iter__'):
        st.error(f"Invalid experiments input: expected iterable, got {type(experiments)}")
        return pl.DataFrame()
    
    # Convert to list if needed
    if not isinstance(experiments, list):
        experiments = list(experiments)
    
    # Prepare parameters DataFrame
    params_data = []
    for exp in experiments:
        try:
            # Validate experiment is a dictionary
            if not isinstance(exp, dict):
                st.warning(f"Skipping non-dictionary experiment: {type(exp)}")
                continue
            
            # Get experiment name and version
            exp_name = str(exp.get('name', 'Unknown Experiment'))
            exp_version = str(exp.get('version', 'Unknown Version'))
            
            # Create base row
            row = {
                'Experiment': exp_name,
                'Version': exp_version
            }
            
            # Process parameters
            params = exp.get('parameters', {})
            if isinstance(params, dict):
                # Add direct parameters
                for key, value in params.items():
                    row[str(key)] = str(value)
            
            # Process model info
            model_info = exp.get('model_info', {})
            if isinstance(model_info, dict):
                # Add best parameters if available
                best_params = model_info.get('best_params', {})
                if isinstance(best_params, dict):
                    for key, value in best_params.items():
                        row[f"best_{key}"] = str(value)
                
                # Add other model info
                for key in ['threshold', 'threshold_f1_score', 'n_features', 'intercept']:
                    if key in model_info:
                        row[key] = str(model_info[key])
            
            params_data.append(row)
            
        except Exception as e:
            st.warning(f"Could not process experiment parameters: {e}")
            continue
    
    # Validate output
    if not params_data:
        st.warning("No valid experiment parameters found")
        return pl.DataFrame()
    
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
    """
    Create feature importance visualization with robust error handling and flexible extraction.
    
    Args:
        experiments (List[Dict]): List of experiment details.
    
    Returns:
        plotly.graph_objs._figure.Figure or None: Feature importance plot or None if no data.
    """
    # Validate input
    if not hasattr(experiments, '__iter__'):
        st.error(f"Invalid experiments input: expected iterable, got {type(experiments)}")
        return None
    
    # Convert to list if needed
    if not isinstance(experiments, list):
        experiments = list(experiments)
    
    # Detailed logging of input
    st.info(f"Total experiments: {len(experiments)}")
    for i, exp in enumerate(experiments):
        st.info(f"Experiment {i}: {type(exp)}")
        if isinstance(exp, dict):
            st.info(f"Experiment keys: {list(exp.keys())}")
    
    feature_data = []
    
    # Comprehensive column detection strategy
    def detect_columns(df):
        """Detect feature and coefficient columns with advanced heuristics."""
        # Comprehensive column name mappings
        feature_mappings = {
            'feature', 'features', 'name', 'names', 
            'variable', 'var', 'predictor', 'predictors'
        }
        coef_mappings = {
            'coefficient', 'coefficients', 'coef', 'coefs', 
            'weight', 'weights', 'importance', 'value', 'values'
        }
        
        # Case-insensitive column detection
        columns = {col.lower(): col for col in df.columns}
        
        feature_col = next((columns[col] for col in columns if col in feature_mappings), None)
        coef_col = next((columns[col] for col in columns if col in coef_mappings), None)
        
        return feature_col, coef_col
    
    # Logging for debugging
    missing_coef_experiments = []
    
    for exp in experiments:
        # Validate experiment is a dictionary
        if not isinstance(exp, dict):
            st.warning(f"Skipping non-dictionary experiment: {type(exp)}")
            continue
        
        try:
            # Robust coefficient extraction
            if 'coefficients' not in exp or exp['coefficients'] is None:
                missing_coef_experiments.append(str(exp.get('name', 'Unknown')))
                continue
            
            coef_df = exp['coefficients']
            
            # Skip empty DataFrames
            if coef_df.is_empty():
                missing_coef_experiments.append(str(exp.get('name', 'Unknown')))
                continue
            
            # Use standard column names
            feature_col = 'feature'
            coef_col = 'coefficient'
            
            # Validate columns exist
            if feature_col not in coef_df.columns or coef_col not in coef_df.columns:
                st.warning(f"Missing required columns in coefficients for {exp.get('name', 'Unknown')}")
                missing_coef_experiments.append(str(exp.get('name', 'Unknown')))
                continue
            
            # Extract data with type conversion and error handling
            for row in coef_df.iter_rows(named=True):
                try:
                    feature = str(row[feature_col])
                    coefficient = float(row[coef_col])
                    
                    feature_data.append({
                        'Experiment': str(exp.get('name', 'Unknown')),
                        'Feature': feature,
                        'Coefficient': coefficient
                    })
                except (ValueError, TypeError) as e:
                    st.warning(f"Data conversion error in {exp.get('name', 'Unknown')}: {e}")
        
        except Exception as e:
            st.warning(f"Error processing experiment {exp.get('name', 'Unknown')}: {e}")
    
    # Log experiments without coefficients
    if missing_coef_experiments:
        st.info(f"Experiments without coefficient data: {', '.join(missing_coef_experiments)}")
    
    # Create visualization if data exists
    if feature_data:
        try:
            df = pl.DataFrame(feature_data)
            
            # Aggregate coefficients by feature and experiment
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
                height=600,  # Increased height
                width=1000,  # Added width
                xaxis_tickangle=-45,  # Rotate labels for readability
                xaxis_tickfont=dict(size=8),  # Smaller font
                legend_title_text='Experiment',
                title_x=0.5,  # Center title
                hovermode='closest'
            )
            
            # Add hover data
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Coefficient: %{y:.4f}<br>Experiment: %{fullData.name}<extra></extra>'
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Visualization creation failed: {e}")
    
    st.warning("No feature importance data could be extracted.")
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