"""Streamlit dashboard for monitoring model predictions."""
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

def load_predictions(file_path):
    """Load predictions from a Parquet file."""
    return pl.read_parquet(file_path)

def calculate_metrics(df, threshold):
    """Calculate comprehensive model performance metrics."""
    # Prepare data with updated predicted class based on threshold
    df_with_threshold = df.with_columns(
        predicted_class_dynamic=(df["predicted_prob"] >= threshold).cast(pl.Int8)
    )
    
    # Convert to pandas for sklearn metrics
    pdf = df_with_threshold.to_pandas()
    
    from sklearn.metrics import (
        precision_score, 
        recall_score, 
        f1_score, 
        accuracy_score, 
        balanced_accuracy_score
    )
    
    return {
        "total_games": len(df),
        "total_hurdle_games": int(df.select("hurdle").to_numpy().sum()),
        "games_above_threshold": int((df.filter(pl.col("predicted_prob") >= threshold)).height),
        
        # Model performance metrics
        "precision": precision_score(pdf["hurdle"], pdf["predicted_class_dynamic"]),
        "recall": recall_score(pdf["hurdle"], pdf["predicted_class_dynamic"]),
        "f1_score": f1_score(pdf["hurdle"], pdf["predicted_class_dynamic"]),
        "accuracy": accuracy_score(pdf["hurdle"], pdf["predicted_class_dynamic"]),
        "balanced_accuracy": balanced_accuracy_score(pdf["hurdle"], pdf["predicted_class_dynamic"])
    }

def plot_probability_distribution(df, threshold):
    """Create a histogram of prediction probabilities with overlapping distributions."""
    import pandas as pd
    
    # Prepare data with actual class
    pdf = df.to_pandas()
    
    # Create plot with overlapping distributions
    fig = px.histogram(
        pdf, 
        x="predicted_prob", 
        color="hurdle",
        color_discrete_map={0: '#1F4E79', 1: '#7FCDBB'},  # Navy for non-hurdle, light blue for hurdle
        title="Distribution of Predicted Probabilities",
        labels={
            "predicted_prob": "Predicted Probability", 
            "hurdle": "Actual Class"
        },
        marginal="box",
        barmode='overlay',  # Overlay the histograms
        opacity=0.5,  # Set transparency
        histnorm='density'  # Use density instead of count
    )
    
    # Add vertical line for threshold
    fig.add_vline(
        x=threshold, 
        line_dash="dash", 
        line_color="black", 
        annotation_text=f"Threshold: {threshold:.2f}", 
        annotation_position="top right"
    )
    
    return fig

def plot_confusion_matrix(df, threshold):
    """Create a confusion matrix based on the current threshold."""
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix
    
    # Prepare data
    pdf = df.to_pandas()
    pdf['predicted_class_dynamic'] = (pdf['predicted_prob'] >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(pdf['hurdle'], pdf['predicted_class_dynamic'])
    
    # Create heatmap with a subtle blue color scale
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',  # Subtle blue color scale
        zmin=0,  # Ensure color starts from white
        showscale=False,  # Hide color scale
        text=cm.astype(str),  # Display count values
        texttemplate='%{text}',  # Show text on each cell
        textfont={'color':'black', 'size':20}  # Make text visible and larger
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Game Hurdle Predictions Dashboard", 
        layout="wide"
    )
    
    st.title("Game Hurdle Predictions Dashboard")
    
    # Automatically load the latest predictions file
    import os
    predictions_dir = "data/predictions"
    
    # Get all parquet files in the directory
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.parquet')]
    
    if not prediction_files:
        st.error("No prediction files found in data/predictions")
        return
    
    # Sort files by modification time, most recent first
    latest_file = max(
        [os.path.join(predictions_dir, f) for f in prediction_files], 
        key=os.path.getmtime
    )
    
    # Load predictions
    df = pl.read_parquet(latest_file)
    
    st.write(f"Loaded predictions from: {os.path.basename(latest_file)}")
    
    # Probability threshold slider
    threshold = st.slider(
        "Probability Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01
    )
    
    # Calculate metrics
    metrics = calculate_metrics(df, threshold)
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Games", metrics["total_games"])
    col2.metric("Games Above Threshold", metrics["games_above_threshold"])
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    col4.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col5.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}")
    
    # Predictions table with pagination and dynamic predicted class
    predictions_df = df.with_columns(
        predicted_prob=pl.col("predicted_prob").round(3),
        predicted_class_dynamic=(pl.col("predicted_prob") >= threshold).cast(pl.Int8)
    ).to_pandas()
    
    # Use Streamlit's AgGrid for advanced table features
    import streamlit.components.v1 as components
    from st_aggrid import AgGrid, GridOptionsBuilder
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(predictions_df[['game_id', 'name', 'year_published', 'predicted_prob', 'predicted_class_dynamic', 'hurdle']])
    gb.configure_pagination(paginationPageSize=25)  # 25 games per page
    gb.configure_column("predicted_prob", type=["numericColumn", "numberColumnFilter"], precision=3)
    gb.configure_selection('single')  # Enable row selection
    gridOptions = gb.build()
    
    # Display the grid
    ag_response = AgGrid(
        predictions_df[['game_id', 'name', 'year_published', 'predicted_prob', 'predicted_class_dynamic', 'hurdle']], 
        gridOptions=gridOptions, 
        enable_enterprise_modules=True,
        height=400,
        width='100%'
    )
    
    # Probability distribution and confusion matrix side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Probability Distribution")
        st.plotly_chart(plot_probability_distribution(df, threshold), use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(df, threshold), use_container_width=True)

if __name__ == "__main__":
    main()
