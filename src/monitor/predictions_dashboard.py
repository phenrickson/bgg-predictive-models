"""Streamlit dashboard for monitoring model predictions."""
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

def format_predictions_table(df, threshold):
    """Format the predictions dataframe for better display."""
    # Prepare data with dynamic predicted class
    formatted_df = df.with_columns([
        pl.col("predicted_prob").round(3).alias("Probability"),
        (pl.col("predicted_prob") >= threshold).cast(pl.Int8).alias("Predicted"),
        pl.col("hurdle").alias("Actual"),
        pl.col("name").alias("Game Name"),
        pl.col("year_published").alias("Year"),
        pl.col("game_id").alias("Game ID")
    ]).to_pandas()
    
    # Create status indicators (simple text)
    def create_status_text(predicted, actual):
        if predicted == 1 and actual == 1:
            return "True Positive"
        elif predicted == 0 and actual == 0:
            return "True Negative"
        elif predicted == 1 and actual == 0:
            return "False Positive"
        else:
            return "False Negative"
    
    formatted_df['Status'] = formatted_df.apply(
        lambda row: create_status_text(row['Predicted'], row['Actual']), axis=1
    )
    
    return formatted_df[['Game ID', 'Game Name', 'Year', 'Probability', 'Predicted', 'Actual', 'Status']]

def plot_probability_distribution(df, threshold):
    """Create a histogram of prediction probabilities with overlapping distributions."""
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
        layout="wide",
        page_icon="🎲"
    )
    
    # Custom CSS for overall styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple title
    st.title("🎲 Game Hurdle Predictions Dashboard")
    
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
    
    # Calculate metrics with default threshold for initial display
    initial_threshold = 0.5
    metrics = calculate_metrics(df, initial_threshold)
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Games", metrics["total_games"])
    col2.metric("Games Above Threshold", metrics["games_above_threshold"])
    
    # Enhanced Predictions Table
    st.subheader("🎲 Game Predictions")
    
    # Add filtering and search options
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        search_term = st.text_input("🔍 Search Game Name", placeholder="Enter game name...")
    with col2:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "True Positives", "True Negatives", "False Positives", "False Negatives"]
        )
    with col3:
        year_filter = st.selectbox(
            "Year Published",
            ["All"] + sorted(df.select("year_published").unique().to_pandas()["year_published"].dropna().astype(int).tolist(), reverse=True)
        )
    with col4:
        min_prob = st.number_input("Min Probability", 0.0, 1.0, 0.0, 0.01)
    with col5:
        max_prob = st.number_input("Max Probability", 0.0, 1.0, 1.0, 0.01)
    
    # Format the table (use initial threshold for now, will be updated when user changes slider)
    formatted_table = format_predictions_table(df, initial_threshold)
    
    # Sort by probability in descending order by default
    formatted_table = formatted_table.sort_values('Probability', ascending=False)
    
    # Apply filters
    filtered_table = formatted_table.copy()
    
    # Filter by search term
    if search_term:
        filtered_table = filtered_table[
            filtered_table['Game Name'].str.contains(search_term, case=False, na=False)
        ]
    
    # Filter by status
    if filter_status != "All":
        status_map = {
            "True Positives": "True Positive",
            "True Negatives": "True Negative", 
            "False Positives": "False Positive",
            "False Negatives": "False Negative"
        }
        filtered_table = filtered_table[filtered_table['Status'] == status_map[filter_status]]
    
    # Filter by year
    if year_filter != "All":
        filtered_table = filtered_table[filtered_table['Year'] == year_filter]
    
    # Filter by probability range
    filtered_table = filtered_table[(filtered_table['Probability'] >= min_prob) & (filtered_table['Probability'] <= max_prob)]
    
    # Add custom CSS for better table styling
    st.markdown("""
    <style>
    .stDataFrame {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stDataFrame td {
        padding: 8px 12px !important;
        border-bottom: 1px solid #e0e0e0 !important;
    }
    .stDataFrame th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border-bottom: 2px solid #dee2e6 !important;
    }
    .stDataFrame tr:hover {
        background-color: #f8f9fa !important;
    }
    /* Force center alignment for probability column with multiple selectors */
    .stDataFrame td:nth-child(4),
    .stDataFrame th:nth-child(4),
    div[data-testid="stDataFrame"] td:nth-child(4),
    div[data-testid="stDataFrame"] th:nth-child(4),
    .dataframe td:nth-child(4),
    .dataframe th:nth-child(4) {
        text-align: center !important;
        justify-content: center !important;
    }
    /* Override any inline styles */
    .stDataFrame [style*="text-align"] {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display summary stats
    total_filtered = len(filtered_table)
    st.write(f"📊 Showing {total_filtered:,} games (filtered from {len(formatted_table):,} total)")
    
    # Display the enhanced table
    if len(filtered_table) > 0:
        # Pagination with buttons
        items_per_page = 25
        total_pages = (len(filtered_table) - 1) // items_per_page + 1
        
        # Initialize page state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        # Reset page if filters changed and current page is out of bounds
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = 1
        
        # Get current page data
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        display_table = filtered_table.iloc[start_idx:end_idx]
        
        # Style the dataframe with colors matching the probability distribution
        def style_probability(val):
            # Create gradient from light blue (0) to green (1) using dashboard colors
            # Light blue: #7FCDBB, Green: #28a745 (or use navy to light blue gradient)
            # Using a gradient from light blue to darker blue/green
            light_blue = [127, 205, 187]  # #7FCDBB
            dark_blue = [31, 78, 121]     # #1F4E79
            
            # Interpolate between light blue and dark blue based on probability
            r = int(light_blue[0] + (dark_blue[0] - light_blue[0]) * val)
            g = int(light_blue[1] + (dark_blue[1] - light_blue[1]) * val)
            b = int(light_blue[2] + (dark_blue[2] - light_blue[2]) * val)
            
            # Remove text-align from inline style to let CSS handle it
            return f'background-color: rgb({r}, {g}, {b}); vertical-align: middle !important; font-weight: bold; color: black; padding: 8px !important;'
        
        def style_prediction(val):
            if val == 1:
                return 'background-color: #7FCDBB; color: #000; font-weight: bold; text-align: center;'  # Light blue for hurdle
            else:
                return 'background-color: #1F4E79; color: white; font-weight: bold; text-align: center;'  # Navy for non-hurdle
        
        styled_table = display_table.style.map(style_probability, subset=['Probability']) \
                                         .map(style_prediction, subset=['Predicted', 'Actual']) \
                                         .format({'Probability': '{:.3f}'})
        
        st.dataframe(
            styled_table, 
            use_container_width=True, 
            hide_index=True
        )
        
        # Pagination controls underneath the table
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", disabled=st.session_state.current_page == 1):
                    st.session_state.current_page = 1
                    st.rerun()
            
            with col2:
                if st.button("◀️ Prev", disabled=st.session_state.current_page == 1):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col3:
                st.write(f"Page {st.session_state.current_page} of {total_pages}")
            
            with col4:
                if st.button("Next ▶️", disabled=st.session_state.current_page == total_pages):
                    st.session_state.current_page += 1
                    st.rerun()
            
            with col5:
                if st.button("Last ⏭️", disabled=st.session_state.current_page == total_pages):
                    st.session_state.current_page = total_pages
                    st.rerun()
            
            # Show pagination info
            start_item = start_idx + 1
            end_item = min(end_idx, len(filtered_table))
            st.caption(f"Showing items {start_item}-{end_item} of {total_filtered}")
    else:
        st.warning("No games match the current filters.")
    
    # Probability threshold slider
    threshold = st.slider(
        "Probability Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.15, 
        step=0.01
    )
    
    # Recalculate metrics with user-selected threshold
    metrics = calculate_metrics(df, threshold)
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    col4.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col5.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}")
    
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
