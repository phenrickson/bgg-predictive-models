"""
Landing page for the BGG Models Dashboard.
"""

import streamlit as st

st.set_page_config(page_title="BGG Models Dashboard", layout="wide")

st.title("BGG Models Dashboard")

st.markdown(
    """
This dashboard provides tools for exploring and analyzing board game predictions and model experiments:

### 📊 Experiments
Monitor and analyze model experiments:
- Track model performance metrics across different datasets
- View detailed predictions for each experiment
- Explore feature importance and model parameters
- Access experiment metadata and details

### 🎲 Predictions
Explore board game predictions:
- View predictions for different publication years
- Analyze prediction distributions and trends
- Compare predictions across different metrics
- Monitor prediction jobs and their statistics

Use the sidebar navigation to explore different sections of the dashboard.
"""
)

# Add some metrics or visualizations to the landing page
st.divider()

# Add footer
st.markdown(
    """
---
Models developed by [Phil Henrickson](github.com/phenrickson/bgg-predictive-models)
"""
)
