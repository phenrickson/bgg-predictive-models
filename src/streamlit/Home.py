"""
Landing page for the BGG Models Dashboard.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.streamlit.components.footer import render_footer

st.set_page_config(page_title="BGG Models Dashboard", layout="wide")

st.title("BGG Models Dashboard")

st.markdown(
    """
This dashboard provides tools for exploring and analyzing board game predictions and model experiments:

### 🎲 Predictions
Explore board game predictions:
- View predictions for different publication years
- Analyze prediction distributions and trends
- Compare predictions across different metrics
- Monitor prediction jobs and their statistics

### 📊 Experiments
Monitor and analyze model experiments:
- Track model performance metrics across different datasets
- View detailed predictions for each experiment
- Explore feature importance and model parameters
- Access experiment metadata and details

### 🎯 Embeddings
Explore game embeddings from user ratings:
- View scree plots and explained variance
- Explore component loadings
- Visualize embeddings in 2D
- Search for games and find similar games

### 📝 Text Embeddings
Explore word embeddings learned from game descriptions:
- Search for words and find similar words
- Compare word similarities
- Generate document embeddings from descriptions
- Visualize word embeddings in 2D

Use the sidebar navigation to explore different sections of the dashboard.
"""
)

# Add footer with BGG logo
render_footer()
