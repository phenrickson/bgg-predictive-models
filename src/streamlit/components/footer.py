"""
Shared footer component for the BGG Models Dashboard.
"""

import streamlit as st
from pathlib import Path
import base64


def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def render_footer():
    """Render the footer with BGG logo and attribution."""
    # Get the path to the logo
    current_dir = Path(__file__).parent
    logo_path = current_dir.parent / "assets" / "bgg_logo.png"

    # Check if logo exists
    if logo_path.exists():
        # Convert image to base64 for embedding
        img_base64 = get_base64_image(logo_path)

        if img_base64:
            # Create footer with logo and attribution
            footer_html = f"""
            <div style="
                margin-top: 3rem;
                padding: 2rem 0 1rem 0;
                border-top: 1px solid #e0e0e0;
                text-align: center;
                background-color: transparent;
            ">
                <div style="margin-bottom: 1rem;">
                    <img src="data:image/png;base64,{img_base64}" 
                         alt="BoardGameGeek Logo" 
                         style="height: 60px; width: auto;">
                </div>
                <div style="
                    font-size: 0.9rem;
                    color: #666;
                    margin-bottom: 0.5rem;
                ">
                    Data sourced from <a href="https://boardgamegeek.com" target="_blank" style="color: #1f77b4; text-decoration: none;">BoardGameGeek</a>
                </div>
                <div style="
                    font-size: 0.8rem;
                    color: #888;
                ">
                    Models developed by <a href="https://github.com/phenrickson/bgg-predictive-models" target="_blank" style="color: #1f77b4; text-decoration: none;">Phil Henrickson</a>
                </div>
            </div>
            """

            st.markdown(footer_html, unsafe_allow_html=True)
        else:
            # Fallback footer without logo
            render_text_footer()
    else:
        # Fallback footer without logo
        render_text_footer()


def render_text_footer():
    """Render a text-only footer as fallback."""
    st.markdown(
        """
        ---
        <div style="text-align: center; margin-top: 2rem;">
            <div style="margin-bottom: 0.5rem;">
                Data sourced from <a href="https://boardgamegeek.com" target="_blank">BoardGameGeek</a>
            </div>
            <div style="font-size: 0.9rem; color: #666;">
                Models developed by <a href="https://github.com/phenrickson/bgg-predictive-models" target="_blank">Phil Henrickson</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
