"""
Streamlit main application.
"""

import os

from pathlib import Path

# Get absolute path to .env file
current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'

# Check if environment variables are set
required_env_vars = ['OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")

import streamlit as st

# from utils.database import db_setup

# Initialize session state variables
if 'alert' not in st.session_state:
    st.session_state.alert = None

# Define pages

test_page = st.Page(
    page="views/test_view.py",
    title="Test",
    default=True
)

# Setup navigation
nav = st.navigation(pages=[test_page])
nav.run()
