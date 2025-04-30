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

# Initialize session state
from view_components.file_loader import initialize_file_cache
initialize_file_cache()

# Define pages
rulebook_page = st.Page(
    page="views/rulebook_page.py",
    title="Rulebooks",
    icon="📚"
)
dataset_page = st.Page(
    page="views/dataset_page.py",
    title="Datasets",
    icon="📊"
)

# Setup navigation
nav = st.navigation([rulebook_page, dataset_page])
nav.run()
