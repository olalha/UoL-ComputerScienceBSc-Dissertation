"""
Streamlit main application.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Check if environment variables are set
required_env_vars = ['OPENROUTER_API_KEY', 'DATABASE_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")

import streamlit as st

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
