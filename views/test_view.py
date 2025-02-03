"""
Streamlit page for managing uploaded files.
"""

import streamlit as st

from components.alert import show_alert
from components.test_component import render_test_component

# Display alert if it exists in session state
if st.session_state.alert:
    show_alert()

st.title("Hello World!")

render_test_component()
