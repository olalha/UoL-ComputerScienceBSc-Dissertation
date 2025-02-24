"""
Alert Component

This component can be added as part of a Streamlit page. 
It displays an alert from the session state and clears it.
"""	

import streamlit as st

def show_alert():
    # Display alert if present in session state
    if st.session_state.stored_alert:
        alert_type = st.session_state.stored_alert['type']
        message = st.session_state.stored_alert['message']
        getattr(st, alert_type)(message)
    
    # Clear the alert
    st.session_state.stored_alert = None