import streamlit as st
import os
import json
import io
import tempfile
import contextlib
from pathlib import Path
import pandas as pd

from view_components.alert import show_alert
from view_components.saved_items_selector import saved_items_selector
from chunk_manager.rulebook_parser import parse_rulebook_excel, validate_rulebook_json, validate_rulebook_values
from utils.settings_manager import get_setting
from view_components.load_and_validate_json import load_and_validate_json

# Display alert if it exists in session state
if st.session_state.stored_alert:
    show_alert()

# Directory to store JSON rulebooks
RB_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'rulebooks_json')
RB_JSON_DIR.mkdir(parents=True, exist_ok=True)

def process_file(uploaded_file):
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    
    # Validate content and save to JSON
    captured_output = io.StringIO()
    result_path = None
    with contextlib.redirect_stdout(captured_output):
        if file_extension in ['.xlsx', '.xls', '.xlsm']:
            result_path = parse_rulebook_excel(tmp_path)
        elif file_extension == '.json':
            result_path = validate_rulebook_json(tmp_path)
        else:
            st.error("Unsupported file type!")
            return None, captured_output.getvalue()
    
        # Clean up the temporary file
        try:
            tmp_path.unlink()
        except Exception as e:
            print(f"Error deleting temporary file: {e}")
    
    return result_path, captured_output.getvalue()

def format_value(val):
    # Helper to format numeric values to two decimal places
    if isinstance(val, (float, int)):
        return f"{val:.2f}"
    return val

def handle_file_upload(uploaded_file):
    """Handle the file upload process and show appropriate messages"""
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            result_path, console_output = process_file(uploaded_file)
            if console_output:
                st.text_area("Console Output", console_output, height=200)
            if result_path is None:
                st.session_state.alert = {
                    'type': 'error',
                    'message': 'File processing failed. Check console output for details.'
                }
                st.error("File processing failed.")
            else:
                st.session_state.alert = {
                    'type': 'success',
                    'message': f"File processed successfully! Saved to {result_path}"
                }
                st.success(f"File processed successfully! Saved to {result_path}")
    else:
        st.error("Please upload a file first.")

def display_rulebook_data(rulebook_json):
    """Display rulebook data in a formatted way"""
    
    # Display rulebook metadata and collection ranges
    with st.container(border=True):
        # Split into two main columns
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            # Review Item at full width
            st.metric(label="Review Item", value=rulebook_json.get('review_item', ''))
            
            # Nested columns for Collection Mode and Total
            mode_col, total_col = st.columns(2)
            with mode_col:
                st.metric(label="Collection Mode", value=rulebook_json.get('collection_mode', ''))
            with total_col:
                st.metric(label="Total", value=rulebook_json.get('total', ''))
        
        # Prepare collection ranges as a table
        collection_ranges = rulebook_json.get("collection_ranges", [])
        cr_table = []
        for item in collection_ranges:
            range_val = item.get("range", [])
            if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
                start, end = range_val
            else:
                start, end = "", ""
            cr_table.append({
                "Range Start": start,
                "Range End": end,
                "Target Fraction": format_value(item.get("target_fraction"))
            })
        df_cr = pd.DataFrame(cr_table)
        if not df_cr.empty:
            df_cr.set_index("Target Fraction", inplace=True)
            
            with right_col:
                st.write("Collection Ranges")
                st.write(df_cr)

    # Prepare content_rules table data with numeric formatting
    st.write("Content Rules")
    table_data = []
    for topic, details in rulebook_json.get("content_rules", {}).items():
        sentiment = details.get("sentiment_proportion", [])
        sentiment_str = ', '.join([f"{x:.2f}" for x in sentiment]) if sentiment else ""
        row = {
            "Topic": topic,
            "Total Proportion": format_value(details.get("total_proportion")),
            "Sentiment Proportion": sentiment_str,
            "Chunk Min WC": details.get("chunk_min_wc"),
            "Chunk Max WC": details.get("chunk_max_wc"),
            "Chunk Pref": format_value(details.get("chunk_pref")),
            "Chunk WC Distribution": format_value(details.get("chunk_wc_distribution"))
        }
        table_data.append(row)
    df_ts = pd.DataFrame(table_data)
    if not df_ts.empty:
        df_ts.set_index("Topic", inplace=True)
        st.dataframe(df_ts)
        
    st.write("Note: The Content Rules table is vertically scrollable.")
    
    with st.expander("Rulebook Explanation", icon="❔"):
        st.write("TODO: Add explanation here.")

# --- Streamlit Page Layout ---
st.title("Rulebooks")

# --- Upload Section ---
st.subheader("Upload a Rulebook")

# Display file upload form
with st.expander("Import a rulebook from a file", icon="📁", expanded=True):
    uploaded_file = st.file_uploader("Choose an Excel or JSON file", type=["xlsx", "xls", "json"])

    if st.button("Submit", icon="✅"):
        handle_file_upload(uploaded_file)

# --- Rulebook List & Display Section ---
st.subheader("Saved Rulebooks")

# Display rulebook selector
selected_rulebook = saved_items_selector(RB_JSON_DIR, "Rulebook")

# Load and display selected rulebook
if selected_rulebook:
    
    # Load and validate the selected rulebook
    file_path = RB_JSON_DIR / selected_rulebook
    rulebook_json = load_and_validate_json(file_path, validate_rulebook_values)

    # Display the rulebook data
    if rulebook_json:
        st.info(f"{selected_rulebook}")
        display_rulebook_data(rulebook_json)
else:
    st.info("Select a rulebook to view its contents.")