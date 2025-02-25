import streamlit as st
import os
import json
import io
import tempfile
import contextlib
from pathlib import Path
import pandas as pd

from view_components.alert import show_alert
from chunk_manager.rulebook_parser import parse_rulebook_excel, validate_rulebook_json, validate_rulebook_values
from utils.settings_manager import get_setting

# Directory to store JSON rulebooks
RB_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'rulebooks_json')
RB_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Display alert if it exists in session state
if st.session_state.stored_alert:
    show_alert()

def get_rulebooks_list():
    # List only .json files in the RB_JSON_DIR
    return [f for f in os.listdir(RB_JSON_DIR) if f.endswith('.json')]

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

def rename_rulebook(selected_rulebook, current_name, new_name):
    """Rename a rulebook and handle the related messages"""
    if new_name != current_name:
        new_filename = f"{new_name}.json"
        new_path = RB_JSON_DIR / new_filename
        
        # Check for duplicate names
        if new_path.exists():
            st.error(f"A rulebook named '{new_filename}' already exists!")
            return False
        else:
            # Rename file
            old_path = RB_JSON_DIR / selected_rulebook
            old_path.rename(new_path)
            st.session_state.stored_alert = {
                'type': 'success',
                'message': f"Rulebook renamed from '{selected_rulebook}' to '{new_filename}'."
            }
            return True
    return False

def delete_rulebook(selected_rulebook):
    """Delete a rulebook and handle the related messages"""
    file_path = RB_JSON_DIR / selected_rulebook
    if file_path.exists():
        os.remove(file_path)
        st.session_state.stored_alert = {
            'type': 'warning',
            'message': f"Rulebook {selected_rulebook} deleted successfully."
        }
        return True
    else:
        st.error("File not found.")
        return False

def display_rulebook_data(rulebook_json):
    """Display rulebook data in a formatted way"""
    
    # Display rulebook metadata and collection ranges
    with st.container(border=True):
        st.metric(label="Review Item", value=rulebook_json.get('review_item', ''))
        
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

        # Display collection mode, total and collection ranges
        col1, col2, col3 = st.columns([1,1,2])
        col1.metric(label="Collection Mode", value=rulebook_json.get('collection_mode', ''))
        col2.metric(label="Total", value=rulebook_json.get('total', ''))
        if not df_cr.empty:
            col3.write("Collection Ranges")
            col3.write(df_cr)

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

def load_and_validate_rulebook(selected_rulebook):
    """Load and validate the selected rulebook"""
    file_path = RB_JSON_DIR / selected_rulebook
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rulebook_json = json.load(f)
            
        # Validate rulebook values
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            valid = validate_rulebook_values(rulebook_json)
        
        if valid:
            st.info(f"{selected_rulebook}")
            display_rulebook_data(rulebook_json)
            
            with st.expander("Rulebook Explanation", icon="❔"):
                st.write("TODO: Add explanation here.")
        else:
            st.error("Rulebook values are invalid - Please delete and re-upload the corrected rulebook.")
            st.text_area("Console Output", captured_output.getvalue(), height=200)
            
    except Exception as e:
        st.error(f"Error loading JSON: {e}")

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
rulebooks = get_rulebooks_list()

# Display rulebook selection if available
if rulebooks:
    with st.container(border=True):
        selected_rulebook = st.selectbox("Selected Rulebook", rulebooks)
        
        # Add delete and button
        if selected_rulebook:
            if st.button("Delete Selected Rulebook", icon="❌"):
                if delete_rulebook(selected_rulebook):
                    st.rerun()

            # Add rename form
            current_name = os.path.splitext(selected_rulebook)[0]
            with st.form("rename_rulebook_form", enter_to_submit=False):
                new_name = st.text_input("Rename Rulebook", value=current_name)
                submitted = st.form_submit_button("Save", icon="💾")
            
            if submitted:
                if rename_rulebook(selected_rulebook, current_name, new_name):
                    st.rerun()

    # Load and display selected rulebook
    st.markdown("##### Rulebook Contents")
    load_and_validate_rulebook(selected_rulebook)
else:
    st.info("No saved rulebooks available.")
