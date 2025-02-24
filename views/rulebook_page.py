import streamlit as st
import os
import json
import io
import tempfile
import contextlib
from pathlib import Path
import pandas as pd

from components.alert import show_alert
from chunk_manager.rulebook_parser import parse_rulebook_excel, validate_rulebook_json

# Display alert if it exists in session state
if st.session_state.stored_alert:
    show_alert()

# Directory to store JSON rulebooks
JSON_DIR = Path(__file__).parent.parent / "_data" / "rulebooks" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

def process_file(uploaded_file):
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    
    # Validate conetnt and save to JSON
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

def get_rulebooks_list():
    # List only .json files in the JSON_DIR
    return [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

def format_value(val):
    # Helper to format numeric values to two decimal places
    if isinstance(val, (float, int)):
        return f"{val:.2f}"
    return val

# --- Streamlit Page Layout ---
st.title("Rulebooks")

# --- Upload Section ---
st.header("Upload a Rulebook")
uploaded_file = st.file_uploader("Choose an Excel or JSON file", type=["xlsx", "xls", "json"])

if st.button("Submit"):
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

st.markdown("---")

# --- Rulebook List & Display Section ---
st.header("Saved Rulebooks")
rulebooks = get_rulebooks_list()

if rulebooks:

    # Display rulebooks in a selectbox
    selected_rulebook = st.selectbox("Rulebook Selector", rulebooks)
    # Display delete button
    if selected_rulebook and st.button("Delete Selected Rulebook", icon="❌"):
        file_path = JSON_DIR / selected_rulebook
        if file_path.exists():
            os.remove(file_path)
            st.session_state.stored_alert = {
                'type': 'warning',
                'message': f"Rulebook {selected_rulebook} deleted successfully."
            }
            st.rerun()
        else:
            st.error("File not found.")

    # Load selected rulebook JSON
    file_path = JSON_DIR / selected_rulebook
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rulebook_json = json.load(f)
        
        st.subheader("Rulebook Details")
        # Display first three attributes in columns
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Review Item:** {rulebook_json.get('review_item', '')}")
        col2.markdown(f"**Collection Mode:** {rulebook_json.get('collection_mode', '')}")
        col3.markdown(f"**Total:** {rulebook_json.get('total', '')}")

        # Prepare content_rules table data with numeric formatting
        st.markdown("### Content Rules")
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
        df = pd.DataFrame(table_data)
        df.index = range(1, len(df) + 1)
        df.index.name = 'Rule Idx'
        df_styled = df.style.set_properties(**{'text-align': 'center'})
        st.write(df_styled)

        # Prepare collection ranges as a table
        st.markdown("### Collection Ranges")
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
        df_cr.index = range(1, len(df_cr) + 1)
        df_cr.index.name = 'Rule Idx'
        df_cr_styled = df_cr.style.set_properties(**{'text-align': 'center'})
        st.write(df_cr_styled)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
else:
    st.info("No saved rulebooks available.")
