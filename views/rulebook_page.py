import streamlit as st
import pandas as pd

from typing import Union, Any
from view_components.file_loader import load_and_validate_rulebook, process_rulebook_upload
from view_components.alerter import show_alert
from view_components.item_selector import saved_file_selector, add_new_file_and_select

# Display alert if it exists in session state
if 'stored_alert' in st.session_state and st.session_state.stored_alert:
    show_alert()

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.subheader(f"This page has run {st.session_state.counter} times.")

def upload_file_form() -> None:
    """ Display file upload interface with processing functionality. """
    
    with st.expander("Import a rulebook from a file", icon="📁", expanded=True):
        # File uploader component
        uploaded_file = st.file_uploader("Choose an Excel or JSON file", type=["xlsx", "xls", "json"])

        # Process the file when submitted
        if st.button("Submit", icon="✅"):
            if uploaded_file is not None:
                with st.spinner("Processing file..."):
                    # Process the uploaded file
                    result_path, console_output = process_rulebook_upload(uploaded_file)
                    
                    # Display console output if any
                    if console_output:
                        st.text_area("Console Output", console_output, height=200)
                    
                    # Handle upload result
                    if result_path:
                        add_new_file_and_select(result_path.name, 'rulebook')
                    else:
                        st.error("File processing failed.")
            else:
                st.error("Please upload a file first.")

def display_rulebook_data(rulebook_json: dict) -> None:
    """ Display rulebook data in a formatted way. """
    
    # Helper function to format numeric values
    def format_value(val: Union[float, int, Any]) -> str:
        if isinstance(val, (float, int)):
            return f"{val:.2f}"
        return val
    
    # Display rulebook metadata and collection ranges in a bordered container
    with st.container(border=True):
        # Split layout into two equal columns
        left_col, right_col = st.columns([1, 1])
        
        # Extract collection mode with empty string as default
        collection_mode = rulebook_json.get('collection_mode', '')
        
        # Display metadata in left column
        with left_col:
            # Display content title with full width
            st.metric(label="Content title", value=rulebook_json.get('content_title', ''))
            
            # Create nested columns for collection mode and total
            mode_col, total_col = st.columns(2)
            with mode_col:
                st.metric(label="Collection Mode", value=collection_mode)
            with total_col:
                st.metric(label="Total", value=rulebook_json.get('total', ''))
        
        # Process collection ranges data for tabular display
        collection_ranges = rulebook_json.get("collection_ranges", [])
        cr_table = []
        for item in collection_ranges:
            # Extract range values safely
            range_val = item.get("range", [])
            if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
                start, end = range_val
            else:
                start, end = "", ""
            
            # Build table row with formatted values
            cr_table.append({
                "Range Start": start,
                "Range End": end,
                "Target Fraction": format_value(item.get("target_fraction"))
            })
        
        # Create DataFrame from collection ranges data
        df_cr = pd.DataFrame(cr_table)
        if not df_cr.empty:
            # Set target fraction as index for better display
            df_cr.set_index("Target Fraction", inplace=True)
            
            # Display collection ranges in right column
            with right_col:
                st.write(f"Collection Ranges ({collection_mode} count)")
                st.write(df_cr)

    # Display content rules section
    st.write("Content Rules")
    
    # Process content rules data for tabular display
    table_data = []
    for topic, details in rulebook_json.get("content_rules", {}).items():
        # Format sentiment proportion values as comma-separated string
        sentiment = details.get("sentiment_proportion", [])
        sentiment_str = ', '.join([f"{x:.2f}" for x in sentiment]) if sentiment else ""
        
        # Build table row with topic details and formatted values
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
    
    # Create DataFrame from content rules data
    df_ts = pd.DataFrame(table_data)
    if not df_ts.empty:
        # Set topic as index for better organization
        df_ts.set_index("Topic", inplace=True)
        # Display dataframe with content rules
        st.dataframe(df_ts)
    
    # Add note about table scrollability
    st.write("Note: The Content Rules table is vertically scrollable.")
    
    # Add expandable section for rulebook explanation
    with st.expander("Rulebook Explanation", icon="❔"):
        st.write("TODO: Add explanation here.")

# --- Streamlit Page Layout ---
st.title("Rulebooks")

# --- Upload Section ---
st.header("Upload a Rulebook")

# Display the file upload form
upload_file_form()

# --- Rulebook List & Display Section ---
st.header("Saved Rulebooks")

# Display rulebook selector
selected_rulebook = saved_file_selector('rulebook')

# Load and display selected rulebook
if selected_rulebook:
    
    # Load and validate the selected rulebook
    rulebook_data, console_output = load_and_validate_rulebook(selected_rulebook)
    if console_output:
        st.text_area("Console Output", console_output, height=200)
    
    # Display the rulebook data
    if rulebook_data:
        st.markdown("---")
        st.info(f"{selected_rulebook}")
        display_rulebook_data(rulebook_data)
else:
    st.info("Select a rulebook to view its contents.")