import os
import json
import io
import contextlib
import streamlit as st
from pathlib import Path

from view_components.alert import show_alert
from view_components.saved_items_selector import saved_items_selector
from view_components.load_and_validate_json import load_and_validate_json
from utils.settings_manager import get_setting
from chunk_manager.rulebook_parser import validate_rulebook_values
from dataset_manager.dataset_structurer import create_dataset_structure, validate_dataset_structure

# Display alert if it exists in session state
if st.session_state.stored_alert:
    show_alert()

# Initialize session state for form submission
if 'structure_form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Directory to store JSON rulebooks
RB_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'rulebooks_json')
RB_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Directory to store JSON datasets
DS_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'datasets_json')
DS_JSON_DIR.mkdir(parents=True, exist_ok=True)

def get_rulebooks_list():
    all_rulebooks = [f for f in os.listdir(RB_JSON_DIR) if f.endswith('.json')]
    valid_rulebooks = []
    print("Datasets View: START SUBPROCESS - Get validated rulebooks for generation")
    for rulebook in all_rulebooks:
        file_path = RB_JSON_DIR / rulebook
        with open(file_path, "r", encoding="utf-8") as f:
            if validate_rulebook_values(json.load(f)):
                valid_rulebooks.append(rulebook)
            else:
                print(f"Datasets View: Invalid rulebook: {rulebook}")
    print("Datasets View: END SUBPROCESS - Get validated rulebooks for generation")
    return valid_rulebooks

def generate_dataset_structure_form(rulebooks):
    """ Displays a form for generating dataset structures from rulebooks. """
    
    with st.expander("Generate Dataset From Rulebook", icon="📚", expanded=True):
        with st.form(key="generate_dataset_form", border=False):
            selected_rulebook = st.selectbox("Rulebook Selector", rulebooks)
            st.write("Warning: Invalid rulebooks will not be displayed.")
            solution_search_time_s = st.slider("Solution Search Time (seconds)", min_value=1, max_value=60, value=5)
            submitted = st.form_submit_button("Generate Dataset Structure")

            if submitted:
                st.session_state.form_submitted = True
                st.session_state.selected_rulebook = selected_rulebook
                st.session_state.solution_search_time_s = solution_search_time_s

    # When the form is submitted, generate the dataset structure
    if st.session_state.form_submitted:
        selected_rulebook = st.session_state.selected_rulebook
        solution_search_time_s = st.session_state.solution_search_time_s

        # Display a loading message and generate dataset structure
        captured_output = io.StringIO()
        with st.spinner("Generating dataset structure. Please wait...", show_time=True):

            # Read the selected rulebook (integrity already validated)
            file_path = RB_JSON_DIR / selected_rulebook
            with open(file_path, "r", encoding="utf-8") as f:
                selected_rulebook = json.load(f)

            # Generate dataset structure
            print("Datasets View: START SUBPROCESS - Generate Dataset Structure")
            with contextlib.redirect_stdout(captured_output):
                result_path = create_dataset_structure(rulebook=selected_rulebook, solution_search_time_s=solution_search_time_s)
            print("Datasets View: END SUBPROCESS - Generate Dataset Structure")

        # Display dataset structure
        if result_path:
            st.success(f"File processed successfully! Saved to {result_path}")
        else:
            st.error("Failed to generate dataset structure. Please try again.")
            st.text_area("Console Output", captured_output.getvalue(), height=200)

# --- Streamlit Page Layout ---
st.title("Datasets")

# --- Generate Structure Section ---
st.subheader("Generate Dataset Structure")

# Display generate dataset structure form
rulebooks = get_rulebooks_list()
if rulebooks:
    generate_dataset_structure_form(rulebooks)
else:
    st.info("No rulebooks found. Please upload a rulebook first.")

# --- Saved Datasets Section ---
st.subheader("Saved Datasets")

# Display dataset selector
selected_dataset = saved_items_selector(DS_JSON_DIR, "Dataset")

# Display selected dataset
if selected_dataset:
    
    # Validate and load the selected dataset
    file_path = DS_JSON_DIR / selected_dataset
    dataset_json = load_and_validate_json(file_path, validate_dataset_structure)

    # Display dataset content
    if dataset_json:
        st.info(f"{selected_dataset}")
        st.json(dataset_json)
else:
    st.info("Generate and select a dataset to view its content.")