import os
import json
import io
import contextlib
import streamlit as st
from pathlib import Path

from view_components.alert import show_alert
from utils.settings_manager import get_setting
from chunk_manager.rulebook_parser import validate_rulebook_values
from dataset_manager.dataset_structurer import create_dataset_structure

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
    print("Datasets View: START SUBPROCESS - Validate Rulebooks")
    for rulebook in all_rulebooks:
        file_path = RB_JSON_DIR / rulebook
        with open(file_path, "r", encoding="utf-8") as f:
            if validate_rulebook_values(json.load(f)):
                valid_rulebooks.append(rulebook)
            else:
                print(f"Datasets View: Invalid rulebook: {rulebook}")
    print("Datasets View: END SUBPROCESS - Validate Rulebooks")
    return valid_rulebooks

# TODO: This needs validation and START/END SUBPROCESS print statements
def get_datasets_list():
    # List only .json files in the DS_JSON_DIR
    return [f for f in os.listdir(DS_JSON_DIR) if f.endswith('.json')]

def delete_dataset(selected_dataset):
    """Delete a dataset and handle the related messages"""
    file_path = DS_JSON_DIR / selected_dataset
    if file_path.exists():
        os.remove(file_path)
        st.session_state.stored_alert = {
            'type': 'warning',
            'message': f"Dataset {selected_dataset} deleted successfully."
        }
        return True
    else:
        st.error("File not found.")
        return False

def rename_dataset(selected_dataset, current_name, new_name):
    """Rename a dataset and handle the related messages"""
    if new_name != current_name:
        new_filename = f"{new_name}.json"
        new_path = DS_JSON_DIR / new_filename
        
        # Check for duplicate names
        if new_path.exists():
            st.error(f"A dataset named '{new_filename}' already exists!")
            return False
        else:
            # Rename file
            old_path = DS_JSON_DIR / selected_dataset
            old_path.rename(new_path)
            st.session_state.stored_alert = {
                'type': 'success',
                'message': f"Dataset renamed from '{selected_dataset}' to '{new_filename}'."
            }
            return True
    return False

# --- Streamlit Page Layout ---
st.title("Datasets")

# --- Generate Structure Section ---
st.subheader("Generate Dataset Structure")

# Display geneate dataset structure form
rulebooks = get_rulebooks_list()
if rulebooks:
    
    # Display rulebooks in a selectbox and search time slider
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
            
    if st.session_state.form_submitted:
        selected_rulebook = st.session_state.selected_rulebook
        solution_search_time_s = st.session_state.solution_search_time_s
        
        # Display a loading message and generate dataset structure
        captured_output = io.StringIO()
        with st.spinner("Generating dataset structure. Please wait...", show_time=True):
            
            # Read the selected rulebook
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

# There are no rulebooks available
else:
    
    # Display an alert and stop the script
    st.info("No rulebooks found. Please upload a rulebook first.")
    st.stop()

# --- Saved Datasets Section ---
st.subheader("Saved Datasets")
datasets = get_datasets_list()

if datasets:
    # Display dataset selection and delete button
    with st.container(border=True):
        selected_dataset = st.selectbox("Selected Dataset", datasets, key="dataset_selector")
        
        # Add delete button
        if selected_dataset:
            if st.button("Delete Dataset", icon="❌"):
                if delete_dataset(selected_dataset):
                    st.rerun()

            # Add rename form
            current_name = os.path.splitext(selected_dataset)[0]
            with st.form("rename_dataset_form", enter_to_submit=False):
                new_name = st.text_input("Rename Dataset", value=current_name, key="new_dataset_name")
                submitted = st.form_submit_button("Rename", icon="💾")
            
            if submitted:
                if rename_dataset(selected_dataset, current_name, new_name):
                    st.rerun()

    # Display dataset content
    if selected_dataset:
        st.markdown("##### Rulebook Contents")
        file_path = DS_JSON_DIR / selected_dataset
        with open(file_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
        st.info(f"{selected_dataset}")
        st.json(dataset_json)
else:
    st.info("No saved datasets available.")