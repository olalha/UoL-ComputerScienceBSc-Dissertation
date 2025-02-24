import os
import json
import streamlit as st
from pathlib import Path

from components.alert import show_alert
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
JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'rulebooks_json')
JSON_DIR.mkdir(parents=True, exist_ok=True)

def get_rulebooks_list():
    all_rulebooks = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    valid_rulebooks = []
    print("Datasets View: START SUBPROCESS - Validate Rulebooks")
    for rulebook in all_rulebooks:
        file_path = JSON_DIR / rulebook
        with open(file_path, "r", encoding="utf-8") as f:
            if validate_rulebook_values(json.load(f)):
                valid_rulebooks.append(rulebook)
            else:
                print(f"Datasets View: Invalid rulebook: {rulebook}")
    print("Datasets View: END SUBPROCESS - Validate Rulebooks")
    return valid_rulebooks

# --- Streamlit Page Layout ---
st.title("Datasets")

# --- Generate Structure Section ---
st.header("Generate Dataset Structure")

rulebooks = get_rulebooks_list()

# Display geneate dataset structure form
if rulebooks:
    
    # Display rulebooks in a selectbox and search time slider
    with st.form(key="generate_dataset_form"):
        selected_rulebook = st.selectbox("Rulebook Selector", rulebooks)
        st.html("<p style=\"color: orange\">Warning: Invalid rulebooks will not be displayed.</p>")
        solution_search_time_s = st.slider("Solution Search Time (seconds)", min_value=1, max_value=60, value=5)
        submitted = st.form_submit_button("Generate Dataset Structure")
        
        if submitted:
            st.session_state.form_submitted = True
            st.session_state.selected_rulebook = selected_rulebook
            st.session_state.solution_search_time_s = solution_search_time_s
            
    if st.session_state.form_submitted:
        
        # Display a loading message
        with st.spinner("Generating dataset structure. Please wait...", show_time=True):
            file_path = JSON_DIR / selected_rulebook
            with open(file_path, "r", encoding="utf-8") as f:
                selected_rulebook = json.load(f)
            print("Datasets View: START SUBPROCESS - Generate Dataset Structure")
            dataset_structure = create_dataset_structure(rulebook=selected_rulebook, solution_search_time_s=solution_search_time_s)
            print("Datasets View: END SUBPROCESS - Generate Dataset Structure")
        
        # Display dataset structure
        if dataset_structure:
            st.write(dataset_structure)
        else:
            st.error("Failed to generate dataset structure. Please try again.")

# There is no rulebook to display
else:
    
    # Display an alert and stop the script
    st.info("No rulebooks found. Please upload a rulebook first.")
    st.stop()
