import streamlit as st
import json
import io
import contextlib

def load_and_validate_json(file_path, validation_function):
    """ Loads a JSON file, validates it using the provided validation function. """
    captured_output = io.StringIO()
    try:
        # Load the JSON file
        with contextlib.redirect_stdout(captured_output):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Validate JSON data
            valid = validation_function(json_data)

            # Return the content if valid
            if valid:
                return json_data
            # Display an error message if invalid
            else:
                st.error("JSON values are invalid - Please delete and re-upload the corrected file.")
                st.text_area("Console Output", captured_output.getvalue(), height=200)
                return None

    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        st.text_area("Console Output", captured_output.getvalue(), height=200)
        return None