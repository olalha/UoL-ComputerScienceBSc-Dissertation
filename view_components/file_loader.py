import streamlit as st
import json
import io
import contextlib
from typing import Optional, Dict, Any, Callable, Tuple

def _handle_json_operation(operation_fn: Callable, validation_fn: Callable, *args) -> Tuple[bool, Optional[Dict], str]:
    """
    Helper function to handle JSON operations with validation and error capturing.
    
    Args:
        operation_fn: Function that performs the core operation (loading or saving)
        validation_fn: Function to validate the JSON data
        *args: Arguments to pass to the operation function
    
    Returns:
        Tuple containing:
        - Success status (bool)
        - JSON data if successful, None otherwise
        - Captured output for error reporting
    """
    captured_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_output):
            # Execute the operation function
            json_data = operation_fn(*args)
            
            # Validate JSON data
            valid = validation_fn(json_data)
            
            if valid:
                return True, json_data, captured_output.getvalue()
            else:
                return False, None, captured_output.getvalue()
    except Exception as e:
        print(f"Error: {e}")
        return False, None, captured_output.getvalue()

def load_and_validate_json(file_path: str, validation_function: Callable) -> Optional[Dict]:
    """ 
    Loads a JSON file and validates it using the provided validation function.
    
    Args:
        file_path: Path to the JSON file
        validation_function: Function to validate the JSON data
        
    Returns:
        Validated JSON data as a dictionary or None if invalid
    """
    # Define the load operation
    def load_operation(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Use the helper function
    success, data, output = _handle_json_operation(load_operation, validation_function, file_path)
    
    if success:
        return data
    else:
        st.error("JSON values are invalid - Please delete and re-upload the corrected file.")
        st.text_area("Console Output", output, height=200)
        return None

def validate_and_save_json(file_path: str, json_data: dict, validation_function: Callable) -> bool:
    """
    Validates JSON data and saves it to the specified file path if valid.
    
    Args:
        file_path: Path where the JSON file will be saved
        json_data: The JSON data to validate and save
        validation_function: Function to validate the JSON data
        
    Returns:
        True if validation and saving succeeded, False otherwise
    """
    # Define the save operation
    def save_operation(path, data):
        if validation_function(data):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return data
    
    # Use the helper function
    success, _, output = _handle_json_operation(save_operation, validation_function, file_path, json_data)
    
    if not success:
        st.error("JSON values are invalid - File was not saved.")
        st.text_area("Console Output", output, height=200)
    
    return success