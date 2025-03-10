import streamlit as st
import os
import json
import io
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, Union

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

def load_and_validate_json(file_path: Union[str, Path], validation_function: Callable) -> Optional[Dict]:
    """ 
    Loads a JSON file and validates it using the provided validation function.
    
    Args:
        file_path: Path to the JSON file (str or Path object)
        validation_function: Function to validate the JSON data
        
    Returns:
        Validated JSON data as a dictionary or None if invalid
    """
    # Convert to Path object if it's a string
    path = Path(file_path)
    
    # Define the load operation
    def load_operation(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    # Use the helper function
    success, data, output = _handle_json_operation(load_operation, validation_function, path)
    
    if success:
        return data
    else:
        st.error("JSON values are invalid - Please delete and re-upload the corrected file.")
        st.text_area("Console Output", output, height=200)
        return None

def validate_and_save_json(file_path: Union[str, Path], json_data: dict, validation_function: Callable) -> Optional[Path]:
    """
    Validates JSON data and saves it to the specified file path if valid.
    
    Args:
        file_path: Path where the JSON file will be saved (str or Path object)
        json_data: The JSON data to validate and save
        validation_function: Function to validate the JSON data
        
    Returns:
        Path object if validation and saving succeeded, None otherwise
    """
    # Convert to Path object if it's a string
    path = Path(file_path)
    
    # Check if the file already exists and modify the name
    if path.exists():
        base_name = path.stem
        extension = path.suffix
        parent = path.parent
        counter = 1
        while True:
            new_path = parent / f"{base_name} ({counter}){extension}"
            if not new_path.exists():
                path = new_path
                break
            counter += 1
    
    # Define the save operation
    def save_operation(path: Path, data):
        if validation_function(data):
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        return data
    
    # Use the helper function
    success, _, output = _handle_json_operation(save_operation, validation_function, path, json_data)
    
    if not success:
        st.error("JSON values are invalid - File was not saved.")
        st.text_area("Console Output", output, height=200)
        return None
    
    return path