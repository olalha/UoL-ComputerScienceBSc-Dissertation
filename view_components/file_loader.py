import streamlit as st
import os
import json
import tempfile
import io
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, Union

from utils.settings_manager import get_setting
from dataset_manager.dataset_structurer import validate_dataset_values
from chunk_manager.rulebook_parser import validate_rulebook_values, parse_rulebook_excel

# Directory to store JSON rulebooks
RB_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'rulebooks_json')
RB_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Directory to store JSON datasets
DS_JSON_DIR = Path(__file__).parent.parent / get_setting('PATH', 'datasets_json')
DS_JSON_DIR.mkdir(parents=True, exist_ok=True)

FILE_DIRS = {
    'rulebook': RB_JSON_DIR,
    'dataset': DS_JSON_DIR
}
FILE_VALIDATORS = {
    'rulebook': validate_rulebook_values,
    'dataset': validate_dataset_values
}

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
    
    return data if success else None, output

def validate_and_save_json(file_path: Union[str, Path], json_data: dict, validation_function: Callable, overwrite: bool = False) -> Optional[Path]:
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
    if not overwrite and path.exists():
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
    
    return path if success else None, output

def validate_directory_files(directory: Path, validation_function: Callable) -> Dict[str, bool]:
    """
    Validates all JSON files in a directory and returns a dictionary of filename: is_valid.
    
    Args:
        directory: Directory containing JSON files
        validation_function: Function to validate each JSON file
        
    Returns:
        Dictionary mapping filenames to validation status
    """
    results = {}
    
    # Skip if directory doesn't exist
    if not directory.exists():
        return results
    
    # Validate each JSON file in the directory
    path_parts = directory.parts
    print(f"START SUBPROCESS - Validating JSON files in directory '{Path(*path_parts[:-1])}'")
    for filename in directory.glob("*.json"):
            results[filename.name] = load_and_validate_json(filename, validation_function) is not None
    print(f"END SUBPROCESS")            
    return results

def initialize_file_cache() -> None:
    """
    Initialize session state cache for valid files in multiple directories.
    
    Args:
        directories_and_validators: Dict mapping session_state_key to (directory_path, validation_function)
    """
    for state_key, directory in FILE_DIRS.items():
        if f"{state_key}_valid_files" not in st.session_state:
            validation_results = validate_directory_files(directory, FILE_VALIDATORS[state_key])
            valid_files = [filename for filename, is_valid in validation_results.items() if is_valid]
            st.session_state[f"{state_key}_valid_files"] = valid_files


def load_and_validate_rulebook(file_name: str) -> Tuple[Optional[Dict], str]:
    """ Load and validate a rulebook JSON file by name. """
    return load_and_validate_json(RB_JSON_DIR / file_name, validate_rulebook_values)

def validate_and_save_rulebook(file_name: str, rulebook: dict, overwrite: bool = False) -> Tuple[Optional[Path], str]:
    """ Validate and save a rulebook JSON file to the specified path. """
    return validate_and_save_json(RB_JSON_DIR / file_name, rulebook, validate_rulebook_values, overwrite)

def load_and_validate_dataset(file_name: str) -> Tuple[Optional[Dict], str]:
    """ Load and validate a dataset JSON file by name. """
    return load_and_validate_json(DS_JSON_DIR / file_name, validate_dataset_values)

def validate_and_save_dataset(file_name: str, dataset: dict, overwrite: bool = False) -> Tuple[Optional[Path], str]:
    """ Validate and save a dataset JSON file to the specified path. """
    return validate_and_save_json(DS_JSON_DIR / file_name, dataset, validate_dataset_values, overwrite)

def process_rulebook_upload(uploaded_file: Any) -> Tuple[Optional[Path], str]:
    """ Process uploaded rulebook file and convert to JSON format. """
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    
    # Capture console output during processing
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        
        rulebook = None
        # Process Excel rulebook files
        if file_extension in ['.xlsx', '.xls', '.xlsm']:
            rulebook = parse_rulebook_excel(tmp_path)
        # Process JSON rulebook files
        elif file_extension == '.json':
            rulebook = load_and_validate_json(tmp_path, validate_rulebook_values)
        # Unsupported file type
        else:
            print("process_rulebook_upload: Unsupported file type - Please upload an Excel or JSON file.")
            return None, captured_output.getvalue()

        # Check if rulebook data has been loaded successfully
        if rulebook is None:
            print("process_rulebook_upload: Error loading rulebook data - Please check the file format and contents.")
            return None, captured_output.getvalue()
        
        try:
            tmp_path.unlink()
        except Exception:
            print(f"process_rulebook_upload: Error deleting temporary file.")
        
        # Save rulebook data to JSON file
        file_name = f"{rulebook['content_title']} - {rulebook['collection_mode']} - {rulebook['total']}.json"
        return validate_and_save_rulebook(file_name, rulebook, overwrite=False)
