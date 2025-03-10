import os
import streamlit as st
from pathlib import Path
from typing import List, Union, Optional

from view_components.file_loader import FILE_DIRS

def get_files_list(cache_key:str) -> List[str]:
    """ Get the list of valid files from the cache if available."""
    if cache_key and f"{cache_key}_valid_files" in st.session_state:
        return st.session_state[f"{cache_key}_valid_files"]
    return []
    
def get_selected_file(cache_key: str) -> Optional[str]:
    """ Get the selected file from the session state. """
    selector_key = f"{cache_key}_selector"
    if selector_key in st.session_state:
        return st.session_state[selector_key]
    return None

def delete_item(directory: Path, selected_item: str, cache_key:str) -> bool:
    """ Delete an item and handle the related messages. """
    file_path = directory / selected_item
    
    # Check if the file exists before deleting
    if file_path.exists():
        # Delete the file
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"An error occurred when deleting the file: {e.__class__.__name__}")
            return False
        
        # Update cache
        if cache_key and f"{cache_key}_valid_files" in st.session_state:
            if selected_item in st.session_state[f"{cache_key}_valid_files"]:
                st.session_state[f"{cache_key}_valid_files"].remove(selected_item)
        
        st.session_state.stored_alert = {
            'type': 'warning',
            'message': f"{selected_item} deleted successfully."
        }
        return True
    # Display an error message if the file is not found
    else:
        st.error("File not found. Cannot delete non-existent file.")
        return False

def rename_item(directory: Path, selected_item: str, current_name: str, new_name: str, cache_key:str) -> bool:
    """ Rename an item and handle the related messages. """
    if new_name != current_name:
        new_filename = f"{new_name}.json"
        new_path = directory / new_filename
        
        # Check for duplicate names
        if new_path.exists():
            st.error(f"A file named '{new_filename}' already exists!")
            return False
        # If unique then rename the file
        else:
            old_path = directory / selected_item
            old_path.rename(new_path)
            
            # Update cache
            if cache_key and f"{cache_key}_valid_files" in st.session_state:
                if selected_item in st.session_state[f"{cache_key}_valid_files"]:
                    st.session_state[f"{cache_key}_valid_files"].remove(selected_item)
                    st.session_state[f"{cache_key}_valid_files"].append(new_filename)
            
            st.session_state.stored_alert = {
                'type': 'success',
                'message': f"Renamed from '{selected_item}' to '{new_filename}'."
            }
            return True
    return False

def saved_file_selector(item_type: str) -> Optional[str]:
    """ Display the saved items selector and handle delete/rename actions. """
    
    if item_type not in FILE_DIRS.keys():
        st.error(f"Directory {item_type} not found in FILE_DIRS.")
        return None
    
    directory = FILE_DIRS[item_type]
    cache_key = item_type
    items = get_files_list(cache_key)
    selector_key = f"{item_type}_selector"
    
    if items:
        with st.container(border=True):
            # Initialize the selector key if needed or if current value is not valid
            if selector_key not in st.session_state or st.session_state[selector_key] not in items:
                if items:  # Make sure we have items before setting to first one
                    st.session_state[selector_key] = items[0]

            st.subheader(f"{item_type.capitalize()} Selector")
            selected_item = st.selectbox(
                f"Selected {item_type}",
                items,
                key=selector_key
            )
            
            if selected_item:
                # Display the delete button
                if st.button(f"Delete {item_type}", key=f"delete_{item_type}", icon="❌"):
                    if delete_item(directory, selected_item, cache_key):
                        # Remove the selector state when item is deleted
                        if selector_key in st.session_state:
                            del st.session_state[selector_key]
                        st.rerun()

                # Display the rename form
                current_name = os.path.splitext(selected_item)[0]
                with st.form(f"rename_{item_type}_form", clear_on_submit=False):
                    new_name = st.text_input(f"Rename {item_type}", value=current_name, key=f"new_{item_type}_name")
                    submitted = st.form_submit_button("Save name", icon="💾")
                
                if submitted:
                    if rename_item(directory, selected_item, current_name, new_name, cache_key):
                        # Update the selector with the new filename
                        new_filename = f"{new_name}.json"
                        st.session_state[selector_key] = new_filename
                        st.rerun()

            return selected_item
    return None

def change_selected_file(cache_key: str, selected_item: str) -> bool:
    """ Update the selected item in the session state. """
    if cache_key in FILE_DIRS.keys():
        # Update the session state with the selected item
        selector_key = f"{cache_key}_selector"
        if selector_key in st.session_state:
            st.session_state[selector_key] = selected_item
            return True
    return False

def add_new_file_to_selector(cache_key: str, new_item: str) -> bool:
    """ Add a new item to the session state. """
    if cache_key in FILE_DIRS.keys():
        # Update the session state with the new item
        if f"{cache_key}_valid_files" in st.session_state:
            st.session_state[f"{cache_key}_valid_files"].append(new_item)
            return True
    return False