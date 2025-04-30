"""
Module to handle the selection, deletion, and renaming of items in a given directory.

This is a custom Streamlit component that allows users to manage files in a specific directory.
Any changes made are reflected in the session state, allowing for a dynamic user experience.
The component is dynamic and can handle different types of items (e.g., rulebooks, datasets) 
based on the provided item type.
"""

import os
import streamlit as st
from pathlib import Path
from typing import List, Optional

from view_components.file_loader import FILE_DIRS

def get_files_list(item_type:str) -> List[str]:
    """ Get the list of valid files from the cache if available."""
    if item_type and f"{item_type}_valid_files" in st.session_state:
        return st.session_state[f"{item_type}_valid_files"]
    return []
    
def get_selected_file(item_type: str) -> Optional[str]:
    """ Get the selected file from the session state. """
    selector_key = f"{item_type}_selector"
    if selector_key in st.session_state:
        return st.session_state[selector_key]
    return None

def delete_item(directory: Path, selected_item: str, item_type:str) -> bool:
    """ Delete an item and handle the related messages. """
    
    # Check if the file exists before deleting
    file_path = directory / selected_item
    if file_path.exists():
        # Delete the file
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"An error occurred when deleting the file: {e.__class__.__name__}")
            return False
        
        # Update cache
        if item_type and f"{item_type}_valid_files" in st.session_state:
            if selected_item in st.session_state[f"{item_type}_valid_files"]:
                st.session_state[f"{item_type}_valid_files"].remove(selected_item)
                # Change selected when item is deleted
                if f"{item_type}_selector" in st.session_state:
                    del st.session_state[f"{item_type}_selector"]
                return True
    
    st.error(f"File not found. Could not delete {selected_item}.")
    return False

def rename_item(directory: Path, selected_item: str, current_name: str, new_name: str, item_type:str) -> bool:
    """ Rename an item and handle the related messages. """
    
    # Check if the new name is different from the current name
    if new_name != current_name:
        new_filename = f"{new_name}.json"
        new_path = directory / new_filename
        
        # Check for duplicate names
        if not new_path.exists():
            old_path = directory / selected_item
            old_path.rename(new_path)
            
            # Update cache
            if item_type and f"{item_type}_valid_files" in st.session_state:
                if selected_item in st.session_state[f"{item_type}_valid_files"]:
                    st.session_state[f"{item_type}_valid_files"].remove(selected_item)
                    st.session_state[f"{item_type}_valid_files"].append(new_filename)
                    # Change selected to new item
                    st.session_state[f"override_selected_{item_type}"] = f"{new_name}.json"
                    return True

    st.error("File name already exists or is the same as the current name.")
    return False

def add_new_file_and_select(new_item: str, item_type: str) -> bool:
    """ Add a new item to the session state. """
    if item_type in FILE_DIRS.keys() and f"{item_type}_valid_files" in st.session_state:
            # Add the new item to the cache
            st.session_state[f"{item_type}_valid_files"].append(new_item)
            # Update the selected item
            st.session_state[f"override_selected_{item_type}"] = new_item
            return True

    st.error(f"Code Error: {item_type} not in FILE_DIRS.")
    return False

def saved_file_selector(item_type: str) -> Optional[str]:
    """ Display the saved items selector and handle delete/rename actions. """
    
    if item_type not in FILE_DIRS.keys():
        st.error(f"Directory {item_type} not found in FILE_DIRS.")
        return None
    
    directory = FILE_DIRS[item_type]
    items = get_files_list(item_type)
    selector_key = f"{item_type}_selector"
    
    # Display the selector if there are items
    if items:
        with st.container(border=True):
            
            # Determine the selected item
            override_selected = f"override_selected_{item_type}"
            if override_selected in st.session_state:
                st.session_state[selector_key] = st.session_state[override_selected]
                del st.session_state[override_selected]
            if selector_key not in st.session_state or st.session_state[selector_key] not in items:
                    st.session_state[selector_key] = items[0] if items else None
            
            # Display the selector
            st.subheader(f"{item_type.capitalize()} Selector")
            selected_item = st.selectbox(label=f"Selected {item_type}", options=items, key=selector_key)
            
            if selected_item:
                # Display the delete button
                if st.button(f"Delete {item_type}", key=f"delete_{item_type}", icon="❌"):
                    if delete_item(directory, selected_item, item_type):
                        st.rerun()

                # Display the rename form
                current_name = os.path.splitext(selected_item)[0]
                with st.form(f"rename_{item_type}_form", clear_on_submit=False):
                    new_name = st.text_input(f"Rename {item_type}", value=current_name, key=f"new_{item_type}_name")
                    submitted = st.form_submit_button("Save name", icon="💾")
                
                # Handle rename form submission
                if submitted:
                    if rename_item(directory, selected_item, current_name, new_name, item_type):
                        st.rerun()
                        
            return selected_item
    return None
