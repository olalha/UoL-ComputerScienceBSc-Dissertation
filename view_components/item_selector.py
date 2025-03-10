import os
import streamlit as st
from pathlib import Path
from typing import List, Union, Optional

def get_items_list(directory: Union[str, Path]) -> List[str]:
    """ Get a list of .json files in the specified directory. """
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def delete_item(directory: Path, selected_item: str) -> bool:
    """ Delete an item and handle the related messages. """
    file_path = directory / selected_item
    
    # Check if the file exists before deleting
    if file_path.exists():
        # Delete the file
        os.remove(file_path)
        st.session_state.stored_alert = {
            'type': 'warning',
            'message': f"{selected_item} deleted successfully."
        }
        return True
    # Display an error message if the file is not found
    else:
        st.error("File not found.")
        return False

def rename_item(directory: Path, selected_item: str, current_name: str, new_name: str) -> bool:
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
            st.session_state.stored_alert = {
                'type': 'success',
                'message': f"Renamed from '{selected_item}' to '{new_filename}'."
            }
            return True
    return False

def saved_items_selector(directory: Path, item_type: str) -> Optional[str]:
    """ Display the saved items selector and handle delete/rename actions. """
    items = get_items_list(directory)
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
                    if delete_item(directory, selected_item):
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
                    if rename_item(directory, selected_item, current_name, new_name):
                        # Update the selector with the new filename
                        new_filename = f"{new_name}.json"
                        st.session_state[selector_key] = new_filename
                        st.rerun()

            return selected_item
    return None