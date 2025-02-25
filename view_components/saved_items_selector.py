import os
import streamlit as st
from pathlib import Path

def get_items_list(directory):
    """Get a list of .json files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def delete_item(directory, selected_item):
    """Delete an item and handle the related messages."""
    file_path = directory / selected_item
    if file_path.exists():
        os.remove(file_path)
        st.session_state.stored_alert = {
            'type': 'warning',
            'message': f"{selected_item} deleted successfully."
        }
        return True
    else:
        st.error("File not found.")
        return False

def rename_item(directory, selected_item, current_name, new_name):
    """Rename an item and handle the related messages."""
    if new_name != current_name:
        new_filename = f"{new_name}.json"
        new_path = directory / new_filename
        
        # Check for duplicate names
        if new_path.exists():
            st.error(f"A file named '{new_filename}' already exists!")
            return False
        else:
            # Rename file
            old_path = directory / selected_item
            old_path.rename(new_path)
            st.session_state.stored_alert = {
                'type': 'success',
                'message': f"Renamed from '{selected_item}' to '{new_filename}'."
            }
            return True
    return False

def saved_items_selector(directory, item_type):
    """Display the saved items selector and handle delete/rename actions."""
    items = get_items_list(directory)
    
    if items:
        with st.container(border=True):
            # Get the index of the previously selected item
            default_index = 0
            if f"{item_type}_index" in st.session_state:
                default_index = st.session_state[f"{item_type}_index"]
                # Ensure the index is still valid
                if default_index >= len(items):
                    default_index = 0

            selected_item = st.selectbox(
                f"Select {item_type}",
                items,
                key=f"{item_type}_selector",
                index=default_index
            )
            
            if selected_item:
                # Save the index of the selected item to session state
                st.session_state[f"{item_type}_index"] = items.index(selected_item)

                if st.button(f"Delete {item_type}", key=f"delete_{item_type}", icon="❌"):
                    if delete_item(directory, selected_item):
                        # remove the index if the item is deleted
                        del st.session_state[f"{item_type}_index"]
                        st.rerun()

                current_name = os.path.splitext(selected_item)[0]
                with st.form(f"rename_{item_type}_form", clear_on_submit=False):
                    new_name = st.text_input(f"Rename {item_type}", value=current_name, key=f"new_{item_type}_name")
                    submitted = st.form_submit_button("Save name", icon="💾")
                
                if submitted:
                    if rename_item(directory, selected_item, current_name, new_name):
                        st.rerun()

            return selected_item