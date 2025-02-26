import os
import json
import io
import contextlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from view_components.alert import show_alert
from view_components.saved_items_selector import saved_items_selector, get_items_list
from view_components.load_and_validate_json import load_and_validate_json
from utils.settings_manager import get_setting
from chunk_manager.rulebook_parser import validate_rulebook_values
from dataset_manager.dataset_structurer import create_dataset_structure, validate_and_update_dataset_meta

# Chart styling constants
SENTIMENT_COLORS = {
    'positive': '#4CAF50', 
    'neutral': '#2196F3', 
    'negative': '#F44336',
    'Unknown': '#9E9E9E'
}
CHART_ALPHA = 0.7
EDGE_COLOR = '#808080'
EDGE_WIDTH = 1
GRID_ALPHA = 0.7

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

def get_rulebooks_list() -> List[str]:
    """ Get a list of validated rulebooks for dataset generation. """
    
    all_rulebooks = [f for f in os.listdir(RB_JSON_DIR) if f.endswith('.json')]
    valid_rulebooks = []
    print("Datasets View: START SUBPROCESS - Get validated rulebooks for generation")
    for rulebook in all_rulebooks:
        file_path = RB_JSON_DIR / rulebook
        
        # Validate the rulebook
        with open(file_path, "r", encoding="utf-8") as f:
            if validate_rulebook_values(json.load(f)):
                valid_rulebooks.append(rulebook)
            else:
                print(f"Datasets View: Invalid rulebook: {rulebook}")
    print("Datasets View: END SUBPROCESS - Get validated rulebooks for generation")
    return valid_rulebooks

def generate_dataset_structure_form(rulebooks: List[str]) -> None:
    """ Displays a form for generating dataset structures from rulebooks. """
    
    # Display the form for generating dataset structure
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

    # When the form is submitted, generate the dataset structure
    if st.session_state.form_submitted:
        selected_rulebook = st.session_state.selected_rulebook
        solution_search_time_s = st.session_state.solution_search_time_s

        # Display a loading message and generate dataset structure
        captured_output = io.StringIO()
        with st.spinner("Generating dataset structure. Please wait...", show_time=True):

            # Read the selected rulebook (integrity already validated)
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
            # Automatically select the newly generated dataset
            items = get_items_list(DS_JSON_DIR)
            new_file_name = Path(result_path).name
            if new_file_name in items:
                st.session_state["Dataset_index"] = items.index(new_file_name)
        else:
            st.error("Failed to generate dataset structure. Please try again.")
            st.text_area("Console Output", captured_output.getvalue(), height=200)

def display_dataset_metrics(dataset_json: Dict[str, Any]) -> None:
    """ Display comprehensive metrics and visualizations for the dataset. """
    
    with st.container(border=True):
        st.subheader("Dataset Metrics")
        # Basic metrics in 5 columns
        metrics_cols = st.columns(5)
        with metrics_cols[0]:
            st.metric("Total Words", dataset_json.get("total_wc", 0))
        with metrics_cols[1]:
            st.metric("Collections", dataset_json.get("collections_count", 0))
        with metrics_cols[2]:
            st.metric("Total Chunks", dataset_json.get("total_cc", 0))
        with metrics_cols[3]:
            total_chunks = dataset_json.get("total_cc", 0)
            chunks_with_text = dataset_json.get("chunks_with_text", 0)
            percentage = (chunks_with_text / total_chunks) * 100 if total_chunks else 0
            st.metric("Chunks Text", f"{percentage:.1f}%")
        with metrics_cols[4]:
            total_collections = dataset_json.get("collections_count", 0)
            collections_with_text = dataset_json.get("collections_with_text", 0)
            percentage = (collections_with_text / total_collections) * 100 if total_collections else 0
            st.metric("Collections Text",  f"{percentage:.1f}%")
                
        # Collection Size Distribution (Stacked Bar Chart)
        with st.expander("Collection Size Distribution", expanded=False, icon="📈"):
            st.subheader("Collection Size Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_size_distribution(dataset_json, by_chunks=True, category='collection')
                
            with tab2:
                plot_size_distribution(dataset_json, by_chunks=False, category='collection')
        
        # Topic Coverage Distribution (Stacked Bar Chart)
        with st.expander("Topic Coverage Distribution", expanded=False, icon="💬"):
            st.subheader("Topic Coverage Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_size_distribution(dataset_json, by_chunks=True, category='topic')
                
            with tab2:
                plot_size_distribution(dataset_json, by_chunks=False, category='topic')
        
        # Overall Sentiment Distribution (Pie Charts and Box Plot)
        with st.expander("Overall Sentiment Distribution", expanded=False, icon="😃"):
            st.subheader("Overall Sentiment Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_sentiment_pie_chart(dataset_json, by_chunks=True)
            with tab2:
                plot_sentiment_pie_chart(dataset_json, by_chunks=False)
            
        # Word Count Distribution by Chunk (Box Plot)
        with st.expander("Word Count Distribution by Chunk", expanded=False, icon="📊"):
            st.subheader("Word Count Distribution by Chunk")
            plot_sentiment_box_plot(dataset_json)

def plot_size_distribution(dataset_json: Dict[str, Any], by_chunks: bool = True, category: str = 'collection') -> None:
    """
    Plot a stacked bar chart showing sentiment distribution by either topic or collection.
    
    Args:
        dataset_json: The dataset JSON object
        by_chunks: If True, use chunk count; otherwise use word count
        category: Display data organized by 'topic's or by 'collection's
    """
    # Check if data is available
    if not dataset_json:
        st.info("No data available for chart.")
        return
    
    # Initialize variables for data processing
    data = {}
    x_labels = []
    x_title = ""
    
    if category == 'topic':
        # --- Topic-based chart data processing ---
        # Get distribution data based on count type
        distribution = dataset_json.get(f"ts_{'cc' if by_chunks else 'wc'}_distribution", {})
        
        if not distribution:
            st.info(f"No {'chunk' if by_chunks else 'word'} count distribution data available by topic.")
            return
        
        # Group data by topic and sentiment
        topic_sentiment = {}
        
        # Process the topic-sentiment distribution data
        for key, value in distribution.items():
            parts = key.split(" - ")
            if len(parts) == 2:
                topic, sentiment = parts
                if topic not in topic_sentiment:
                    topic_sentiment[topic] = {"positive": 0, "neutral": 0, "negative": 0}
                topic_sentiment[topic][sentiment] = value
        
        # Sort topics by total size (descending)
        topics = list(topic_sentiment.keys())
        topic_totals = {topic: sum(values.values()) for topic, values in topic_sentiment.items()}
        topics.sort(key=lambda x: topic_totals[x], reverse=True)
        
        # Extract data for plotting
        data = {
            "pos_vals": [topic_sentiment[topic]["positive"] for topic in topics],
            "neu_vals": [topic_sentiment[topic]["neutral"] for topic in topics],
            "neg_vals": [topic_sentiment[topic]["negative"] for topic in topics]
        }
        
        x_labels = topics
        x_title = "Topics"
    
    elif category == 'collection':
        # --- Collection-based chart data processing ---
        collections = dataset_json.get("collections", [])
        if not collections:
            st.info("No collections data available for chart.")
            return
        
        # Group collections by sentiment
        collection_sentiment = {}
        
        # Process each collection's chunk data
        for i, collection in enumerate(collections):
            collection_id = f"Collection {i+1}"
            collection_sentiment[collection_id] = {"positive": 0, "neutral": 0, "negative": 0, "Unknown": 0}
            
            # Process chunks within each collection
            chunks = collection.get("chunks", [])
            for chunk in chunks:
                chunk_dict = chunk.get("chunk_dict", {})
                sentiment = chunk_dict.get("sentiment", "Unknown")
                
                if by_chunks:
                    # Count each chunk
                    collection_sentiment[collection_id][sentiment] += 1
                else:
                    # Sum word counts
                    collection_sentiment[collection_id][sentiment] += chunk_dict.get("wc", 0)
        
        # Sort collections by total size (descending)
        collection_ids = list(collection_sentiment.keys())
        collection_totals = {cid: sum(values.values()) for cid, values in collection_sentiment.items()}
        collection_ids.sort(key=lambda x: collection_totals[x], reverse=True)
        
        # Extract data for plotting
        data = {
            "pos_vals": [collection_sentiment[cid].get("positive", 0) for cid in collection_ids],
            "neu_vals": [collection_sentiment[cid].get("neutral", 0) for cid in collection_ids],
            "neg_vals": [collection_sentiment[cid].get("negative", 0) for cid in collection_ids]
        }
        
        x_labels = collection_ids
        x_title = "Collections"
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Set up x positions for bars
    x = np.arange(len(x_labels))
    
    # Create stacked bars for sentiments
    ax.bar(x, data["pos_vals"], label='Positive', color=SENTIMENT_COLORS['positive'], alpha=CHART_ALPHA)
    ax.bar(x, data["neu_vals"], bottom=data["pos_vals"], label='Neutral', color=SENTIMENT_COLORS['neutral'], alpha=CHART_ALPHA)
    
    # Calculate the bottom position for negative values and add them
    bottom_pos = [data["pos_vals"][i] + data["neu_vals"][i] for i in range(len(data["pos_vals"]))]
    ax.bar(x, data["neg_vals"], bottom=bottom_pos, label='Negative', color=SENTIMENT_COLORS['negative'], alpha=CHART_ALPHA)
    
    # Set chart labels and title
    metric_type = 'Chunks' if by_chunks else 'Words'
    ax.set_xlabel(x_title)
    ax.set_ylabel(f'Count ({metric_type})')
    
    # Set x-axis ticks
    ax.set_xticks(x)
    
    # Format x-axis labels based on category
    if category == 'collection':
        # For collections, hide labels to avoid overcrowding
        ax.set_xticklabels([""] * len(x_labels))
    else:
        # For topics, truncate long labels
        truncated_labels = [label[:20] + "..." if len(label) > 20 else label for label in x_labels]
        if len(x_labels) > 8:
            # Rotate labels for better readability when there are many
            ax.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticklabels(truncated_labels)
    
    # Add legend and adjust layout
    ax.legend()
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig)

def plot_sentiment_pie_chart(dataset_json: Dict[str, Any], by_chunks: bool = True) -> None:
    """
    Plot pie chart showing sentiment distribution by either chunk count or word count.
    
    Args:
        dataset_json: The dataset JSON object
        by_chunks: If True, use chunk count; otherwise use word count
    """
    # Get the appropriate sentiment distribution data
    sentiment_data = dataset_json.get(f"sentiment_{'cc' if by_chunks else 'wc'}_distribution", {})
    
    # Check if data is available
    if not sentiment_data:
        st.info(f"No {'chunk' if by_chunks else 'word'} count sentiment distribution data available.")
        return
        
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(sentiment_data.keys())
    sizes = list(sentiment_data.values())
    
    # Get colors for each sentiment label
    pie_colors = [SENTIMENT_COLORS.get(label, SENTIMENT_COLORS['Unknown']) for label in labels]
    
    # Plot the pie chart
    ax.pie(
        sizes, 
        labels=None, 
        autopct='%1.1f%%', 
        colors=pie_colors,  
        wedgeprops={
            'alpha': CHART_ALPHA, 
            'edgecolor': EDGE_COLOR, 
            'linewidth': EDGE_WIDTH
        }
    )
    
    # Create legend labels with values
    legend_labels = [f"{label} ({value})" for label, value in sentiment_data.items()]
    
    # Add total value to legend
    total = sum(sentiment_data.values())
    legend_labels.append(f"Total: {total}")
    
    # Add legend to chart
    ax.legend(legend_labels, loc="center right", bbox_to_anchor=(1.4, 0.5))
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_sentiment_box_plot(dataset_json: Dict[str, Any]) -> None:
    """ 
    Plot a box plot representing the word count distribution of chunks grouped by sentiment. 
    
    Args:
        dataset_json: The dataset JSON object
    """
    # Extract chunk data from collections
    chunks_data = []
    collections = dataset_json.get("collections", [])
    
    # Process each collection's chunks to gather sentiment-based word count data
    for collection in collections:
        chunks = collection.get("chunks", [])
        for chunk in chunks:
            chunk_dict = chunk.get("chunk_dict", {})
            wc = chunk_dict.get("wc", 0)
            sentiment = chunk_dict.get("sentiment", "Unknown")
            
            # Only include chunks that have word counts
            if wc > 0:
                chunks_data.append({
                    'count': wc,
                    'sentiment': sentiment
                })
    
    # Check if data is available
    if not chunks_data:
        st.info("No word count data available for box plot.")
        return
        
    # Create DataFrame for analysis
    df = pd.DataFrame(chunks_data)
    
    # Define the sentiment order for plotting
    sentiment_order = ['positive', 'neutral', 'negative', 'All']
    available_sentiments = [s for s in sentiment_order if s in df['sentiment'].unique() or s == 'All']
    
    # Prepare data for the box plot - group by sentiment
    data_to_plot = [df.loc[df['sentiment'] == s, 'count'] if s != 'All' else df['count'] for s in available_sentiments]
    
    # Create the box plot
    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot(data_to_plot, labels=available_sentiments, patch_artist=True)
    
    # Customize box colors based on sentiment
    for patch, sentiment in zip(box['boxes'], available_sentiments):
        patch.set_facecolor(SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS['Unknown']))
        patch.set_alpha(CHART_ALPHA)
    
    # Make the borders thinner and lighter
    for element in ['whiskers', 'medians', 'caps']:
        plt.setp(box[element], color='#696969', linewidth=EDGE_WIDTH)
    
    # Set axis labels and add grid
    ax.set_ylabel('Word Count')
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    
    plt.tight_layout()
    st.pyplot(fig)

# --- Streamlit Page Layout ---
st.title("Datasets")

# --- Generate Structure Section ---
st.header("Generate Dataset Structure")

# Display generate dataset structure form
rulebooks = get_rulebooks_list()
if rulebooks:
    generate_dataset_structure_form(rulebooks)
else:
    st.info("No rulebooks found. Please upload a rulebook first.")

# --- Saved Datasets Section ---
st.header("Saved Datasets")

# Display dataset selector
selected_dataset = saved_items_selector(DS_JSON_DIR, "Dataset")

# Display selected dataset
if selected_dataset:
    # Validate and load the selected dataset
    file_path = DS_JSON_DIR / selected_dataset
    dataset_json = load_and_validate_json(file_path, validate_and_update_dataset_meta)

    # Display dataset content
    if dataset_json:
        st.markdown("---")
        st.info(f"{selected_dataset}")
        
        # Call the modularized function to display metrics
        display_dataset_metrics(dataset_json)
        
        # Display the full JSON in an expander at the bottom
        with st.expander("View Raw JSON Data", expanded=False):
            st.json(dataset_json)
else:
    st.info("Generate and select a dataset to view its content.")