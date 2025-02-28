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

from view_components.alerter import show_alert
from view_components.item_selector import saved_items_selector, get_items_list
from view_components.file_loader import load_and_validate_json, validate_and_save_json
from utils.settings_manager import get_setting
from chunk_manager.rulebook_parser import validate_rulebook_values
from dataset_manager.dataset_structurer import create_dataset_structure, validate_and_update_dataset_meta
from dataset_manager.text_generator import generate_collection_text

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
                st.session_state["Dataset_selected"] = new_file_name
        else:
            st.error("Failed to generate dataset structure. Please try again.")
            st.text_area("Console Output", captured_output.getvalue(), height=200)

def display_dataset_metrics(dataset: Dict[str, Any]) -> None:
    """ Display comprehensive metrics and visualizations for the dataset. """
    
    st.subheader("Dataset Metrics")
    with st.container(border=True):
        # Basic metrics in 5 columns
        metrics_cols = st.columns(5)
        with metrics_cols[0]:
            st.metric("Total Words", dataset.get("total_wc", 0))
        with metrics_cols[1]:
            st.metric("Collections", dataset.get("collections_count", 0))
        with metrics_cols[2]:
            st.metric("Total Chunks", dataset.get("total_cc", 0))
        with metrics_cols[3]:
            total_chunks = dataset.get("total_cc", 0)
            chunks_with_text = dataset.get("chunks_with_text", 0)
            percentage = (chunks_with_text / total_chunks) * 100 if total_chunks else 0
            st.metric("Chunks Text", f"{percentage:.1f}%")
        with metrics_cols[4]:
            total_collections = dataset.get("collections_count", 0)
            collections_with_text = dataset.get("collections_with_text", 0)
            percentage = (collections_with_text / total_collections) * 100 if total_collections else 0
            st.metric("Collections Text",  f"{percentage:.1f}%")
                
        # Collection Size Distribution (Stacked Bar Chart)
        with st.expander("Collection Size Distribution", expanded=False, icon="📈"):
            st.subheader("Collection Size Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_size_distribution(dataset, by_chunks=True, category='collection')
                
            with tab2:
                plot_size_distribution(dataset, by_chunks=False, category='collection')
        
        # Topic Coverage Distribution (Stacked Bar Chart)
        with st.expander("Topic Coverage Distribution", expanded=False, icon="💬"):
            st.subheader("Topic Coverage Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_size_distribution(dataset, by_chunks=True, category='topic')
                
            with tab2:
                plot_size_distribution(dataset, by_chunks=False, category='topic')
        
        # Overall Sentiment Distribution (Pie Charts and Box Plot)
        with st.expander("Overall Sentiment Distribution", expanded=False, icon="😃"):
            st.subheader("Overall Sentiment Distribution")
            tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
            with tab1:
                plot_sentiment_pie_chart(dataset, by_chunks=True)
            with tab2:
                plot_sentiment_pie_chart(dataset, by_chunks=False)
            
        # Word Count Distribution by Chunk (Box Plot)
        with st.expander("Word Count Distribution by Chunk", expanded=False, icon="📊"):
            st.subheader("Word Count Distribution by Chunk")
            plot_sentiment_box_plot(dataset)

def plot_size_distribution(dataset: Dict[str, Any], by_chunks: bool = True, category: str = 'collection') -> None:
    """
    Plot a stacked bar chart showing sentiment distribution by either topic or collection.
    
    Args:
        dataset: The dataset JSON object
        by_chunks: If True, use chunk count; otherwise use word count
        category: Display data organized by 'topic's or by 'collection's
    """
    # Check if data is available
    if not dataset:
        st.info("No data available for chart.")
        return
    
    # Initialize variables for data processing
    data = {}
    x_labels = []
    x_title = ""
    
    if category == 'topic':
        # --- Topic-based chart data processing ---
        # Get distribution data based on count type
        distribution = dataset.get(f"ts_{'cc' if by_chunks else 'wc'}_distribution", {})
        
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
        collections = dataset.get("collections", [])
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

def plot_sentiment_pie_chart(dataset: Dict[str, Any], by_chunks: bool = True) -> None:
    """
    Plot pie chart showing sentiment distribution by either chunk count or word count.
    
    Args:
        dataset: The dataset JSON object
        by_chunks: If True, use chunk count; otherwise use word count
    """
    # Get the appropriate sentiment distribution data
    sentiment_data = dataset.get(f"sentiment_{'cc' if by_chunks else 'wc'}_distribution", {})
    
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

def plot_sentiment_box_plot(dataset: Dict[str, Any]) -> None:
    """ 
    Plot a box plot representing the word count distribution of chunks grouped by sentiment. 
    
    Args:
        dataset: The dataset JSON object
    """
    # Extract chunk data from collections
    chunks_data = []
    collections = dataset.get("collections", [])
    
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

def initialize_collection_filters(collections, dataset_id=None):
    """
    Initialize session state for collection filters based on the current dataset.
    
    Args:
        collections: List of collections from the dataset
        dataset_id: Identifier for the current dataset to track changes
    """
    # Check if filters need initialization (first load) or re-initialization (dataset changed)
    if ("filter_dataset_id" not in st.session_state or 
            st.session_state.filter_dataset_id != dataset_id or
            "filter_min_wc" not in st.session_state):
        
        # Store the current dataset ID for change detection
        st.session_state.filter_dataset_id = dataset_id
        
        # Calculate maximum values based on actual dataset
        max_wc = max([c.get("collection_wc", 0) for c in collections]) if collections else 100
        max_cc = max([c.get("collection_cc", 0) for c in collections]) if collections else 10
        
        # Initialize filter values
        st.session_state.filter_min_wc = 0
        st.session_state.filter_max_wc = max_wc
        st.session_state.filter_min_cc = 0
        st.session_state.filter_max_cc = max_cc
        st.session_state.filter_topic = "All"
        st.session_state.filter_sentiment = "All"
        st.session_state.filter_text_status = "All"
        st.session_state.filter_applied = False
        
        # Important: Don't set widget values directly, just store the initial values
        # These will be used when the widgets are created
        st.session_state.initial_wc_min = 0
        st.session_state.initial_wc_max = max_wc
        st.session_state.initial_cc_min = 0
        st.session_state.initial_cc_max = max_cc
        st.session_state.initial_topic = "All"
        st.session_state.initial_sentiment = "All"
        st.session_state.initial_text_status = "All"

def display_collection_filters(collections, all_topics, all_sentiments):
    """
    Display collection filter controls.
    
    Args:
        collections: List of collections from the dataset
        all_topics: Set of all available topics
        all_sentiments: Set of all available sentiments
        
    Returns:
        bool: True if filters should be applied, False otherwise
    """
    with st.container(border=True):
        st.markdown("#### Filter Collections")
        
        col1, col2 = st.columns(2)
        
        # Get current max values
        max_wc = max([c.get("collection_wc", 0) for c in collections]) if collections else 100
        max_cc = max([c.get("collection_cc", 0) for c in collections]) if collections else 10
        
        with col1:
            # Word count range slider - using the current filter values
            wc_values = st.slider(
                "Word Count Range",
                min_value=0,
                max_value=max_wc,
                value=(st.session_state.filter_min_wc, st.session_state.filter_max_wc),
                key="temp_wc_range"
            )
            
            # Topic select box
            topic = st.selectbox(
                "Filter by Topic",
                options=["All"] + sorted(list(all_topics)),
                index=0 if st.session_state.filter_topic == "All" else 
                     sorted(list(all_topics)).index(st.session_state.filter_topic) + 1 
                     if st.session_state.filter_topic in all_topics else 0,
                key="temp_topic"
            )
            
            # Text status selectbox
            text_status_options = ["All", "Has Text", "No Text"]
            text_status = st.selectbox(
                "Filter by Text Status",
                options=text_status_options,
                index=text_status_options.index(st.session_state.filter_text_status) 
                      if st.session_state.filter_text_status in text_status_options else 0,
                key="temp_text_status"
            )
        
        with col2:
            # Chunk count range slider
            cc_values = st.slider(
                "Chunk Count Range",
                min_value=0,
                max_value=max_cc,
                value=(st.session_state.filter_min_cc, st.session_state.filter_max_cc),
                key="temp_cc_range"
            )
            
            # Sentiment select box
            sentiment = st.selectbox(
                "Filter by Sentiment",
                options=["All"] + sorted(list(all_sentiments)),
                index=0 if st.session_state.filter_sentiment == "All" else 
                     sorted(list(all_sentiments)).index(st.session_state.filter_sentiment) + 1 
                     if st.session_state.filter_sentiment in all_sentiments else 0,
                key="temp_sentiment"
            )
        
        # Apply filters button
        apply_filters = st.button("Apply Filters", type="primary")
        
        if apply_filters:
            # Copy temporary values to actual filter values
            st.session_state.filter_min_wc = wc_values[0]
            st.session_state.filter_max_wc = wc_values[1]
            st.session_state.filter_min_cc = cc_values[0]
            st.session_state.filter_max_cc = cc_values[1]
            st.session_state.filter_topic = topic
            st.session_state.filter_sentiment = sentiment
            st.session_state.filter_text_status = text_status
            st.session_state.filter_applied = True
            
        return apply_filters

def display_collections_table(dataset: Dict[str, Any]) -> None:
    """
    Display a table of collections with advanced filtering.
    
    Args:
        dataset: The dataset JSON object
    """
    collections = dataset.get("collections", [])
    if not collections:
        st.info("No collections found in this dataset.")
        return

    st.subheader("Collections")
    
    # Reset filters if a new dataset is selected
    selected_dataset = st.session_state.get("Dataset_selected", None)
    if st.session_state.get("current_dataset") != selected_dataset:
        keys_to_reset = [
            "filter_dataset_id", "filter_min_wc", "filter_max_wc", "filter_min_cc", "filter_max_cc",
            "filter_topic", "filter_sentiment", "filter_text_status", "filter_applied",
            "initial_wc_min", "initial_wc_max", "initial_cc_min", "initial_cc_max", "initial_topic",
            "initial_sentiment", "initial_text_status"
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state["current_dataset"] = selected_dataset

    # Get the dataset ID to detect changes
    dataset_id = dataset.get("id", dataset.get("name", None))
    
    # Initialize filters based on this dataset
    initialize_collection_filters(collections, dataset_id)
    
    # Get all available topics and sentiments
    all_topics = set()
    all_sentiments = set()
    
    for collection in collections:
        for chunk in collection.get("chunks", []):
            chunk_dict = chunk.get("chunk_dict", {})
            topic = chunk_dict.get("topic")
            sentiment = chunk_dict.get("sentiment")
            if topic:
                all_topics.add(topic)
            if sentiment:
                all_sentiments.add(sentiment)
    
    # Display filter controls
    display_collection_filters(collections, all_topics, all_sentiments)
    
    # Handle filter reset if applicable
    if st.session_state.filter_applied:
        
        # Reset filters button
        if st.button("Reset Table", type="secondary"):
            # Calculate maximum values based on actual dataset
            max_wc = max([c.get("collection_wc", 0) for c in collections]) if collections else 100
            max_cc = max([c.get("collection_cc", 0) for c in collections]) if collections else 10
            
            # Reset filter values
            st.session_state.filter_min_wc = 0
            st.session_state.filter_max_wc = max_wc
            st.session_state.filter_min_cc = 0
            st.session_state.filter_max_cc = max_cc
            st.session_state.filter_topic = "All"
            st.session_state.filter_sentiment = "All"
            st.session_state.filter_text_status = "All"
            st.session_state.filter_applied = False
            
            # Apply the reset
            st.rerun()
        
        # Show summary of applied filters
        filter_summary = []
        max_wc = max([c.get("collection_wc", 0) for c in collections]) if collections else 100
        max_cc = max([c.get("collection_cc", 0) for c in collections]) if collections else 10
        
        if st.session_state.filter_min_wc > 0 or st.session_state.filter_max_wc < max_wc:
            filter_summary.append(f"Words: {st.session_state.filter_min_wc}-{st.session_state.filter_max_wc}")
        
        if st.session_state.filter_min_cc > 0 or st.session_state.filter_max_cc < max_cc:
            filter_summary.append(f"Chunks: {st.session_state.filter_min_cc}-{st.session_state.filter_max_cc}")
        
        if st.session_state.filter_topic != "All":
            filter_summary.append(f"Topic: {st.session_state.filter_topic}")
            
        if st.session_state.filter_sentiment != "All":
            filter_summary.append(f"Sentiment: {st.session_state.filter_sentiment}")
            
        if st.session_state.filter_text_status != "All":
            filter_summary.append(f"Text: {st.session_state.filter_text_status}")
            
        if filter_summary:
            st.caption(f"Filters applied: {', '.join(filter_summary)}")
    
    # Create a dataframe with collection information
    collection_data = []
    
    max_chunk_count = max([c.get("collection_cc", 0) for c in collections])
    
    for i, collection in enumerate(collections):
        # Get collection metrics
        cc = collection.get("collection_cc", 0)
        wc = collection.get("collection_wc", 0)
        has_text = collection.get("full_text") is not None
        
        # Get topics and sentiments with counts
        topic_sentiment = {}
        chunk_word_counts = []
        
        # Track if collection matches sentiment/topic filters
        matches_sentiment_filter = st.session_state.filter_sentiment == "All"
        matches_topic_filter = st.session_state.filter_topic == "All"
        
        for chunk in collection.get("chunks", []):
            chunk_dict = chunk.get("chunk_dict", {})
            sentiment = chunk_dict.get("sentiment", "Unknown")
            topic = chunk_dict.get("topic", "Unknown")
            wc_chunk = chunk_dict.get("wc", 0)
            
            # Add to word count list for bar chart
            chunk_word_counts.append(wc_chunk)
            
            # Create topic-sentiment pairs
            key = (topic, sentiment)
            if key in topic_sentiment:
                topic_sentiment[key] += 1
            else:
                topic_sentiment[key] = 1
            
            # Check if matches filters
            if sentiment == st.session_state.filter_sentiment:
                matches_sentiment_filter = True
            if topic == st.session_state.filter_topic:
                matches_topic_filter = True
        
        # Apply filters
        if st.session_state.filter_applied:
            # Text status filter check
            text_status_match = True
            if st.session_state.filter_text_status == "Has Text" and not has_text:
                text_status_match = False
            elif st.session_state.filter_text_status == "No Text" and has_text:
                text_status_match = False
                
            # Skip if doesn't match filters
            if (wc < st.session_state.filter_min_wc or 
                wc > st.session_state.filter_max_wc or
                cc < st.session_state.filter_min_cc or 
                cc > st.session_state.filter_max_cc or
                (st.session_state.filter_sentiment != "All" and not matches_sentiment_filter) or
                (st.session_state.filter_topic != "All" and not matches_topic_filter) or
                not text_status_match):
                continue
        
        # Format topic-sentiment pairs into a readable format
        topic_sentiment_display = []
        for (topic, sentiment), count in sorted(topic_sentiment.items(), key=lambda x: x[1], reverse=True):
            # Add emoji based on sentiment
            emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😟" if sentiment == "negative" else "❓"
            topic_sentiment_display.append(f"{emoji} {topic}")
        
        # Normalize chunk word counts for consistent bar widths
        normalized_word_counts = chunk_word_counts
        
        # Add zeros to fill up to max_chunk_count
        while len(normalized_word_counts) < max_chunk_count:
            normalized_word_counts.append(0)
        
        # Add to data
        collection_data.append({
            "Chunks": cc,
            "Words": wc,
            "Topics & Sentiment": topic_sentiment_display,
            "Chunk Distribution": normalized_word_counts,
            "Has Text": "✓" if has_text else "✗"
        })
    
    # Create dataframe
    df = pd.DataFrame(collection_data)
    
    # Find the max word count in any chunk for y-axis scaling
    max_chunk_wc = max([max(row["Chunk Distribution"]) if row["Chunk Distribution"] else 0 for row in collection_data]) + 5
    
    # Display as a dataframe with custom formatting
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                help="Collection ID",
                width="small",
            ),
            "Chunks": st.column_config.NumberColumn(
                "Chunks",
                help="Number of chunks",
                format="%d",
                width="small",
            ),
            "Words": st.column_config.NumberColumn(
                "Words",
                help="Word count",
                format="%d",
                width="small",
            ),
            "Topics & Sentiment": st.column_config.ListColumn(
                "Topics & Sentiment",
                help="Topics with sentiment indicators (😊 positive, 😐 neutral, ☹️ negative)",
                width="large",
            ),
            "Chunk Distribution": st.column_config.BarChartColumn(
                "Word Count by Chunk",
                help="Distribution of word count across chunks",
                y_min=0,
                y_max=max_chunk_wc,
            ),
            "Has Text": st.column_config.TextColumn(
                "Has Text",
                help="Whether the collection has full text available",
                width="small",
            ),
        },
        hide_index=False,
        use_container_width=True,
    )

def display_text_generation_tab(file_path: str, dataset: Dict[str, Any]) -> None:
    """
    Display text generation interface for a selected collection.
    
    Args:
        dataset: The dataset JSON object
    """
    # Check if collections exist
    collections = dataset.get("collections", [])
    if not collections:
        st.info("No collections found in this dataset.")
        return
    
    # Check if review item exists
    review_item = dataset.get("review_item", None)
    if not review_item:
        st.info("No review item found in this dataset.")
        return
        
    st.subheader("Select a Collection")
    
    # Collection selector
    col_count = len(collections)
    col_index = st.number_input(
        "Selected Index", 
        min_value=0, 
        max_value=col_count-1, 
        value=0,
        step=1
    )
    st.caption(f"Collection index range: 0-{col_count-1}")
    
    # Get selected collection
    selected_collection = collections[col_index]
    
    # Display generation button
    models = [i for i in get_setting("OPENAI_LLM_MODELS").values()]
    selected_model = st.selectbox(
        f"LLM Model Selector",
        models,
        key=f"llm_model_selector"
    )
    collection_text = selected_collection.get("collection_text", "")
    if st.button(f"{"Re-G" if collection_text else "G"}enerate Collection Text", icon="🤖"):
        # Generate text for the collection
        captured_output = io.StringIO()
        print("""Datasets View: START SUBPROCESS - Generate Collection Text""")
        with contextlib.redirect_stdout(captured_output):
            with st.spinner("Generating collection text. Please wait...", show_time=True):
                collection_with_generated_text = generate_collection_text(selected_collection, review_item, selected_model)
            if collection_with_generated_text:
                collections[col_index] = collection_with_generated_text
                dataset["collections"] = collections
                validate_and_save_json(file_path=file_path, json_data=dataset, validation_function=validate_and_update_dataset_meta)
        print("""Datasets View: END SUBPROCESS - Generate Collection Text""")
    
    st.divider()
    st.subheader("Collection Information")
    
    # Display the generated text
    collection_text = selected_collection.get("collection_text", "")
    if collection_text:
        with st.expander("Collection Text", expanded=True, icon="💬"):
            st.markdown(collection_text)
    
    
    # Display metrics for the selected collection
    with st.container(border=True):
        cols = st.columns([2, 1, 2, 2, 2])
        with cols[0]:
            st.metric(f"Collection Index", col_index)
        with cols[1]:
            st.empty()
        with cols[2]:
            st.metric("Chunk Count", selected_collection.get("collection_cc", 0))
        with cols[3]:
            st.metric("Given Word Count", selected_collection.get("collection_wc", 0))
        with cols[4]:
            collection_text_len = len(str(selected_collection.get("collection_text", "")).split())
            st.metric("Actual Word Count", collection_text_len if collection_text_len > 1 else "N/A")
    
    # Display chunk information
    chunks = selected_collection.get("chunks", [])
    for i, chunk in enumerate(chunks):
        chunk_dict = chunk.get("chunk_dict", {})
        chunk_text = chunk.get("chunk_text", "")
        
        with st.container(border=True):
            
            cols = st.columns([2, 1, 4, 2, 2])
            with cols[0]:
                st.info(f"Chunk {i+1}")
            with cols[1]:
                st.empty()
            with cols[2]:
                st.caption("Topic")
                st.write(chunk_dict.get("topic", "Unknown"))
            with cols[3]:
                st.caption("Sentiment")
                st.write(chunk_dict.get("sentiment", "Unknown"))
            with cols[4]:
                st.caption("Word Count")
                st.write(chunk_dict.get("wc", 0))
            
            # Display chunk text
            st.divider()
            if chunk_text:
                st.markdown(chunk_text)
            else:
                st.markdown("No text available for this chunk.")

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
    dataset = load_and_validate_json(file_path, validate_and_update_dataset_meta)

    # Display dataset content
    if dataset:
        st.markdown("---")
        st.info(f"{selected_dataset}")
        
        metrics_tab, collections_tab, generation_tab = st.tabs(["Dataset Metrics", "Collections Table", "Text Generation"])
        
        with metrics_tab:
            display_dataset_metrics(dataset)
        
        with collections_tab:
            display_collections_table(dataset)
            
        with generation_tab:
            display_text_generation_tab(file_path, dataset)
        
else:
    st.info("Generate and select a dataset to view its content.")