import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, Any, List, Optional

from analysis_manager.dataset_analyser import (
    get_collection_sentiment_data,
    get_topic_sentiment_distribution, 
    get_sentiment_distribution,
    get_chunk_data_for_analysis
)

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
FIG_SIZE = (6, 4)

def get_dataset_copy_without_text(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes text content from chunks and collections in the dataset.
    
    This function sets all 'chunk_text' and 'collection_text' fields to null
    while preserving the metrics data. Used for caching purposes to prevent
    unnecessary recalculation of metrics when only text content changes in
    the dataset.
    
    Args:
        dataset: The dataset JSON object
        
    Returns:
        Dict[str, Any]: The dataset with text content removed
    """
    if not dataset or "collections" not in dataset:
        return dataset
    
    # Create a deep copy to avoid modifying the original dataset
    processed_dataset = deepcopy(dataset)
    
    # Process each collection
    for collection in processed_dataset.get("collections", []):
        # Set collection text to null
        collection["collection_text"] = None
        
        # Process each chunk in the collection
        for chunk in collection.get("chunks", []):
            # Set chunk text to null
            chunk["chunk_text"] = None
    
    return processed_dataset

def plot_collection_distribution(dataset: Dict[str, Any], mode: str) -> Optional[plt.Figure]:
    """
    Create a stacked bar chart showing sentiment distribution by collection.
    
    Args:
        dataset: The dataset JSON object
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Validate mode parameter
    if mode not in ["chunk", "word"]:
        print("plot_collection_distribution: Invalid mode parameter")
        return None
    
    # Check if data is available
    if not dataset:
        print("plot_collection_distribution: No dataset available")
        return None
    
    # Get sentiment distribution by collection
    collection_sentiment = get_collection_sentiment_data(dataset, mode)
    
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
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Set up x positions for bars
    x = np.arange(len(x_labels))
    
    # Create stacked bars for sentiments
    ax.bar(x, data["pos_vals"], label='Positive', color=SENTIMENT_COLORS['positive'], alpha=CHART_ALPHA)
    ax.bar(x, data["neu_vals"], bottom=data["pos_vals"], label='Neutral', color=SENTIMENT_COLORS['neutral'], alpha=CHART_ALPHA)
    
    # Calculate the bottom position for negative values and add them
    bottom_pos = [data["pos_vals"][i] + data["neu_vals"][i] for i in range(len(data["pos_vals"]))]
    ax.bar(x, data["neg_vals"], bottom=bottom_pos, label='Negative', color=SENTIMENT_COLORS['negative'], alpha=CHART_ALPHA)
    
    # Set chart labels and title
    metric_type = mode.capitalize() + "s"
    ax.set_xlabel("Collections")
    ax.set_ylabel(f'Count ({metric_type})')
    
    # Set x-axis ticks
    ax.set_xticks(x)
    
    # For collections, hide labels to avoid overcrowding
    ax.set_xticklabels([""] * len(x_labels))
    
    # Add legend and adjust layout
    ax.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)
    
    return fig

def plot_topic_distribution(dataset: Dict[str, Any], mode: str = "chunk") -> Optional[plt.Figure]:
    """
    Create a stacked bar chart showing sentiment distribution by topic.
    
    Args:
        dataset: The dataset JSON object
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Validate mode parameter
    if mode not in ["chunk", "word"]:
        print("plot_topic_distribution: Invalid mode parameter")
        return None
    
    # Check if data is available
    if not dataset:
        print("plot_topic_distribution: No dataset available")
        return None
    
    # Get distribution data based on count type
    distribution = get_topic_sentiment_distribution(dataset, mode)
    
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
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Set up x positions for bars
    x = np.arange(len(x_labels))
    
    # Create stacked bars for sentiments
    ax.bar(x, data["pos_vals"], label='Positive', color=SENTIMENT_COLORS['positive'], alpha=CHART_ALPHA)
    ax.bar(x, data["neu_vals"], bottom=data["pos_vals"], label='Neutral', color=SENTIMENT_COLORS['neutral'], alpha=CHART_ALPHA)
    
    # Calculate the bottom position for negative values and add them
    bottom_pos = [data["pos_vals"][i] + data["neu_vals"][i] for i in range(len(data["pos_vals"]))]
    ax.bar(x, data["neg_vals"], bottom=bottom_pos, label='Negative', color=SENTIMENT_COLORS['negative'], alpha=CHART_ALPHA)
    
    # Set chart labels and title
    metric_type = mode.capitalize() + "s"
    ax.set_xlabel("Topics")
    ax.set_ylabel(f'Count ({metric_type})')
    
    # Set x-axis ticks
    ax.set_xticks(x)
    
    # Truncate long topic labels
    truncated_labels = [label[:20] + "..." if len(label) > 20 else label for label in x_labels]
    if len(x_labels) > 8:
        # Rotate labels for better readability when there are many
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels(truncated_labels)
    
    # Add legend and adjust layout
    ax.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=GRID_ALPHA)
    
    return fig

def plot_sentiment_pie_chart(dataset: Dict[str, Any], mode: str) -> Optional[plt.Figure]:
    """
    Create a pie chart showing sentiment distribution by either chunk count or word count.
    
    Args:
        dataset: The dataset JSON object
        by_chunks: If True, use chunk count; otherwise use word count
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if mode not in ["chunk", "word"]:
        print("plot_sentiment_pie_chart: Invalid mode parameter")
        return None
    
    # Check if data is available
    if not dataset:
        print("plot_sentiment_pie_chart: No dataset available")
        return None
    
    # Get the appropriate sentiment distribution data
    sentiment_data = get_sentiment_distribution(dataset, mode)
        
    # Create the pie chart
    fig, ax = plt.subplots(figsize=FIG_SIZE)
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
    return fig

def plot_sentiment_box_plot(dataset: Dict[str, Any]) -> Optional[plt.Figure]:
    """ 
    Create a box plot representing the word count distribution of chunks grouped by sentiment. 
    
    Args:
        dataset: The dataset JSON object
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if not dataset:
        print("plot_sentiment_box_plot: No dataset available")
        return None
    
    # Extract chunk data from collections using the analyser function
    chunks_data = get_chunk_data_for_analysis(dataset)
    
    # Check if data is available
    if not chunks_data:
        return None
        
    # Create DataFrame for analysis
    df = pd.DataFrame(chunks_data)
    
    # Define the sentiment order for plotting
    sentiment_order = ['positive', 'neutral', 'negative', 'All']
    available_sentiments = [s for s in sentiment_order if s in df['sentiment'].unique() or s == 'All']
    
    # Prepare data for the box plot - group by sentiment
    data_to_plot = [df.loc[df['sentiment'] == s, 'wc'] if s != 'All' else df['wc'] for s in available_sentiments]
    
    # Create the box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
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
    return fig
