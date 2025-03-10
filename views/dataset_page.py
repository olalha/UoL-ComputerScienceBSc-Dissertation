import io
import contextlib
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from utils.settings_manager import get_setting
from view_components.alerter import show_alert
from view_components.item_selector import saved_file_selector, change_selected_file, get_files_list, get_selected_file, add_new_file_to_selector
from view_components.file_loader import load_and_validate_rulebook, validate_and_save_dataset, load_and_validate_dataset
from dataset_manager.dataset_structurer import create_dataset_structure
from dataset_manager.text_generator import generate_collection_text
from dataset_manager.dataset_visualizer import plot_collection_distribution, plot_topic_distribution, plot_sentiment_pie_chart, plot_sentiment_box_plot
from dataset_manager.dataset_analyser import get_basic_counts, get_min_max_counts, get_unique_topics, get_unique_sentiments, filter_collections

# Display alert if it exists in session state
if st.session_state.stored_alert:
    show_alert()

# Initialize session state for form submission
if 'dataset_form_submitted' not in st.session_state:
    st.session_state.dataset_form_submitted = False

def generate_dataset_structure_form() -> None:
    """ Displays a form for generating dataset structures from rulebooks. """
    
    # Get all available rulebooks
    rulebooks = get_files_list('rulebook')
    if not rulebooks:
        st.info("No rulebooks found. Please upload a rulebook first.")
        return
    
    # Display the form for generating dataset structure
    with st.expander("Generate Dataset From Rulebook", icon="📚", expanded=True):
        with st.form(key="generate_dataset_form", border=False):
            dataset_selected_rulebook = st.selectbox("Rulebook Selector", rulebooks)
            st.write("Warning: Invalid rulebooks will not be displayed.")
            datatset_search_time_s = st.slider("Solution Search Time (seconds)", min_value=1, max_value=60, value=5)
            submitted = st.form_submit_button("Generate Dataset Structure")

            if submitted:
                st.session_state.dataset_form_submitted = True
                st.session_state.dataset_selected_rulebook = dataset_selected_rulebook
                st.session_state.datatset_search_time_s = datatset_search_time_s

    # When the form is submitted, generate the dataset structure
    if st.session_state.dataset_form_submitted:
        dataset_selected_rulebook = st.session_state.dataset_selected_rulebook
        datatset_search_time_s = st.session_state.datatset_search_time_s

        # Load and validate the selected rulebook
        rulebook, console_output = load_and_validate_rulebook(dataset_selected_rulebook)
        
        # Display console output if any
        if console_output:
            st.text_area("Console Output", console_output, height=200)
        
        # Generate the dataset structure if the rulebook is valid
        if rulebook:
            with st.spinner("Generating dataset structure. Please wait...", show_time=True):
                
                captured_output = io.StringIO()
                with contextlib.redirect_stdout(captured_output):
                    # Generate dataset structure
                    dataset = create_dataset_structure(rulebook=dataset_selected_rulebook, 
                                                        solution_search_time_s=datatset_search_time_s)
                
                # Display console output if any
                if captured_output.getvalue():
                    st.text_area("Console Output", captured_output.getvalue, height=200)

                if dataset:
                    # Save the dataset structure to a JSON file
                    ds_meta = get_basic_counts(dataset)
                    file_name = f"{dataset['content_title']} - {ds_meta['total_wc']}wc - {ds_meta['total_cc']}cc.json"
                    result_path, console_output = validate_and_save_dataset(file_name, dataset)
                    
                    # Handle the result of saving the dataset
                    if console_output:
                        st.text_area("Console Output", console_output, height=200)
                    if result_path:
                        st.success(f"File processed successfully! Saved to {result_path}")
                        add_new_file_to_selector('dataset', result_path.name)
                        change_selected_file('dataset', result_path.name)
                else:
                    st.error("Failed to generate dataset structure. Please try again.")

def display_dataset_metrics(dataset: Dict[str, Any]) -> None:
    """ Display comprehensive metrics and visualizations for the dataset. """
    
    st.subheader("Dataset Metrics")
    with st.container(border=True):
        
        # Use a spinner while generating all visualizations
        with st.spinner("Calculating metrics and generating visualizations..."):
            # Pre-generate all visualizations
            visualizations = {
                # Collection Distribution
                'collection_chunk': plot_collection_distribution(dataset, mode='chunk'),
                'collection_word': plot_collection_distribution(dataset, mode='word'),
                
                # Topic Distribution
                'topic_chunk': plot_topic_distribution(dataset, mode='chunk'),
                'topic_word': plot_topic_distribution(dataset, mode='word'),
                
                # Sentiment Distribution
                'sentiment_chunk': plot_sentiment_pie_chart(dataset, mode='chunk'),
                'sentiment_word': plot_sentiment_pie_chart(dataset, mode='word'),
                
                # Word Count Box Plot
                'word_count_box': plot_sentiment_box_plot(dataset)
            }
        
            # Collection Size Distribution (Stacked Bar Chart)
            with st.expander("Collection Size Distribution", expanded=False, icon="📈"):
                st.subheader("Collection Size Distribution")
                tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
                with tab1:
                    if visualizations['collection_chunk']:
                        st.pyplot(visualizations['collection_chunk'])
                    else:
                        st.info("No chunk count distribution data available by collection.")
                    
                with tab2:
                    if visualizations['collection_word']:
                        st.pyplot(visualizations['collection_word'])
                    else:
                        st.info("No word count distribution data available by collection.")
            
            # Topic Coverage Distribution (Stacked Bar Chart)
            with st.expander("Topic Coverage Distribution", expanded=False, icon="💬"):
                st.subheader("Topic Coverage Distribution")
                tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
                with tab1:
                    if visualizations['topic_chunk']:
                        st.pyplot(visualizations['topic_chunk'])
                    else:
                        st.info("No chunk count distribution data available by topic.")
                    
                with tab2:
                    if visualizations['topic_word']:
                        st.pyplot(visualizations['topic_word'])
                    else:
                        st.info("No word count distribution data available by topic.")
            
            # Overall Sentiment Distribution (Pie Charts and Box Plot)
            with st.expander("Overall Sentiment Distribution", expanded=False, icon="😃"):
                st.subheader("Overall Sentiment Distribution")
                tab1, tab2 = st.tabs(["By Chunk Count (cc)", "By Word Count (wc)"])
                with tab1:
                    if visualizations['sentiment_chunk']:
                        st.pyplot(visualizations['sentiment_chunk'])
                    else:
                        st.info("No chunk count sentiment distribution data available.")
                with tab2:
                    if visualizations['sentiment_word']:
                        st.pyplot(visualizations['sentiment_word'])
                    else:
                        st.info("No word count sentiment distribution data available.")
                
            # Word Count Distribution by Chunk (Box Plot)
            with st.expander("Word Count Distribution by Chunk", expanded=False, icon="📊"):
                st.subheader("Word Count Distribution by Chunk")
                if visualizations['word_count_box']:
                    st.pyplot(visualizations['word_count_box'])
                else:
                    st.info("No word count data available for box plot.")

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
    
    # Get all available topics and sentiments
    all_topics = get_unique_topics(dataset)
    all_sentiments = get_unique_sentiments(dataset)
    
    # Get min/max counts for the dataset
    min_max_counts = get_min_max_counts(dataset)
    max_wc = min_max_counts["max_collection_wc"]
    max_cc = min_max_counts["max_collection_cc"]
    
    # Display filter UI
    with st.container(border=True):
        st.markdown("#### Filter Collections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word count range slider - apply directly
            wc_range = st.slider(
                "Word Count Range",
                min_value=0,
                max_value=max_wc,
                value=(0, max_wc),
                key="wc_range"
            )
            
            # Topic select box
            topic_filter = st.selectbox(
                "Filter by Topic",
                options=["All"] + sorted(list(all_topics)),
                index=0,
                key="topic_filter"
            )
            
            # Text status selectbox
            text_status = st.selectbox(
                "Filter by Text Status",
                options=["All", "Has Text", "No Text"],
                index=0,
                key="text_status_filter"
            )
        
        with col2:
            # Chunk count range slider
            cc_range = st.slider(
                "Chunk Count Range",
                min_value=0,
                max_value=max_cc,
                value=(0, max_cc),
                key="cc_range"
            )
            
            # Sentiment select box
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=["All"] + sorted(list(all_sentiments)),
                index=0,
                key="sentiment_filter"
            )
    
    # Convert filter values to appropriate types for filter_collections
    min_wc, max_wc = wc_range
    min_cc, max_cc = cc_range
    topic = None if topic_filter == "All" else topic_filter
    sentiment = None if sentiment_filter == "All" else sentiment_filter
    has_text = None
    if text_status == "Has Text":
        has_text = True
    elif text_status == "No Text":
        has_text = False
    
    # Apply filters using filter_collections
    matching_indices = filter_collections(
        dataset, 
        min_wc=min_wc,
        max_wc=max_wc, 
        min_cc=min_cc, 
        max_cc=max_cc,
        topic=topic,
        sentiment=sentiment,
        has_text=has_text
    )
    
    # Show filter summary if any filters are applied
    filter_summary = []
    if min_wc > 0 or max_wc < min_max_counts["max_chunk_wc"] * min_max_counts["max_collection_cc"]:
        filter_summary.append(f"Words: {min_wc}-{max_wc}")
    
    if min_cc > 0 or max_cc < min_max_counts["max_collection_cc"]:
        filter_summary.append(f"Chunks: {min_cc}-{max_cc}")
    
    if topic:
        filter_summary.append(f"Topic: {topic}")
        
    if sentiment:
        filter_summary.append(f"Sentiment: {sentiment}")
        
    if has_text is not None:
        filter_summary.append(f"Text: {'Has Text' if has_text else 'No Text'}")
        
    if filter_summary:
        st.caption(f"Filters applied: {', '.join(filter_summary)}")
    
    # Create the filtered collection data for the table
    filtered_collections = [collections[i] for i in matching_indices]
    collection_data = []
    
    max_chunk_count = max([len(c.get("chunks", [])) for c in collections]) if collections else 0
    
    for i, collection in enumerate(filtered_collections):
        # Get collection metrics
        cc = len(collection.get("chunks", []))
        wc = sum([chunk.get("chunk_dict", {}).get("wc", 0) for chunk in collection.get("chunks", [])])
        has_text = collection.get("collection_text") is not None
        
        # Get topics and sentiments with counts
        topic_sentiment = {}
        chunk_word_counts = []
        
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
    max_chunk_wc = max([max(row["Chunk Distribution"]) if row["Chunk Distribution"] else 0 for row in collection_data]) + 5 if collection_data else 5
    
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

def display_text_generation_tab(dataset: Dict[str, Any]) -> None:
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
    content_title = dataset.get("content_title", None)
    if not content_title:
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
        with contextlib.redirect_stdout(captured_output):
            with st.spinner("Generating collection text. Please wait...", show_time=True):
                collection_with_generated_text = generate_collection_text(selected_collection, content_title, selected_model)
        
        # Display console output if any
        if captured_output.getvalue():
            st.text_area("Console Output", captured_output.getvalue(), height=200)
        
        # Update the collection with generated text if successful
        if collection_with_generated_text:
            collections[col_index] = collection_with_generated_text
            dataset["collections"] = collections
            dataset, console_output = validate_and_save_dataset(get_selected_file('dataset'), dataset)
            if console_output:
                st.text_area("Console Output", console_output, height=200)
            if dataset:
                st.success("Collection text generated successfully.")
    
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
generate_dataset_structure_form()

# --- Saved Datasets Section ---
st.header("Saved Datasets")

# Display dataset selector
selected_dataset = saved_file_selector('dataset')

# Display selected dataset
if selected_dataset:
    
    # Load and validate the selected dataset
    dataset, console_output = load_and_validate_dataset(selected_dataset)
    if console_output:
        st.text_area("Console Output", console_output, height=200)

    # Display dataset content
    if dataset:
        st.markdown("---")
        st.info(f"{selected_dataset}")
        
        metrics_tab, collections_tab, generation_tab = st.tabs(["Dataset Metrics", "Collections Table", "Text Generation"])
        with metrics_tab:
            if st.button("Caculate Metrics", icon="➗"):
                display_dataset_metrics(dataset)
        with collections_tab:
            display_collections_table(dataset)
        with generation_tab:
            display_text_generation_tab(dataset)
        
else:
    st.info("Generate and select a dataset to view its content.")