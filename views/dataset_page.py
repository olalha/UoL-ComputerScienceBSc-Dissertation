import io
import contextlib
import streamlit as st
import pandas as pd
import asyncio
import json
from typing import Dict, Any

from utils.settings_manager import get_setting
from prompt_manager.prompt_builder import list_available_templates
from view_components.item_selector import saved_file_selector, get_files_list, get_selected_file, add_new_file_and_select
from view_components.file_loader import load_and_validate_rulebook, validate_and_save_dataset, load_and_validate_dataset
from dataset_manager.dataset_structurer import create_dataset_structure
from dataset_manager.text_generator import generate_collection_texts_multi_prompt, generate_collection_texts_single_prompt
from dataset_manager.dataset_visualizer import (
    plot_collection_distribution, 
    plot_topic_distribution, 
    plot_sentiment_pie_chart, 
    plot_sentiment_box_plot, 
    get_dataset_copy_without_text
)
from dataset_manager.dataset_analyser import (
    get_basic_counts, 
    get_text_presence_percentages, 
    get_min_max_counts, 
    get_unique_topics, 
    get_unique_sentiments, 
    get_collection_metrics, 
    filter_collections,
    compare_topic_proportions,
    compare_global_sentiment_proportions,
    compare_topic_sentiment_pair_proportions,
    compare_collection_size_range_distribution
)

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
            rulebook_for_dataset = st.selectbox("Rulebook Selector", rulebooks)
            max_iterations = st.slider(
                "Max Number of Iterations for Optimization:",
                min_value=5000, 
                max_value=100000, 
                value=10000, 
                step=5000
            )
            st.caption("ℹ️ More iterations take longer but may yield better accuracy.")
            
            submitted = st.form_submit_button("Generate Dataset Structure", icon="🛠️")

        # Process the form submission
        if submitted:
            # Load and validate the selected rulebook
            rulebook_data, console_output = load_and_validate_rulebook(rulebook_for_dataset)
            
            # Display console output if any
            if console_output:
                st.text_area("Console Output", console_output, height=200)
            
            # Generate the dataset structure if the rulebook is valid
            if rulebook_data:
                with st.spinner("Generating dataset structure. Please wait...", show_time=True):
                    
                    # Generate dataset structure and capture console output
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        dataset = create_dataset_structure(
                            rulebook_data=rulebook_data,
                            rulebook_file_name=rulebook_for_dataset, 
                            max_iterations=max_iterations
                        )
                    
                    # Display console output if any
                    if captured_output.getvalue():
                        st.text_area("Console Output", captured_output.getvalue(), height=200)

                    if dataset:
                        # Save the dataset structure to a JSON file
                        ds_meta = get_basic_counts(dataset)
                        file_name = f"{dataset['content_title']} - {ds_meta['total_wc']}wc - {ds_meta['total_cc']}cc.json"
                        result_path, console_output = validate_and_save_dataset(file_name, dataset, overwrite=False)
                        
                        # Handle the result of saving the dataset
                        if console_output:
                            st.text_area("Console Output", console_output, height=200)
                            st.error("Failed to save the dataset structure.")
                        if result_path:
                            add_new_file_and_select(result_path.name, 'dataset')
                            st.success("Dataset structure generated successfully!")
                    else:
                        st.error("Failed to generate dataset structure. Please try again.")

@st.cache_data
def plot_collection_distribution_with_st_cache(ds, mode: str) -> None:
    return plot_collection_distribution(ds, mode=mode)

@st.cache_data
def plot_topic_distribution_with_st_cache(ds, mode: str) -> None:
    return plot_topic_distribution(ds, mode=mode)

@st.cache_data
def plot_sentiment_pie_chart_with_st_cache(ds, mode: str) -> None:
    return plot_sentiment_pie_chart(ds, mode=mode)

@st.cache_data
def plot_sentiment_box_plot_with_st_cache(ds) -> None:
    return plot_sentiment_box_plot(ds)

def display_dataset_metrics(dataset: Dict[str, Any]) -> None:
    """ Display comprehensive metrics and visualizations for the dataset. """
    
    st.subheader("Dataset Metrics")
    with st.container(border=True):
        
        # Get basic metrics
        meta = get_basic_counts(dataset)
        meta.update(get_text_presence_percentages(dataset))
        
        cols = st.columns([2, 2, 2, 2, 2])
        with cols[0]:
            st.metric("Total Collections", meta.get("collections_count", 0))
        with cols[1]:
            st.metric("Total Chunks", meta.get("total_cc", 0))
        with cols[2]:
            st.metric("Total Words", meta.get("total_wc", 0))
        with cols[3]:
            st.metric("Chunks w/ Text", f"{meta.get("chunks_text_percent"):.1f}%")
        with cols[4]:
            st.metric("Collections w/ Text", f"{meta.get("collections_text_percent"):.1f}%")
            
        show_visualizations = st.checkbox("Show Visualizations", value=False, help="Display visualizations for dataset metrics.")
        st.caption("ℹ️ Visualizations may take time to load and slow down the application, especially for large datasets.")
        
        # Display all visualizations if enabled
        if show_visualizations:
        
            # Retrieve textless dataset for increased cache hits when preparing visualizations
            ds_without_text = get_dataset_copy_without_text(dataset)
            
            # Pre-generate all visualizations
            visualizations = {
                # Collection Distribution
                'collection_chunk': plot_collection_distribution_with_st_cache(ds_without_text, mode='chunk'),
                'collection_word': plot_collection_distribution_with_st_cache(ds_without_text, mode='word'),
                
                # Topic Distribution
                'topic_chunk': plot_topic_distribution_with_st_cache(ds_without_text, mode='chunk'),
                'topic_word': plot_topic_distribution_with_st_cache(ds_without_text, mode='word'),
                
                # Sentiment Distribution
                'sentiment_chunk': plot_sentiment_pie_chart_with_st_cache(ds_without_text, mode='chunk'),
                'sentiment_word': plot_sentiment_pie_chart_with_st_cache(ds_without_text, mode='word'),
                
                # Word Count Box Plot
                'word_count_box': plot_sentiment_box_plot_with_st_cache(ds_without_text)
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
                
    st.subheader("Rulebook Conformity")
    with st.container(border=True):
        cols = st.columns([2, 2, 2, 2])
        with cols[0]:
            st.metric("Topic Alignment", f"{compare_topic_proportions(dataset):.1f}%")
        with cols[1]:
            st.metric("Setiment Alignment", f"{compare_global_sentiment_proportions(dataset):.1f}%")
        with cols[2]:
            st.metric("T/S-pair Alignment", f"{compare_topic_sentiment_pair_proportions(dataset):.1f}%")
        with cols[3]:
            st.metric("Range Aligment", f"{compare_collection_size_range_distribution(dataset):.1f}%")
        
        st.write(f"Rulebook file name:")
        file_name = dataset.get("rulebook_file_name", "")
        if file_name:
            st.success(file_name)
            st.caption("⚠️ At the time of dataset generation - this may not be the current file name.")
        else:
            st.error("Rulebook file name not found in dataset.")

def display_collections_table(dataset: Dict[str, Any]) -> None:
    """
    Display a table of collections with advanced filtering.
    
    Args:
        dataset: The dataset JSON object
    """

    st.subheader("Collections")
    collections = dataset.get("collections", [])
    
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
    
    for i, collection_index in enumerate(matching_indices):
        # Get the collection using the matching index
        collection = collections[collection_index]
        
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
        
        # Add to data with the actual collection index
        collection_data.append({
            "ID": collection_index,  # Use the actual collection index
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
        hide_index=True,  # Hide DataFrame's index since we're showing collection ID as a column
        use_container_width=True,
    )

def display_collection_veiwer(dataset: Dict[str, Any]) -> None:
    """
    Display text generation interface for a selected collection.
    
    Args:
        dataset: The dataset JSON object
    """
    
    content_title = dataset.get("content_title", "No Title")
    
    st.subheader("Collection Selector")
    with st.container(border=True):
        collections = dataset.get("collections", [])
        
        # Collection selector
        col_count = len(collections)
        col_id = st.number_input(
            "Selected Collection ID", 
            min_value=0, 
            max_value=col_count-1, 
            value=0,
            step=1
        )
        st.caption(f"Collection ID range: 0-{col_count-1}")
    
    # Get selected collection
    selected_collection = collections[col_id]
    
    st.subheader("Text Generation")
    with st.container(border=True):
        
        cols = st.columns([2,2])
        with cols[0]: 
            # Model selector
            models = [i for i in get_setting("OPENAI_LLM_MODELS").values()]
            selected_model = st.selectbox(
                f"LLM Model Selector",
                models,
                key=f"llm_model_selector"
            )
        with cols[1]:
            # Strategy selector
            strategy = st.selectbox(
                "Generation Strategy",
                ["Single-prompt", "Multi-prompt"],
                index=0,
                key="text_gen_strategy",
                help="Single-prompt: Uses a single prompt to generate the complete text at once.\n \
                    Multi-prompt: Generates text for each chunk separately and combines them."
            )
        
        prompt_templates = list_available_templates()
        
        # Prompt template selector
        if strategy == "Single-prompt":
            colletion_template = st.selectbox(
                "Prompt Template",
                prompt_templates,
                index=0,
                key="collection_prompt_template"
            )
        else :
            cols = st.columns([2, 2])
            with cols[0]:
                chunk_template = st.selectbox(
                    "Chunk Prompt Template",
                    prompt_templates,
                    index=0,
                    key="chunk_prompt_template"
                )
            with cols[1]:
                merge_template = st.selectbox(
                    "Merge Prompt Template",
                    prompt_templates,
                    index=0,
                    key="merge_prompt_template"
                )
        
        # Generate collection text button
        collection_text = selected_collection.get("collection_text", "")
        if st.button(f"{"Re-G" if collection_text else "G"}enerate Collection Text", icon="🤖"):
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                with st.spinner("Generating collection text. Please wait...", show_time=True):
                    
                    # Create a new event loop for async operations
                    loop = asyncio.new_event_loop()
                    try:
                        # Generate all texts in parallel
                        if strategy == "Single-prompt":
                            collection_with_generated_text = loop.run_until_complete(
                                generate_collection_texts_single_prompt(
                                    all_collections=collections,
                                    collections_to_process=[col_id],
                                    review_item=content_title,
                                    model=selected_model,
                                    prompt=colletion_template
                                )
                            )
                        else:
                            collection_with_generated_text = loop.run_until_complete(
                                generate_collection_texts_multi_prompt(
                                    all_collections=collections,
                                    collections_to_process=[col_id],
                                    review_item=content_title,
                                    model=selected_model,
                                    chunk_prompt=chunk_template,
                                    merge_prompt=merge_template
                                )
                            )
                    finally:
                        loop.close()

            # Save the updated dataset
            dataset_path, console_output = validate_and_save_dataset(
                get_selected_file('dataset'), 
                dataset, 
                overwrite=True
            )
            
            # Save status of text generation for display after rerun
            if dataset_path and collection_with_generated_text:
                st.session_state['collection_text_gen'] = len(collection_with_generated_text)
            else:
                # Indicate saving failed with a negative count
                st.session_state['collection_text_gen'] = -1
                
            # Store the console output
            full_console_output = captured_output.getvalue() + console_output
            if full_console_output:
                st.session_state['collection_gen_console_output'] = full_console_output
            
            # Update the page to show the generated text
            st.rerun()
    
    # Display success or failure message from previous run if any
    if 'collection_text_gen' in st.session_state:
        generated_collections = st.session_state['collection_text_gen']
        if generated_collections == -1:
            st.error("Failed to save the dataset after generating text.")
        elif generated_collections == 0:
            st.warning("Failed to generate text for the selected collection.")
        else:
            st.success(f"Successfully generated text for selected collection.")
        del st.session_state['collection_text_gen']
        
    # Display console output from previous run if any
    if 'collection_gen_console_output' in st.session_state:
        st.text_area("Console Output", st.session_state['collection_gen_console_output'], height=200)
        del st.session_state['collection_gen_console_output']
        
    st.divider()
    st.subheader("Collection Information")
    
    # Display metrics for the selected collection
    collection_meta = get_collection_metrics(dataset, col_id)
    with st.container(border=True):
        cols = st.columns([2, 1, 2, 2, 2])
        with cols[0]:
            st.metric(f"Collection ID", col_id)
        with cols[1]:
            st.empty()
        with cols[2]:
            st.metric("Chunk Count", collection_meta.get("collection_cc", "Error"))
        with cols[3]:
            st.metric("Given Word Count", collection_meta.get("collection_wc", "Error"))
        with cols[4]:
            collection_text_len = len(str(selected_collection.get("collection_text", "")).split())
            st.metric("Actual Word Count", collection_text_len if collection_text_len > 1 else "N/A")
    
    # Display the generated text
    collection_text = selected_collection.get("collection_text")
    if collection_text:
        with st.expander("Collection Text", expanded=True, icon="💬"):
            st.markdown(collection_text)
            
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

def display_text_generator(dataset: Dict[str, Any]) -> None:
    """
    Display interface for generating text for all collections at once.
    
    Args:
        dataset: The dataset JSON object
    """
    
    # Display success or failure message from previous run if any
    if 'failed_generations_count' in st.session_state:
        failed_generations = st.session_state['failed_generations_count']
        if failed_generations == -1:
            st.error("Failed to save the dataset after generating text.")
        elif failed_generations > 0:
            st.warning(f"Failed to generate text for {failed_generations} collections.")
        else:
            st.success("All collections have text generated.")
        del st.session_state['failed_generations_count']
    
    # Display console output from previous run if any
    if 'bulk_gen_console_output' in st.session_state:
        st.text_area("Console Output", st.session_state['bulk_gen_console_output'], height=200)
        del st.session_state['bulk_gen_console_output']
    
    st.subheader("Bulk Text Generation")
    
    # Get text statistics
    meta = get_basic_counts(dataset)
    meta.update(get_text_presence_percentages(dataset))
    content_title = dataset.get("content_title", "Dataset")
    
    # Get collections without text
    collections = dataset.get("collections", [])
    collections_without_text = [i for i, col in enumerate(collections) if not col.get("collection_text")]
    
    # Display current text coverage stats - simplified to show only percentages
    with st.container(border=True):
        cols = st.columns([2, 2, 2, 2])
        
        with cols[0]:
            st.metric("Total Collections", meta.get("collections_count", 0))
        with cols[1]:
            st.metric("Collections w/o Text", len(collections_without_text))
        with cols[2]:
            st.metric("Collections w/ Text", f"{meta.get('collections_text_percent', 0):.1f}%")
        with cols[3]:
            st.metric("Collections w/o Text", f"{100 - meta.get('collections_text_percent', 0):.1f}%")
    
    # Help text
    st.write("ℹ️ For individual collection text generation, use the Collection Viewer tab.")
    
    # Text generation settings
    with st.container(border=True):
        cols = st.columns([2, 2])
        with cols[0]:
            models = [i for i in get_setting("OPENAI_LLM_MODELS").values()]
            selected_model = st.selectbox(
                f"LLM Model Selector",
                models,
                key=f"bulk_llm_model_selector"
            )
        with cols[1]:
            # Strategy selector
            strategy = st.selectbox(
                "Generation Strategy",
                ["Single-prompt", "Multi-prompt"],
                index=0,
                key="bulk_text_gen_strategy",
                help="Single-prompt: Uses a single prompt to generate the complete text at once.\n \
                    Multi-prompt: Generates text for each chunk separately and combines them."
            )
        
        if strategy == "Single-prompt":
            colletion_template = st.selectbox(
                "Prompt Template",
                list_available_templates(),
                index=0,
                key="bulk_collection_prompt_template"
            )
        else:
            cols = st.columns([2, 2])
            with cols[0]:
                chunk_template = st.selectbox(
                    "Chunk Prompt Template",
                    list_available_templates(),
                    index=0,
                    key="bulk_chunk_prompt_template"
                )
            with cols[1]:
                merge_template = st.selectbox(
                    "Merge Prompt Template",
                    list_available_templates(),
                    index=0,
                    key="bulk_merge_prompt_template"
                )
    
        # Generate text button with disabled state if all collections have text
        are_empty_collections = len(collections_without_text) == 0
        if not are_empty_collections:
            btn_text = f"Generate Text for {len(collections_without_text)} Collections"
            collection_to_process = collections_without_text
        else:
            btn_text = f"Re-Generate Text for {len(collections)} Collections"
            collection_to_process = [i for i in range(len(collections))]
        if st.button(btn_text, icon="🤖"):
            
            # Prepare for console output capture
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                with st.spinner("Generating text for all collections without text. This may take a while...", show_time=True):
                    
                    # Create a new event loop for async operations
                    loop = asyncio.new_event_loop()
                    try:
                        # Generate all texts in parallel
                        if strategy == "Single-prompt":
                            successful_collections = loop.run_until_complete(
                                generate_collection_texts_single_prompt(
                                    all_collections=collections,
                                    collections_to_process=collection_to_process,
                                    review_item=content_title,
                                    model=selected_model,
                                    prompt=colletion_template
                                )
                            )
                        else:
                            successful_collections = loop.run_until_complete(
                                generate_collection_texts_multi_prompt(
                                    all_collections=collections,
                                    collections_to_process=collection_to_process,
                                    review_item=content_title,
                                    model=selected_model,
                                    chunk_prompt=chunk_template,
                                    merge_prompt=merge_template
                                )
                            )
                    finally:
                        loop.close()
            
            # Save the updated dataset
            dataset_path, console_output = validate_and_save_dataset(
                get_selected_file('dataset'), 
                dataset, 
                overwrite=True
            )
            
            # Save status of text generation for display after rerun
            if dataset_path and successful_collections:
                successful_generations = len(successful_collections)
                failed_generations = len(collections_without_text) - successful_generations
                st.session_state['failed_generations_count'] = failed_generations
            else:
                # Indicate saving failed with a negative count
                st.session_state['failed_generations_count'] = -1
            
            # Store the console output
            full_console_output = captured_output.getvalue() + console_output
            if full_console_output:
                st.session_state['bulk_gen_console_output'] = full_console_output
            
            # Update the page
            st.rerun()

def display_export_options(dataset: Dict[str, Any]) -> None:
    """
    Display options for exporting the dataset in different formats.
    
    Args:
        dataset: The dataset JSON object
    """
    st.subheader("Export Dataset")
    
    # Content title for default filenames
    content_title = dataset.get("content_title", "dataset")
    
    with st.container(border=True):
        st.markdown("#### Download Dataset JSON")
        st.write("Download the complete dataset structure as a JSON file.")
        
        # Prepare JSON data for download
        json_data = json.dumps(dataset, indent=4)
        
        # Text field for customizing filename
        json_filename = st.text_input(
            "JSON Filename", 
            value=f"{content_title}.json",
            key="json_export_filename"
        )

        # Create download button
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=json_filename,
            mime="application/json",
            key="download_dataset_json",
            help="Download the full dataset structure as a JSON file",
            type="primary"
        )
    
    with st.container(border=True): 
        st.markdown("#### Download Collection Texts")
        st.write("Download all collection texts as a single TXT file.")
        
        # Divider characters for separating collection texts
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Divider", value=f"{10*'-'}", key="collection_text_divider")
            st.caption("Separator between collection texts.")
        with col2:
            st.selectbox("Text Encoding", ["UTF-8", "UTF-16", "ISO-8859-1"], index=0, key="text_encoding")
        
        # Get all collection texts
        collections = dataset.get("collections", [])
        collection_texts = []
        divider = f"\n\n{st.session_state['collection_text_divider']}\n\n"
        for i, collection in enumerate(collections):
            text = collection.get("collection_text", "")
            if text:
                collection_texts.append(f"{divider if i > 0 else ""}{text}")
        encoding = st.session_state['text_encoding']
        export_text = f"".join(collection_texts).encode("utf-8" if encoding is None else encoding)
        collections_with_text = sum(1 for col in collections if col.get("collection_text"))
         
        # Text field for customizing filename
        txt_filename = st.text_input(
            "TXT Filename", 
            value=f"{content_title}.txt",
            key="txt_export_filename"
        )
        
        # Create download button (disabled if no texts available)
        st.download_button(
            label="Download TXT",
            data=export_text,
            file_name=txt_filename,
            mime="text/plain",
            key="download_collection_texts",
            help="Download all collection texts as a single TXT file",
            disabled=len(export_text) == 0,
            type="primary"
        )
        
        # Show info about text availability
        if collections_with_text == 0:
            st.info("No collection texts available for export.")
        else:
            st.caption(f"{collections_with_text} of {len(collections)} collections have text available for export.")

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
    dataset_structure, console_output = load_and_validate_dataset(selected_dataset)
    if console_output:
        st.text_area("Console Output", console_output, height=200)

    # Display dataset content
    if dataset_structure:
        st.markdown("---")
        st.write("Currently Selected Dataset:")
        st.info(f"{selected_dataset}")

        tab_names = ["Dataset Metrics", "Collections Table", "Collection Viewer", "Text Generator", "Export Data"]
        metrics_tab, table_table, viewer_tab, generation_tab, export_tab = st.tabs(tab_names)
        with metrics_tab:
            display_dataset_metrics(dataset_structure)
        with table_table:
            display_collections_table(dataset_structure)
        with viewer_tab:
            display_collection_veiwer(dataset_structure)
        with generation_tab:
            display_text_generator(dataset_structure)
        with export_tab:
            display_export_options(dataset_structure)

else:
    st.info("Generate and select a dataset to view its content.")