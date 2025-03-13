import io
import contextlib
import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, Any

from utils.settings_manager import get_setting
from view_components.item_selector import saved_file_selector, get_files_list, get_selected_file, add_new_file_and_select
from view_components.file_loader import load_and_validate_rulebook, validate_and_save_dataset, load_and_validate_dataset
from dataset_manager.dataset_structurer import create_dataset_structure
from dataset_manager.text_generator import generate_collection_text
from dataset_manager.dataset_visualizer import plot_collection_distribution, plot_topic_distribution, plot_sentiment_pie_chart, plot_sentiment_box_plot, get_dataset_copy_without_text
from dataset_manager.dataset_analyser import get_basic_counts, get_text_presence_percentages, get_min_max_counts, get_unique_topics, get_unique_sentiments, get_collection_metrics, filter_collections

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
            st.write("Warning: Invalid rulebooks will not be displayed.")
            search_time_for_dataset = st.slider("Solution Search Time (seconds)", min_value=1, max_value=60, value=5)
            submitted = st.form_submit_button("Generate Dataset Structure")

    # Process the form submission
    if submitted:
        # Load and validate the selected rulebook
        rulebook, console_output = load_and_validate_rulebook(rulebook_for_dataset)
        
        # Display console output if any
        if console_output:
            st.text_area("Console Output", console_output, height=200)
        
        # Generate the dataset structure if the rulebook is valid
        if rulebook:
            with st.spinner("Generating dataset structure. Please wait...", show_time=True):
                
                # Generate dataset structure and capture console output
                captured_output = io.StringIO()
                with contextlib.redirect_stdout(captured_output):
                    dataset = create_dataset_structure(rulebook=rulebook, solution_search_time_s=search_time_for_dataset)
                
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
                    if result_path:
                        add_new_file_and_select(result_path.name, 'dataset')
                else:
                    st.error("Failed to generate dataset structure. Please try again.")

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
        
        # Retrieve textless dataset for increased cache hits when preparing visualizations
        ds_without_text = get_dataset_copy_without_text(dataset)
        
        # Pre-generate all visualizations
        visualizations = {
            # Collection Distribution
            'collection_chunk': plot_collection_distribution(ds_without_text, mode='chunk'),
            'collection_word': plot_collection_distribution(ds_without_text, mode='word'),
            
            # Topic Distribution
            'topic_chunk': plot_topic_distribution(ds_without_text, mode='chunk'),
            'topic_word': plot_topic_distribution(ds_without_text, mode='word'),
            
            # Sentiment Distribution
            'sentiment_chunk': plot_sentiment_pie_chart(ds_without_text, mode='chunk'),
            'sentiment_word': plot_sentiment_pie_chart(ds_without_text, mode='word'),
            
            # Word Count Box Plot
            'word_count_box': plot_sentiment_box_plot(ds_without_text)
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
    
    # Display generation button
    models = [i for i in get_setting("OPENAI_LLM_MODELS").values()]
    selected_model = st.selectbox(
        f"LLM Model Selector",
        models,
        key=f"llm_model_selector"
    )
    collection_text = selected_collection.get("collection_text", "")
    
    # Generate collection text button
    if st.button(f"{"Re-G" if collection_text else "G"}enerate Collection Text", icon="🤖"):
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            with st.spinner("Generating collection text. Please wait...", show_time=True):
                # Create a new event loop for async operations
                loop = asyncio.new_event_loop()
                
                try:
                    # Run the text generation in the event loop
                    collection_with_generated_text = loop.run_until_complete(
                        generate_collection_text(selected_collection, content_title, selected_model)
                    )
                finally:
                    # Always close the loop
                    loop.close()
        
        # Display console output if any
        if captured_output.getvalue():
            st.text_area("Console Output", captured_output.getvalue(), height=200)
        
        # Update the dataset with generated text if successful
        if collection_with_generated_text:
            collections[col_id] = collection_with_generated_text
            dataset["collections"] = collections
            dataset_path, console_output = validate_and_save_dataset(get_selected_file('dataset'), dataset, overwrite=True)
            if console_output:
                st.text_area("Console Output", console_output, height=200)
            if dataset_path:
                st.session_state['generation_success'] = True
                st.rerun()
        else:
            st.error("Failed to generate text. Please try again.")
    
    # Display success message if text generation was successful            
    if 'generation_success' in st.session_state:
        st.success("Text generation completed successfully.")
        del st.session_state['generation_success']
        
    st.divider()
    st.subheader("Collection Information")
    
    # Display the generated text
    collection_text = selected_collection.get("collection_text")
    if collection_text:
        with st.expander("Collection Text", expanded=True, icon="💬"):
            st.markdown(collection_text)
    
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
    
    st.subheader("Bulk Text Generation")
    
    # Get text statistics
    meta = get_basic_counts(dataset)
    meta.update(get_text_presence_percentages(dataset))
    
    # Get collections without text
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
    st.warning("This tool generates text for all collections that don't have text yet. \
              For individual collection text generation, use the Collection Viewer tab.")
    
    # Display generation button
    models = [i for i in get_setting("OPENAI_LLM_MODELS").values()]
    selected_model = st.selectbox(
        f"LLM Model Selector",
        models,
        key=f"bulk_llm_model_selector"
    )
    
    # Generate text button
    if st.button(f"Generate Text for {len(collections_without_text)} Collections", 
                icon="🤖", 
                disabled=len(collections_without_text) == 0):
        if len(collections_without_text) == 0:
            st.success("All collections already have text!")
        else:
            # Track successful generations
            successful_generations = 0
            failed_generations = 0
            
            # Display progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Prepare for console output capture
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                with st.spinner("Generating text for all collections without text. This may take a while..."):
                    # Create a new event loop for async operations
                    loop = asyncio.new_event_loop()
                    try:
                        for i, col_idx in enumerate(collections_without_text):
                            progress_text.text(f"Processing collection {col_idx} ({i+1}/{len(collections_without_text)})")
                            
                            # Generate text for this collection
                            collection_with_generated_text = loop.run_until_complete(
                                generate_collection_text(collections[col_idx], content_title, selected_model)
                            )
                            
                            # Update collection if generation was successful
                            if collection_with_generated_text:
                                collections[col_idx] = collection_with_generated_text
                                successful_generations += 1
                            else:
                                failed_generations += 1
                                
                            # Update progress
                            progress_bar.progress((i + 1) / len(collections_without_text))
                    finally:
                        loop.close()
            
            # Display console output if any
            if captured_output.getvalue():
                with st.expander("Console Output", expanded=False):
                    st.text_area("", captured_output.getvalue(), height=200)
            
            # Save the updated dataset
            dataset["collections"] = collections
            dataset_path, console_output = validate_and_save_dataset(
                get_selected_file('dataset'), 
                dataset, 
                overwrite=True
            )
            
            if console_output:
                with st.expander("Save Output", expanded=False):
                    st.text_area("", console_output, height=200)
            
            # Display results - only count as success if dataset_path is valid
            if dataset_path:
                st.divider()
                st.write(f"Text generation completed: {successful_generations} successful, {failed_generations} failed.")
                st.divider()
                
                # Show remaining stats
                collections_without_text_after = [i for i, col in enumerate(collections) if not col.get("collection_text")]
                if collections_without_text_after:
                    st.warning(f"{len(collections_without_text_after)} collections still don't have text.")
                    if st.button("Generate Text for Remaining Collections", icon="🔄"):
                        st.rerun()
                else:
                    st.success("All collections now have text!")
                    
                # Update metrics
                new_meta = get_basic_counts(dataset)
                new_meta.update(get_text_presence_percentages(dataset))
                
                with st.container(border=True):
                    cols = st.columns([2, 2])
                    with cols[0]:
                        st.metric("Collections with Text", 
                                f"{new_meta.get('collections_text_percent', 0):.1f}%",
                                delta=f"{new_meta.get('collections_text_percent', 0) - meta.get('collections_text_percent', 0):.1f}%")
                    with cols[1]:
                        st.metric("Chunks with Text", 
                                f"{new_meta.get('chunks_text_percent', 0):.1f}%", 
                                delta=f"{new_meta.get('chunks_text_percent', 0) - meta.get('chunks_text_percent', 0):.1f}%")
            else:
                st.error("Failed to save the dataset. No changes were preserved.")
    
    else:
        if len(collections_without_text) == 0:
            st.success("All collections already have text!")

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
        st.info(f"{selected_dataset}")

        tab_names = ["Dataset Metrics", "Collections Table", "Collection Viewer", "Text Generator"]
        metrics_tab, table_table, viewer_tab, generation_tab = st.tabs(tab_names)
        with metrics_tab:
            display_dataset_metrics(dataset_structure)
        with table_table:
            display_collections_table(dataset_structure)
        with viewer_tab:
            display_collection_veiwer(dataset_structure)
        with generation_tab:
            display_text_generator(dataset_structure)

else:
    st.info("Generate and select a dataset to view its content.")