"""
Text generation functions for collection using the API handler.

This module contains functions to generate text for collections using
two didferent approaches: multi-prompt and single-prompt.

It renders prompts using Jinja2 templates, handles the API calls and
processes the responses. It also includes error graceful handling.
"""

from typing import List
from generation_manager.prompt_builder import render_prompt
from generation_manager.api_handler import prompt_openai_llm_parallel

async def generate_collection_texts_multi_prompt(
    all_collections: List[dict], 
    collections_to_process: List[int], 
    review_item: str, 
    model: str,
    chunk_prompt: str,
    merge_prompt: str) -> List[int]:
    """
    Generate text for multiple collections in parallel using a two-phase approach:
    1. Generate all chunk texts across all collections in parallel
    2. Generate all collection texts in parallel for collections with successful chunks
    
    Args:
        all_collections: The complete list of collections
        collections_to_process: List of collection indices to process
        review_item: The item being reviewed
        model: The LLM model to use
        chunk_prompt: The prompt for generating chunk texts
        merge_prompt: The prompt for combining chunk texts into collection texts
        
    Returns:
        List of collection indices that were successfully processed
    """
    # Phase 1: Generate all chunk texts across all collections in parallel
    all_chunk_messages, chunk_map = [], []
    
    for col_idx in collections_to_process:
        collection = all_collections[col_idx]
        temp_messages, temp_map = [], []
        for chunk_idx, chunk in enumerate(collection['chunks']):
            chunk_dict = chunk['chunk_dict']
            context = {
                'review_item': review_item,
                'topic': chunk_dict['topic'],
                'sentiment': chunk_dict['sentiment'],
                'word_count': chunk_dict['wc']
            }
            prompt = render_prompt(chunk_prompt, context)
            if prompt:
                temp_messages.append([{'role': 'user', 'content': prompt}])
                temp_map.append((col_idx, chunk_idx))
            else:
                print(f"Failed to render prompt for collection: {col_idx}, chunk: {chunk_dict}")
                return None
        else:
            # Executes if no 'break' occurred i.e. all prompts were rendered
            all_chunk_messages.extend(temp_messages)
            chunk_map.extend(temp_map)

    # Generate all chunks in parallel
    if all_chunk_messages:
        all_chunks_responses = await prompt_openai_llm_parallel(model=model, messages=all_chunk_messages)
    
    # Process chunk results
    for i, response in enumerate(all_chunks_responses):
        col_idx, chunk_idx = chunk_map[i]
        
        if response['success']:
            try:
                chunk_text = response['response']['choices'][0]['message']['content']
                # Update the chunk with generated text
                all_collections[col_idx]['chunks'][chunk_idx]['chunk_text'] = chunk_text
            except Exception:
                chunk_dict = all_collections[col_idx]['chunks'][chunk_idx]['chunk_dict']
                print(f"Error processing chunk response for collection: {col_idx}, chunk: {chunk_dict}")
        else:
            print(f"Error generating chunk for collection: {col_idx}, chunk: {chunk_dict}")
    
    # Phase 2: Generate collection texts in parallel for collections with all chunks successful
    collections_ready = []
    collection_messages = []
    
    for col_idx in collections_to_process:
        collection = all_collections[col_idx]
        collection_chunk_texts = [chunk.get('chunk_text') for chunk in collection['chunks']]
        
        # If all chunks have text, generate collection text
        if len(collection_chunk_texts) == len(collection['chunks']):
            prompt_context = {'chunks': collection_chunk_texts}
            prompt = render_prompt(merge_prompt, prompt_context)
            
            if prompt:
                messages_json = [{'role': 'user', 'content': prompt}]
                collection_messages.append(messages_json)
                collections_ready.append(col_idx)
            else:
                print(f"Failed to render collection prompt for collection: {col_idx}")
                return None
    
    # Generate all collection texts in parallel
    if collection_messages:
        collection_responses = await prompt_openai_llm_parallel(model=model, messages=collection_messages)
        
        # Process collection results
        successful_collections = []
        for i, response in enumerate(collection_responses):
            col_idx = collections_ready[i]
            
            if response['success']:
                try:
                    collection_text = response['response']['choices'][0]['message']['content']
                    # Update the collection with generated text
                    all_collections[col_idx]['collection_text'] = collection_text
                    successful_collections.append(col_idx)
                except Exception:
                    print(f"Error processing collection response for collection: {col_idx}")
            else:
                print(f"Failed to generate text for collection: {col_idx}")
        
        return successful_collections
    return None

async def generate_collection_texts_single_prompt(
    all_collections: List[dict],
    collections_to_process: List[int],
    review_item: str,
    model: str,
    prompt: str
) -> List[int]:
    """
    Generate text for multiple collections in parallel using a single prompt per collection.

    Args:
        all_collections: The complete list of collections.
        collections_to_process: List of collection indices to process.
        review_item: The item being reviewed.
        model: The LLM model to use.
        prompt: The prompt template for generating collection texts.

    Returns:
        List of collection indices that were successfully processed.
    """
    collection_messages = []
    collections_ready = []

    for col_idx in collections_to_process:
        collection = all_collections[col_idx]
        # Prepare context for the prompt
        context = {
            'review_item': review_item,
            'total_words': sum(chunk['chunk_dict']['wc'] for chunk in collection.get('chunks', [])),
            'chunks': [chunk['chunk_dict'] for chunk in collection.get('chunks', [])]
        }
        
        # Render the prompt with the context
        rendered_prompt = render_prompt(prompt, context)
        
        if rendered_prompt:
            messages_json = [{'role': 'user', 'content': rendered_prompt}]
            collection_messages.append(messages_json)
            collections_ready.append(col_idx)
        else:
            print(f"Failed to render prompt for collection: {col_idx}")
            return None

    # Generate all collection texts in parallel
    if collection_messages:
        responses = await prompt_openai_llm_parallel(model=model, messages=collection_messages)
        successful_collections = []
        for i, response in enumerate(responses):
            col_idx = collections_ready[i]
            if response['success']:
                try:
                    collection_text = response['response']['choices'][0]['message']['content']
                    all_collections[col_idx]['collection_text'] = collection_text
                    successful_collections.append(col_idx)
                except Exception:
                    print(f"Error processing collection response for collection: {col_idx}")
            else:
                print(f"Failed to generate text for collection: {col_idx}")
        return successful_collections
    return None
