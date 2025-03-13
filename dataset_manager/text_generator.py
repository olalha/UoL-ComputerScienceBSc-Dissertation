import asyncio

from typing import Optional, List, Dict
from prompt_manager.prompt_builder import render_prompt
from utils.api_request_handler import prompt_openai_llm_parallel, prompt_openai_llm_single

async def generate_all_collection_texts_parallel(collections_to_process: List[int], 
                                                 all_collections: List[dict], 
                                                 review_item: str, 
                                                 model: str) -> List[int]:
    """
    Generate text for multiple collections in parallel using a two-phase approach:
    1. Generate all chunk texts across all collections in parallel
    2. Generate all collection texts in parallel for collections with successful chunks
    
    Args:
        collections_to_process: List of collection indices to process
        all_collections: The complete list of collections
        review_item: The item being reviewed
        model: The LLM model to use
        
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
            prompt = render_prompt("usr_chunk_gen.html", context)
            if prompt:
                temp_messages.append([{'role': 'user', 'content': prompt}])
                temp_map.append((col_idx, chunk_idx))
            else:
                print(f"Failed to render prompt for collection: {col_idx}, chunk: {chunk_dict}")
                break
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
            prompt = render_prompt("usr_collection_gen.html", prompt_context)
            
            if prompt:
                messages_json = [{'role': 'user', 'content': prompt}]
                collection_messages.append(messages_json)
                collections_ready.append(col_idx)
            else:
                print(f"Failed to render collection prompt for collection: {col_idx}")
    
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
    return []

async def generate_chunk_text(chunk: dict, review_item: str, model: str) -> Optional[str]:
    """
    Asynchronously generates text for a single chunk.
    
    Args:
        chunk (dict): The chunk to generate text for
        review_item (str): The item being reviewed
        model (str): The LLM model to use
        
    Returns:
        Optional[str]: Generated text for the chunk or None if generation failed
    """
    chunk_dict = chunk['chunk_dict']
    
    # Create prompt for this chunk
    prompt_context = {
        'review_item': review_item,
        'topic': chunk_dict['topic'],
        'sentiment': chunk_dict['sentiment'],
        'word_count': chunk_dict['wc']
    }
    
    prompt = render_prompt("usr_chunk_gen.html", prompt_context)
    if not prompt:
        return None
        
    messages_json = [{'role': 'user', 'content': prompt}]
    
    # Send to LLM API
    response = await prompt_openai_llm_single(model=model, messages=messages_json)
    
    if response['success']:
        try:
            return response['response']['choices'][0]['message']['content']
        except Exception:
            return None
    
    return None

async def generate_collection_text_from_chunks(chunks_texts: List[str], model: str) -> Optional[str]:
    """
    Asynchronously generates collection text from existing chunk texts.
    
    Args:
        chunks_texts (List[str]): The texts from all chunks
        model (str): The LLM model to use
        
    Returns:
        Optional[str]: Generated collection text or None if generation failed
    """
    # Generate collection text
    prompt_context = {'chunks': chunks_texts}
    prompt = render_prompt("usr_collection_gen.html", prompt_context)
    if not prompt:
        return None
        
    messages_json = [{'role': 'user', 'content': prompt}]
    response = await prompt_openai_llm_single(model=model, messages=messages_json)
    
    if response['success']:
        try:
            return response['response']['choices'][0]['message']['content']
        except Exception:
            return None
    
    return None

async def generate_collection_text(collection: dict, review_item: str, model: str) -> Optional[dict]:
    """
    Asynchronously generates text for individual chunks and then combines them into a collection text.
    
    Args:
        collection (dict): The collection with chunks to generate text for
        review_item (str): The item being reviewed
        model (str): The LLM model to use
        
    Returns:
        Optional[dict]: Updated collection with generated text or None if generation failed
    """
    # Step 1: Generate text for all chunks in parallel
    all_chunk_messages = []
    
    for i, chunk in enumerate(collection['chunks']):
        chunk_dict = chunk['chunk_dict']
        
        prompt_context = {
            'review_item': review_item,
            'topic': chunk_dict['topic'],
            'sentiment': chunk_dict['sentiment'],
            'word_count': chunk_dict['wc']
        }
        prompt = render_prompt("usr_chunk_gen.html", prompt_context)
        if not prompt:
            return None
        messages_json = [{'role': 'user', 'content': prompt}]
        all_chunk_messages.append(messages_json)
    
    # Generate text for all chunks in parallel
    chunks_gen_responses = await prompt_openai_llm_parallel(model=model, messages=all_chunk_messages)
    
    # Step 2: Process chunk responses and update collection
    chunk_texts = []
    for r in chunks_gen_responses:
        chunk_idx = r['idx']
        
        if r['success']:
            try:
                generated_text_chunk = r['response']['choices'][0]['message']['content']
                collection['chunks'][chunk_idx]['chunk_text'] = generated_text_chunk
                chunk_texts.append(generated_text_chunk)
            except Exception:
                return None
        else:
            return None
    
    # Step 3: Generate collection text from chunk texts
    if len(chunk_texts) == len(collection['chunks']):
        collection_text = await generate_collection_text_from_chunks(chunk_texts, model)
        if not collection_text:
            return None
        
        collection['collection_text'] = collection_text
        return collection
    
    return None