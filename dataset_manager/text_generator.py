import asyncio

from typing import Optional, List, Dict
from prompt_manager.prompt_builder import render_prompt
from utils.api_request_handler import prompt_openai_llm_parallel, prompt_openai_llm_single

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