
import asyncio

from typing import Optional
from prompt_manager.prompt_builder import render_prompt
from utils.api_request_handler import prompt_openai_llm_parallel, prompt_openai_llm_single

def generate_collection_text(collection: list[dict], review_item: str, model: str) -> Optional[list[dict]]:
    
    all_chunk_messages = []
    
    # Generate prompt for each chunk
    for i, chunk in enumerate(collection['chunks']):
        chunk_dict = chunk['chunk_dict']
        
        # Generate prompt
        prompt_context = {
            'review_item': review_item,
            'topic': chunk_dict['topic'],
            'sentiment': chunk_dict['sentiment'],
            'word_count': chunk_dict['wc']
        }
        prompt = render_prompt("usr_chunk_gen.html", prompt_context)
        messages_json = [{'role': 'user', 'content': prompt}]
        all_chunk_messages.append(messages_json)
    
    # Generate text for all chunks
    async def get_chunks_text():
        return await prompt_openai_llm_parallel(model=model, messages=all_chunk_messages)
    
    chunks_gen_responses = asyncio.run(get_chunks_text())
    
    # Update all chunks with generated text
    chunk_texts = []
    for r in chunks_gen_responses:
        chunk_idx = r['idx']
        
        # Extract generated text
        generated_text_chunk = None
        if r['success']:
            try:
                generated_text_chunk = r['response']['choices'][0]['message']['content']
            except Exception as e:
                pass
        
        if generated_text_chunk:
            collection['chunks'][chunk_idx]['chunk_text'] = generated_text_chunk
            chunk_texts.append(generated_text_chunk)
        else:
            print(f"generate_collection_text: Failed to generate text for chunk {chunk_idx}")
            return None
    
    # Generate collection text if all chunks are successful
    if len(chunk_texts) == len(collection['chunks']):
        
        prompt_context = {
            'chunks': chunk_texts
        }
        prompt = render_prompt("usr_collection_gen.html", prompt_context)
        messages_json = [{'role': 'user', 'content': prompt}]
        
        print(messages_json)
        
        collection_gen_response = prompt_openai_llm_single(model=model, messages=messages_json)
        
        generated_text_collection = None
        if collection_gen_response['success']:
            try:
                generated_text_collection = collection_gen_response['response']['choices'][0]['message']['content']
            except Exception as e:
                pass
        
        print(generated_text_collection)
        
        if generated_text_collection:
            collection['collection_text'] = generated_text_collection
        else:
            print(f"generate_collection_text: Failed to generate collection text from {len(chunk_texts)} chunks")
            return None
        
        return collection
    else:
        print("generate_collection_text: Failed to generate text for all chunks")
        return None
        
        
