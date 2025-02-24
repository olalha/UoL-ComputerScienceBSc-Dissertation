from typing import Optional

from utils.api_request import prompt_openai_llm_parallel
from prompt_manager.prompt_builder import render_prompt
from chunk_manager.chunk_partitioner import get_chunks
from chunk_manager.chunk_aggregator import aggregate_chunks

def generate_dataset_outline(rulebook: dict, solution_search_time_s: int) -> Optional[str]:
    
    # Validate rulebook
    if not rulebook or not isinstance(rulebook, dict):
        print(f"generate_dataset_outline: Invalid rulebook")
        return None
    
    # Generate chunks
    all_chunks_dicts = get_chunks(rulebook=rulebook)
    
    # Check if partitioning failed
    if not all_chunks_dicts:
        print("generate_dataset_outline: Failed to partition chunks.")
        return None
    
    # Find best allocation of chunks to collections
    collection_ranges = rulebook['collection_ranges']
    solution = aggregate_chunks(chunks=all_chunks_dicts, 
                                collections=collection_ranges, 
                                time_limit=solution_search_time_s)
    
    # Check if a solution was found
    if not solution:
        print("generate_dataset_outline: Failed to allocate chunks to collections.")
        return None
    
    # Prepare dataset structure
    collections = []
    for idx, item in enumerate(solution):
        collection = {'idx': idx, 'collection_wc': 0, 'collection_cc': 0, 'chunks': []}
        wc, cc = 0, 0
        for chunk_dict in item['chunks']:
            collection['chunks'].append({'chunk_dict': chunk_dict})
            wc += chunk_dict['wc']
            cc += 1
        collection['collection_wc'] = wc
        collection['collection_cc'] = cc
        collections.append(collection)
        
    # Update dataset outline metadata and return
    return update_dataset_structure_metadata({'collections': collections})

def update_dataset_structure_metadata(dataset_outline: dict) -> Optional[dict]:
    
    # Validate dataset outline
    if not dataset_outline or not isinstance(dataset_outline, dict):
        print(f"update_dataset_outline_metadata: Invalid dataset_outline")
        return None
    # Validate collections
    if 'collections' not in dataset_outline or not dataset_outline['collections']:
        print(f"update_dataset_outline_metadata: Invalid collections")
        return None
    
    # Collect metadata
    updated_dataset_outline = {}
    
    total_wc = 0
    total_cc = 0
    collection_cc_distribution = {}
    sentiment_wc_distribution = {}
    sentiment_cc_distribution = {}
    topic_wc_distribution = {}
    topic_cc_distribution = {}
    ts_wc_distribution = {}
    ts_cc_distribution = {}
    
    collections = dataset_outline['collections']
    for collection in collections:
        collection_wc = 0
        collection_cc = 0
        
        for chunk in collection['chunks']:
            
            # Update word/chunk count metadata
            chunk_dict = chunk['chunk_dict']
            collection_wc += chunk_dict['wc']
            collection_cc += 1
            total_wc += chunk_dict['wc']
            total_cc += 1
            
            # Update sentiment distribution
            sentiment = chunk_dict['sentiment']
            sentiment_wc_distribution[sentiment] = sentiment_wc_distribution.get(sentiment, 0) + chunk_dict['wc']
            sentiment_cc_distribution[sentiment] = sentiment_cc_distribution.get(sentiment, 0) + 1
            
            # Update topic distribution
            topic = chunk_dict['topic']
            topic_wc_distribution[topic] = topic_wc_distribution.get(topic, 0) + chunk_dict['wc']
            topic_cc_distribution[topic] = topic_cc_distribution.get(topic, 0) + 1
            
            # Update topic-sentiment distribution
            ts = f"{topic} - {sentiment}"
            ts_wc_distribution[ts] = ts_wc_distribution.get(ts, 0) + chunk_dict['wc']
            ts_cc_distribution[ts] = ts_cc_distribution.get(ts, 0) + 1
        
        # Update collection metadata    
        collection['collection_wc'] = collection_wc
        collection['collection_cc'] = collection_cc
        
        # Update collection distribution
        collection_cc_distribution[collection_cc] = collection_cc_distribution.get(collection_cc, 0) + 1
        
    # Update dataset metadata
    updated_dataset_outline['total_wc'] = total_wc
    updated_dataset_outline['total_cc'] = total_cc
    updated_dataset_outline['collection_cc_distribution'] = collection_cc_distribution
    updated_dataset_outline['sentiment_wc_distribution'] = sentiment_wc_distribution
    updated_dataset_outline['sentiment_cc_distribution'] = sentiment_cc_distribution
    updated_dataset_outline['topic_wc_distribution'] = topic_wc_distribution
    updated_dataset_outline['topic_cc_distribution'] = topic_cc_distribution
    updated_dataset_outline['ts_wc_distribution'] = ts_wc_distribution
    updated_dataset_outline['ts_cc_distribution'] = ts_cc_distribution
    updated_dataset_outline['collections'] = collections
    
    return updated_dataset_outline

""" 
NOTE:
This function is not complete. 
It is a placeholder for the actual function that will generate the dataset text.
"""
def generate_dataset_text(collections: list[list[dict]], review_item: str, model: str) -> Optional[str]:
    
    all_chunk_messages = []
    all_chunk_messages_idx = 0
    chunk_collection_map = {}
    
    for collection in collections:
        for chunk in collection:
            chunk_dict = chunk['chunk_dict']
            
            # Generate prompt
            prompt_context = {
                'review_item': review_item,
                'topic': chunk_dict['topic'],
                'sentiment': chunk_dict['sentiment'],
                'word_count': chunk_dict['wc']
            }
            prompt = render_prompt("usr_chunk_gen.html", prompt_context)
            messages = [{'role': 'user', 'content': prompt}]
            
            # Add the collection and chunk dict to the collection map
            all_chunk_messages.append(messages)
            chunk_collection_map[all_chunk_messages_idx] = {'collection': collection, 'chunk_dict': chunk_dict}
            all_chunk_messages_idx += 1
        
    # Generate text for each chunk
    responses = prompt_openai_llm_parallel(model=model, messages=all_chunk_messages)
    
    # Update all collections with generated text
    for r in responses:
        collection = chunk_collection_map[r['idx']]['collection']
        chunk_dict = chunk_collection_map[r['idx']]['chunk_dict']
        
        # Extract generated text
        generated_text = None
        try:
            generated_text = r['response']['choices'][0]['message']['content']
        except Exception as e:
            pass
        
        # Update collection with generated text
        for chunk in collection:
            if chunk['chunk_dict'] == chunk_dict:
                if generated_text:
                    chunk['generated_text'] = generated_text
                else:
                    print(f"generate_dataset: Failed to generate chunk: {chunk} - Removing from collection.")
                    collection.remove(chunk)
        
    # Print all collections
    for idx, collection in enumerate(collections):
        # Remove collection if empty
        if not collection:
            print(f"generate_dataset: Removing empty collection.")
            collections.remove(collection)
            continue
        # Print collection
        print(f"\n\nCollection: {idx}\n{10*'-'}\n")
        for chunk in collection:
            print(f"Chunk_dict:\n{chunk['chunk_dict']}\n")
            print(f"Generated_text:\n{chunk.get('generated_text', 'NOT GENERATED')}\n")
                
    return None
