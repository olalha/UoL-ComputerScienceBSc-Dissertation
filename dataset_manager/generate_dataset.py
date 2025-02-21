
from typing import Optional

from utils.api_request import prompt_openai_llm_parallel
from prompt_manager.prompt_builder import render_prompt
from chunk_manager.chunk_partitioner import get_chunks
from chunk_manager.chunk_aggregator import aggregate_chunks

def generate_dataset(rulebook: dict, solution_search_time_s: int, model: str) -> Optional[str]:
    
    try:
        # Validate rulebook
        if not rulebook or not isinstance(rulebook, dict):
            raise ValueError(f"generate_dataset: Invalid rulebook")
        
        print("generate_dataset: Partitioning chunks...")
        
        # Generate chunks
        all_chunks_dicts = []
        TOTAL_WC = rulebook['total_wc']
        for topic_name, topic_dict in rulebook['content_rules'].items():
            # Get topic word count
            topic_wc = int(TOTAL_WC * topic_dict['total_proportion'])
            
            # Get sentiment word count
            for index, sentiment in enumerate(["positive", "neutral", "negative"]):
                topic_sentiment_wc = int(topic_wc * topic_dict['sentiment_proportion'][index])
                
                # Skip if no word count
                if topic_sentiment_wc == 0:
                    continue
                
                # Partition topic-sentiment word count into chunks
                chunks = get_chunks(
                    N=topic_sentiment_wc, 
                    min_wc=topic_dict['chunk_min_wc'], 
                    max_wc=topic_dict['chunk_max_wc'], 
                    chunk_count_pref=topic_dict['chunk_count_pref'], 
                    dirichlet_a=topic_dict['chunk_wc_distribution'])

                # Check if partitioning failed
                if not chunks:
                    raise ValueError(f"generate_dataset: Partitioning fail - '{topic_name}' - '{sentiment}' - wc:{topic_sentiment_wc}.")
                
                # Add chunks to all_chunks if partitioning succeeded
                all_chunks_dicts.extend([{'topic': topic_name, 'sentiment': sentiment, 'wc': i} for i in chunks])
        
        print("generate_dataset: Aggregating chunks...")
        
        # Find best allocation of chunks to collections
        WC_RANGES = rulebook['collection_wc_ranges']
        solution = aggregate_chunks(all_chunks_dicts, WC_RANGES, time_limit=solution_search_time_s)
        
        # Check if a solution was found
        if not solution:
            raise ValueError("generate_dataset: Failed to allocate chunks to collections.")
        
        all_collections = [[{'chunk_dict': chunk_dict} for chunk_dict in i['chunks']] for i in solution]
        
        print("generate_dataset: Generating dataset...")
        
        all_chunk_messages = []
        all_chunk_messages_idx = 0
        chunk_collection_map = {}
        
        # Render individual chunk prompts
        REVIEW_ITEM = rulebook['review_item']
        for collection in all_collections:
            for chunk in collection:
                chunk_dict = chunk['chunk_dict']
                
                # Generate prompt
                prompt_context = {
                    'review_item': REVIEW_ITEM,
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
        for idx, collection in enumerate(all_collections):
            # Remove collection if empty
            if not collection:
                print(f"generate_dataset: Removing empty collection.")
                all_collections.remove(collection)
                continue
            # Print collection
            print(f"\n\nCollection: {idx}\n{10*'-'}\n")
            for chunk in collection:
                print(f"Chunk_dict:\n{chunk['chunk_dict']}\n")
                print(f"Generated_text:\n{chunk.get('generated_text', 'NOT GENERATED')}\n")
                
        return None

    # Return none if an exception occurred during generation
    except Exception as e:
        print(f"generate_dataset: {e}")
        return None
