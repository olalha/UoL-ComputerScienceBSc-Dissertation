
# NOTE: This is not finished and is not used in the current implementation of the dataset manager.

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
