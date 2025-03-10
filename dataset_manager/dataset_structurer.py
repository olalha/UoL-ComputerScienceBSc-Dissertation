from typing import Optional

from chunk_manager.chunk_partitioner import get_chunks
from chunk_manager.chunk_aggregator import aggregate_chunks
from chunk_manager.rulebook_parser import validate_rulebook_values

def create_dataset_structure(rulebook: dict, solution_search_time_s: int) -> Optional[str]:
    """
    Creates a structured dataset from a rulebook and writes it to a JSON file.

    The function validates the rulebook, generates chunks, allocates them to collections
    according to the rulebook specifications, and builds a complete dataset structure with
    metadata. The resulting dataset is then written to a JSON file.

    Args:
        rulebook (dict): The rulebook containing dataset parameters and constraints.
        solution_search_time_s (int): Maximum time in seconds to search for a solution
                                     when allocating chunks to collections.

    Returns:
        Optional[str]: Path to the created JSON file on success, or None if an error occurred.
    """
    
    # Validate rulebook values
    if not validate_rulebook_values(rulebook):
        print(f"create_dataset_structure: Invalid rulebook")
        return None
    
    # Generate chunks
    all_chunks_dicts = get_chunks(rulebook=rulebook)
    
    # Check if partitioning failed
    if not all_chunks_dicts:
        print("create_dataset_structure: Failed to partition chunks.")
        return None
    
    # Find best allocation of chunks to collections
    collection_ranges = rulebook['collection_ranges']
    collection_mode = rulebook['collection_mode']
    solution = aggregate_chunks(chunks=all_chunks_dicts, 
                                collections=collection_ranges,
                                collection_mode=collection_mode, 
                                time_limit=solution_search_time_s)
    
    # Check if a solution was found
    if not solution:
        print("create_dataset_structure: Failed to allocate chunks to collections.")
        return None
    
    # Prepare dataset structure
    collections = []
    for item in solution:
        collection = {'chunks': [], 'collection_text': None}
        for chunk_dict in item['chunks']:
            collection['chunks'].append({'chunk_dict': chunk_dict, 'chunk_text': None})
        collections.append(collection)
        
    # Build dataset structure and return it
    return {'content_title': rulebook['content_title'], 'collections': collections}

def validate_dataset_values(dataset_structure: dict) -> bool:
    """
    Validates that a dataset structure meets all required specifications.

    The function performs extensive validation on the dataset structure, checking for
    correct types, required fields, and valid values at all levels of the hierarchy
    (dataset, collections, chunks, and chunk dictionaries).

    Args:
        dataset_structure (dict): The dataset structure to validate.

    Returns:
        bool: True if the dataset structure is valid, False otherwise with error messages
              printed to the console.
    """
    
    # Check if dataset_structure is a dict
    if dict is None or not isinstance(dataset_structure, dict):
        print("validate_dataset_structure: dataset_structure must be a dictionary and cannot be None")
        return False
    
    # Check content_title exists and is a string
    if 'content_title' not in dataset_structure:
        print("validate_dataset_structure: Missing 'content_title' key")
        return False
    if not isinstance(dataset_structure['content_title'], str):
        print("validate_dataset_structure: 'content_title' must be a string")
        return False
    
    # Check collections exists and is a list
    if 'collections' not in dataset_structure:
        print("validate_dataset_structure: Missing 'collections' key")
        return False
    if not isinstance(dataset_structure['collections'], list):
        print("validate_dataset_structure: 'collections' must be a list")
        return False
    
    # Validate each collection
    for i, collection in enumerate(dataset_structure['collections']):
        if not isinstance(collection, dict):
            print(f"validate_dataset_structure: Collection at index {i} must be a dictionary")
            return False
        
        # Check collection_text is None or string
        if 'collection_text' not in collection:
            print(f"validate_dataset_structure: Collection at index {i} missing 'collection_text' key")
            return False
        if collection['collection_text'] is not None and not isinstance(collection['collection_text'], str):
            print(f"validate_dataset_structure: 'collection_text' in collection {i} must be None or a string")
            return False
        
        # Check chunks exists and is a list
        if 'chunks' not in collection:
            print(f"validate_dataset_structure: Collection at index {i} missing 'chunks' key")
            return False
        if not isinstance(collection['chunks'], list):
            print(f"validate_dataset_structure: 'chunks' in collection {i} must be a list")
            return False
        
        # Validate each chunk
        for j, chunk in enumerate(collection['chunks']):
            if not isinstance(chunk, dict):
                print(f"validate_dataset_structure: Chunk at index {j} in collection {i} must be a dictionary")
                return False
            
            # Check chunk_text is None or string
            if 'chunk_text' not in chunk:
                print(f"validate_dataset_structure: Chunk at index {j} in collection {i} missing 'chunk_text' key")
                return False
            if chunk['chunk_text'] is not None and not isinstance(chunk['chunk_text'], str):
                print(f"validate_dataset_structure: 'chunk_text' in chunk {j} of collection {i} must be None or a string")
                return False
            
            # Check collection_text and chunk_text are consistent
            if collection['collection_text'] is not None and chunk['chunk_text'] is None:
                print(f"validate_dataset_structure: Collection at index {i} has collection_text but chunk at index {j} has no chunk_text")
                return False
            
            # Check chunk_dict exists and is a dict
            if 'chunk_dict' not in chunk:
                print(f"validate_dataset_structure: Chunk at index {j} in collection {i} missing 'chunk_dict' key")
                return False
            if not isinstance(chunk['chunk_dict'], dict):
                print(f"validate_dataset_structure: 'chunk_dict' in chunk {j} of collection {i} must be a dictionary")
                return False
            
            chunk_dict = chunk['chunk_dict']
            
            # Check topic is a non-empty string
            if 'topic' not in chunk_dict:
                print(f"validate_dataset_structure: Missing 'topic' in chunk_dict at collection {i}, chunk {j}")
                return False
            if not isinstance(chunk_dict['topic'], str) or not chunk_dict['topic']:
                print(f"validate_dataset_structure: 'topic' must be a non-empty string in collection {i}, chunk {j}")
                return False
            
            # Check sentiment is one of the valid values
            if 'sentiment' not in chunk_dict:
                print(f"validate_dataset_structure: Missing 'sentiment' in chunk_dict at collection {i}, chunk {j}")
                return False
            valid_sentiments = ['positive', 'neutral', 'negative']
            if chunk_dict['sentiment'] not in valid_sentiments:
                print(f"validate_dataset_structure: 'sentiment' must be one of {valid_sentiments} in collection {i}, chunk {j}")
                return False
            
            # Check wc is an int greater than 0
            if 'wc' not in chunk_dict:
                print(f"validate_dataset_structure: Missing 'wc' in chunk_dict at collection {i}, chunk {j}")
                return False
            if not isinstance(chunk_dict['wc'], int) or chunk_dict['wc'] <= 0:
                print(f"validate_dataset_structure: 'wc' must be an integer greater than 0 in collection {i}, chunk {j}")
                return False
    
    # All validations passed
    return True
