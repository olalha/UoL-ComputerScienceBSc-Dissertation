from typing import Optional
from pathlib import Path
from typing import Optional
import json

from utils.settings_manager import get_setting
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
        
    # Update dataset outline metadata and return
    base_dataset_structure = {'review_item': rulebook['review_item'], 'collections': collections}
    complete_dataset_structure = validate_and_update_dataset_meta(dataset_structure=base_dataset_structure)
    
    if not complete_dataset_structure:
        print("create_dataset_structure: Failed to validate and update dataset metadata.")
        return None
    
    return write_dataset_json(complete_dataset_structure)

def validate_and_update_dataset_meta(dataset_structure: dict) -> Optional[dict]:
    """
    Validates the dataset structure and updates it with comprehensive metadata.

    The function first validates that the dataset structure is properly formed, then
    calculates and adds metadata such as word counts, chunk counts, and distributions
    across collections, topics, and sentiments.

    Args:
        dataset_structure (dict): The base dataset structure to validate and enhance.

    Returns:
        Optional[dict]: The updated dataset structure with complete metadata on success,
                       or None if validation fails.
    """
    
    # Validate dataset structure
    if not validate_dataset_structure(dataset_structure):
        print(f"validate_and_update_dataset_meta: Invalid dataset_structure")
        return None
    
    # Collect metadata
    updated_dataset_structure = {}
    
    total_wc = 0
    total_cc = 0
    chunks_with_text = 0
    collections_with_text = 0
    collection_cc_distribution = {}
    sentiment_wc_distribution = {}
    sentiment_cc_distribution = {}
    topic_wc_distribution = {}
    topic_cc_distribution = {}
    ts_wc_distribution = {}
    ts_cc_distribution = {}
    
    collections = dataset_structure['collections']
    for collection in collections:
        collection_wc = 0
        collection_cc = 0
        chunk_text_count = 0
        
        for chunk in collection['chunks']:
            
            # Update word/chunk count metadata
            chunk_dict = chunk['chunk_dict']
            collection_wc += chunk_dict['wc']
            collection_cc += 1
            total_wc += chunk_dict['wc']
            total_cc += 1
            
            # Update text count metadata
            chunk_text_count += 1 if chunk['chunk_text'] else 0
            
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
        
        # Update collection text count
        chunks_with_text += chunk_text_count
        if collection['collection_text']:
            collections_with_text += 1
        
        # Update collection distribution
        collection_cc_distribution[collection_cc] = collection_cc_distribution.get(collection_cc, 0) + 1
        
    # Update dataset metadata
    updated_dataset_structure['review_item'] = dataset_structure['review_item']
    updated_dataset_structure['total_wc'] = total_wc
    updated_dataset_structure['total_cc'] = total_cc
    updated_dataset_structure['collections_count'] = len(collections)
    updated_dataset_structure['chunks_with_text'] = chunks_with_text
    updated_dataset_structure['collections_with_text'] = collections_with_text
    updated_dataset_structure['collection_cc_distribution'] = collection_cc_distribution
    updated_dataset_structure['sentiment_wc_distribution'] = sentiment_wc_distribution
    updated_dataset_structure['sentiment_cc_distribution'] = sentiment_cc_distribution
    updated_dataset_structure['topic_wc_distribution'] = topic_wc_distribution
    updated_dataset_structure['topic_cc_distribution'] = topic_cc_distribution
    updated_dataset_structure['ts_wc_distribution'] = ts_wc_distribution
    updated_dataset_structure['ts_cc_distribution'] = ts_cc_distribution
    updated_dataset_structure['collections'] = collections
    
    return updated_dataset_structure

def validate_dataset_structure(dataset_structure: dict) -> bool:
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
    
    # Check review_item exists and is a string
    if 'review_item' not in dataset_structure:
        print("validate_dataset_structure: Missing 'review_item' key")
        return False
    if not isinstance(dataset_structure['review_item'], str):
        print("validate_dataset_structure: 'review_item' must be a string")
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

def write_dataset_json(dataset: dict) -> Optional[Path]:
    """
    Writes the dataset structure dictionary to a JSON file in the designated directory.

    The function creates a unique filename based on the dataset's 'review_item', 'total_wc',
    and 'total_cc' values. It writes the JSON file in the  directory, 
    creating the directory if it does not exist.

    Args:
        dataset (dict): The validated dataset structure dictionary to write.

    Returns:
        Optional[Path]: The full path to the newly created JSON file on success, or None if an error occurred.
    """
    try:
        # Create the JSON directory if it does not exist
        json_dir = Path(__file__).parent.parent / get_setting('PATH', 'datasets_json')
        json_dir.mkdir(parents=True, exist_ok=True)
        base_filename = f"{dataset['review_item']} - {dataset['total_wc']}wc - {dataset['total_cc']}cc.json"
        json_path = json_dir / base_filename
        
        # Ensure the filename is unique
        counter = 1
        while json_path.exists():
            json_path = json_dir / f"{dataset['review_item']} - {dataset['total_wc']}wc - {dataset['total_cc']}cc ({counter}).json"
            counter += 1

        # Write the JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        return json_path
    
    except Exception as e:
        print(f"write_dataset_json: Error writing JSON file: {e}")
        return None