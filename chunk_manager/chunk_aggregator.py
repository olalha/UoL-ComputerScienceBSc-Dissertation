"""
Original Chunk Aggregator Module

This module provides the original implementation of the chunk aggregation problem,
which aims to:
1. Group chunks into collections with unique topics (hard constraint)
2. Match a target distribution of collection sizes (soft constraint)

This version uses a simulated annealing approach with a simple cost function
focused solely on distribution matching.
"""

import random
import math
import time
import copy
from typing import Optional, List, Dict, Any, Callable, Tuple

""" Tuneable Parameters """

# Simulated annealing parameters
INITIAL_TEMPERATURE = 1.0
COOLING_RATE = 0.995
MIN_TEMPERATURE = 0.1

""" Input Validation Functions """

def validate_chunks(chunks):
    """
    Validate the input chunks list.
    Each chunk must be a dictionary with keys 'topic', 'sentiment', and 'wc'.
    """
    if not isinstance(chunks, list):
        print("validate_chunks: Chunks must be a list.")
        return False
    for chunk in chunks:
        if not isinstance(chunk, dict):
            print("validate_chunks: Each chunk must be a dict.")
            return False
        for key in ['topic', 'sentiment', 'wc']:
            if key not in chunk:
                print(f"validate_chunks: Each chunk must have key '{key}'.")
                return False
        if not isinstance(chunk['topic'], str):
            print("validate_chunks: Chunk 'topic' must be a string.")
            return False
        if not isinstance(chunk['sentiment'], str):
            print("validate_chunks: Chunk 'sentiment' must be a string.")
            return False
        if not isinstance(chunk['wc'], int):
            print("validate_chunks: Chunk 'wc' must be an integer.")
            return False
    return True

def validate_collections(collections):
    """
    Validate the input collections list.
    Each collection must be a dictionary with keys 'range' and 'target_fraction'.
    The 'range' key must contain a tuple of two integers.
    Ranges must not overlap or have gaps between them.
    For collection size collections, ranges like (1, 1) are allowed.
    """
    if not isinstance(collections, list):
        print("validate_collections: Collections must be a list.")
        return False
    total_fraction = 0.0
    ranges = []
    for collection in collections:
        if not isinstance(collection, dict):
            print("validate_collections: Each collection must be a dict.")
            return False
        for key in ['range', 'target_fraction']:
            if key not in collection:
                print(f"validate_collections: Each collection must have '{key}'.")
                return False
        if not (isinstance(collection['range'], (tuple, list)) and len(collection['range']) == 2):
            print("validate_collections: Collection 'range' must be a tuple of two ints.")
            return False
        low, high = collection['range']
        if not (isinstance(low, int) and isinstance(high, int)):
            print("validate_collections: Collection range values must be integers.")
            return False
        if low > high:
            print("validate_collections: Collection range low must be less than or equal to high.")
            return False
        if not isinstance(collection['target_fraction'], (int, float)):
            print("validate_collections: Collection 'target_fraction' must be numeric.")
            return False
        ranges.append((low, high))
        total_fraction += collection['target_fraction']
    
    # Check for overlapping ranges
    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            print("validate_collections: Collection ranges must not overlap.")
            return False
    
    # Check for gaps in ranges
    for i in range(len(ranges) - 1):
        if ranges[i][1] + 1 < ranges[i + 1][0]:
            print("validate_collections: Collection ranges must not have gaps.")
            return False
    
    # Check that fractions sum to approximately 1.0
    if abs(total_fraction - 1.0) > 1e-6:
        print("validate_collections: Sum of target fractions must equal 1.")
        return False
        
    return True

""" Collection Manipulation Functions """

def compute_total_wc(collection):
    """
    Return the total word count of a collection.
    
    Args:
        collection: List of chunk dictionaries
        
    Returns:
        int: Total word count
    """
    return sum(chunk['wc'] for chunk in collection)

def get_collection_index(collection, collections, value_extractor):
    """
    Return the index of the collection into which this collection falls,
    based on the numerical value returned by value_extractor.
    (For example, total word count or number of chunks.)
    
    Args:
        collection: The collection of chunks to categorize
        collections: List of collection specifications
        value_extractor: Function that computes a numerical value
        
    Returns:
        Optional[int]: Index of the matching collection, or None if no match
    """
    value = value_extractor(collection)
    for i, collection_obj in enumerate(collections):
        low, high = collection_obj['range']
        if low <= value <= high:
            return i
    return None

def valid_collection(collection):
    """
    Validate that a collection has unique topics.
    
    Args:
        collection: List of chunk dictionaries
        
    Returns:
        bool: True if all topics are unique, False otherwise
    """
    topics = [chunk['topic'] for chunk in collection]
    return len(topics) == len(set(topics))

def initial_solution(chunks, collections, value_extractor):
    """
    Simple initial state: Put each chunk in its own collection.
    (This always respects the hard constraint.)
    
    Args:
        chunks: List of chunk dictionaries
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        List[Dict[str, Any]]: Initial state with each chunk in its own collection
    """
    # Start with an empty list of collections
    state = []
    
    # Add each chunk to its own collection
    for chunk in chunks:
        collection_chunks = [chunk]
        # Determine which collection category this belongs to
        collection_idx = get_collection_index(collection_chunks, collections, value_extractor)
        # Add to state
        state.append({'chunks': collection_chunks, 'collection': collection_idx})
    
    return state

def update_collection_for_collection(collection_obj, collections, value_extractor):
    """
    Update the collection assignment for a single collection and return a tuple
    (old_collection, new_collection) so that an incremental cost update can be performed.
    
    Args:
        collection_obj: The collection object to update
        collections: List of collection specifications
        value_extractor: Function to extract value from collection
        
    Returns:
        Tuple[Optional[int], Optional[int]]: Old and new collection indices
    """
    new_collection = get_collection_index(collection_obj['chunks'], collections, value_extractor)
    old_collection = collection_obj['collection']
    collection_obj['collection'] = new_collection
    return old_collection, new_collection

""" Heuristic / Penalty Functions """

def compute_cost(state, collections, value_extractor):
    """
    Original simple cost function: Compute the overall penalty as the sum over collections 
    of the squared difference between the actual fraction of collections in that collection
    and the target fraction.
    
    Args:
        state: Current state (list of collections)
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        float: Distribution penalty score (lower is better)
    """
    # Count how many collections we have
    N = len(state)
    
    # Count how many collections belong to each collection category
    counts = [0] * len(collections)
    for coll in state:
        idx = coll['collection']
        if idx is not None:
            counts[idx] += 1
    
    # Calculate penalty using squared difference from targets
    penalty = 0.0
    for i, collection_obj in enumerate(collections):
        # Calculate actual fraction of collections in this category
        actual_fraction = counts[i] / N if N > 0 else 0
        # Get target fraction
        target_fraction = collection_obj['target_fraction']
        # Add squared difference to penalty
        penalty += (actual_fraction - target_fraction) ** 2
    
    return penalty

""" Functions For Simulated Annealing """

def propose_neighbor(state, collections, value_extractor):
    """
    Propose a neighboring state by randomly selecting one of the following moves:
      - move: Remove a chunk from one collection and add it to another (or a new collection).
      - swap: Swap a chunk between two collections.
      - merge: Merge two collections (if valid).
      - split: Split one collection into two.
    
    The move is only accepted if it maintains the hard constraint.
    Returns a new state (deep copy) if a move was made, or None if no move was possible.
    
    Args:
        state: Current state
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        Optional[List[Dict[str, Any]]]: New state or None if no move was possible
    """
    # Create a deep copy of the current state
    new_state = copy.deepcopy(state)
    move_made = False

    # Select move type with equal probability (original approach)
    move_type = random.choice(["move", "swap", "merge", "split"])

    if move_type == "move":
        # Check if state is empty
        if len(new_state) == 0:
            return None
            
        # Select a random source collection
        source_idx = random.randint(0, len(new_state) - 1)
        if len(new_state[source_idx]['chunks']) == 0:
            return None
            
        # Select a random chunk from the source collection
        source_coll = new_state[source_idx]['chunks']
        chunk_idx = random.randint(0, len(source_coll) - 1)
        chunk = source_coll.pop(chunk_idx)
        
        # Find valid target collections (those without this topic)
        target_candidates = [
            i for i in range(len(new_state)) 
            if i != source_idx and not any(c['topic'] == chunk['topic'] for c in new_state[i]['chunks'])
        ]
        
        if target_candidates:
            # Move to an existing collection
            target_idx = random.choice(target_candidates)
            target_coll = new_state[target_idx]['chunks']
            target_coll.append(chunk)
            # Update collection category for target
            update_collection_for_collection(new_state[target_idx], collections, value_extractor)
            move_made = True
        else:
            # Only create a new collection if there are no valid existing collections
            new_state.append({
                'chunks': [chunk], 
                'collection': get_collection_index([chunk], collections, value_extractor)
            })
            move_made = True
            
        # Update source collection category
        update_collection_for_collection(new_state[source_idx], collections, value_extractor)
        
        # Remove empty collections
        if len(new_state[source_idx]['chunks']) == 0:
            del new_state[source_idx]

    elif move_type == "swap":
        # Need at least 2 collections to swap
        if len(new_state) < 2:
            return None
            
        # Pick two random collections
        idx1, idx2 = random.sample(range(len(new_state)), 2)
        coll1 = new_state[idx1]['chunks']
        coll2 = new_state[idx2]['chunks']
        
        # Make sure both collections have chunks
        if len(coll1) == 0 or len(coll2) == 0:
            return None
            
        # Pick random positions in each collection
        pos1 = random.randint(0, len(coll1) - 1)
        pos2 = random.randint(0, len(coll2) - 1)
        
        # Check if swap would create valid collections
        temp1 = coll1.copy()
        temp2 = coll2.copy()
        temp1[pos1], temp2[pos2] = coll2[pos2], coll1[pos1]
        
        if valid_collection(temp1) and valid_collection(temp2):
            # Perform the swap
            coll1[pos1], coll2[pos2] = coll2[pos2], coll1[pos1]
            # Update collection categories
            update_collection_for_collection(new_state[idx1], collections, value_extractor)
            update_collection_for_collection(new_state[idx2], collections, value_extractor)
            move_made = True
        else:
            # Swap would violate constraint
            return None

    elif move_type == "merge":
        # Need at least 2 collections to merge
        if len(new_state) < 2:
            return None
            
        # Pick two random collections
        idx1, idx2 = random.sample(range(len(new_state)), 2)
        coll1 = new_state[idx1]['chunks']
        coll2 = new_state[idx2]['chunks']
        
        # Check if merged collection would be valid
        merged = coll1 + coll2
        if valid_collection(merged):
            # Perform the merge
            new_state[idx1]['chunks'] = merged
            # Update collection category
            update_collection_for_collection(new_state[idx1], collections, value_extractor)
            # Remove the second collection
            del new_state[idx2]
            move_made = True
        else:
            # Merge would violate constraint
            return None

    elif move_type == "split":
        # Find collections with at least 2 chunks (needed for splitting)
        candidate_indices = [i for i, coll in enumerate(new_state) if len(coll['chunks']) >= 2]
        if not candidate_indices:
            return None
            
        # Pick a random collection to split
        idx = random.choice(candidate_indices)
        coll = new_state[idx]['chunks']
        
        # Choose random split point
        split_point = random.randint(1, len(coll) - 1)
        new_coll1 = coll[:split_point]
        new_coll2 = coll[split_point:]
        
        # Check if both parts would be valid collections
        if valid_collection(new_coll1) and valid_collection(new_coll2):
            # Update first collection in place
            new_state[idx]['chunks'] = new_coll1
            new_state[idx]['collection'] = get_collection_index(new_coll1, collections, value_extractor)
            # Add second collection
            new_state.append({
                'chunks': new_coll2, 
                'collection': get_collection_index(new_coll2, collections, value_extractor)
            })
            move_made = True
        else:
            # Split would violate constraint
            return None

    # Return the new state if a move was made, otherwise None
    return new_state if move_made else None

def simulated_annealing(initial_state, collections, value_extractor, time_limit=10, max_iter=None):
    """
    Perform simulated annealing to find an optimal allocation of chunks to collections.
    The algorithm runs for a specified time limit or maximum number of iterations.
    Returns the best state found during the search.
    
    Args:
        initial_state: Starting state
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        time_limit: Maximum run time in seconds
        max_iter: Maximum number of iterations (optional)
        
    Returns:
        List[Dict[str, Any]]: Best state found
    """
    start_time = time.time()
    current_state = initial_state
    best_state = copy.deepcopy(initial_state)
    current_cost = compute_cost(current_state, collections, value_extractor)
    best_cost = current_cost
    
    # Initialize temperature
    T = INITIAL_TEMPERATURE
    iteration = 0
    stagnant_iterations = 0
    max_stagnant = 1000  # Reset if stuck
    
    if max_iter is None:
        max_iter = float('inf')
        
    # Track progress
    costs = []
    collection_counts = []
    
    while (time.time() - start_time < time_limit) and (iteration < max_iter):
        iteration += 1
        
        # Generate neighbor state
        neighbor = propose_neighbor(current_state, collections, value_extractor)
        if neighbor is None:
            continue
            
        # Calculate costs
        new_cost = compute_cost(neighbor, collections, value_extractor)
        delta = new_cost - current_cost
        
        # Accept or reject move
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_cost = new_cost
            stagnant_iterations = 0
            
            # Update best if improved
            if new_cost < best_cost:
                best_state = copy.deepcopy(neighbor)
                best_cost = new_cost
        else:
            stagnant_iterations += 1
        
        # Cool the temperature
        T *= COOLING_RATE
        
        # Ensure T doesn't get too low
        if T < MIN_TEMPERATURE:
            T = MIN_TEMPERATURE
        
        # If stuck, give small boost to temperature
        if stagnant_iterations > max_stagnant:
            T = INITIAL_TEMPERATURE * 0.5
            stagnant_iterations = 0
        
        # Track metrics every 10 iterations
        if iteration % 10 == 0:
            costs.append(current_cost)
            collection_counts.append(len(current_state))
    
    # Return the best state found
    return best_state

""" Main Function """

def aggregate_chunks(chunks: list[dict], 
                     collections: list[dict], 
                     collection_mode: str,
                     time_limit: float = 10, 
                     max_iter: int = None) -> Optional[list[dict]]:
    """
    Aggregate the chunks into collections such that:
      - No collection contains duplicate topics (hard constraint)
      - The overall distribution (by either total word count or number of chunks per collection)
        approximates the target fractions provided in the collections (soft constraint, via a penalty function).
    
    Args:
        chunks: List of chunk dictionaries
        collections: Collection specifications
        collection_mode: "word" or "chunk" for size calculation
        time_limit: Maximum run time in seconds
        max_iter: Maximum iterations (for simulated annealing)
        
    Returns:
        Optional[List[Dict[str, Any]]]: The best state found, or None if invalid
    """
    # Validate inputs
    if not validate_chunks(chunks):
        return None
    if not validate_collections(collections):
        return None

    # Define value_extractor based on collection_mode
    if collection_mode == "word":
        # Use word count as the value
        value_extractor = compute_total_wc
    elif collection_mode == "chunk":
        # Use number of chunks as the value
        value_extractor = lambda collection: len(collection)
    else:
        print("aggregate_chunks: Invalid collection_mode (must be 'word' or 'chunk').")
        return None

    # Create initial solution
    state = initial_solution(chunks, collections, value_extractor)
    
    # Validate initial solution
    for coll in state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: Initial solution violates hard constraint")
            return None

    # Run simulated annealing to find solution
    best_state = simulated_annealing(state, collections, value_extractor, time_limit, max_iter)

    # Validate final solution
    for coll in best_state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: No solution has been found")
            return None

    return best_state
