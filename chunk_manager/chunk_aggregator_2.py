"""
Improved Chunk Aggregator Module

This module provides an improved implementation of the chunk aggregation problem,
which aims to:
1. Group chunks into collections with unique topics (hard constraint)
2. Match a target distribution of collection sizes (soft constraint)
3. Minimize the total number of collections

Three different optimization approaches are provided:
- Improved Simulated Annealing: Enhanced version with better cost function and move operations
- Greedy Algorithm: Fast approximation focusing on collection minimization
- Two-Phase Approach: Combines greedy initialization with simulated annealing refinement
"""

import random
import math
import time
import copy
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

from .chunk_aggregator import validate_chunks, validate_collections

""" Tuneable Parameters """

# Simulated annealing parameters
INITIAL_TEMPERATURE = 1.0
COOLING_RATE = 0.995
MIN_TEMPERATURE = 0.1

# Cost function weights
DISTRIBUTION_WEIGHT = 0.5
COLLECTION_COUNT_WEIGHT = 1.0
OOR_PENALTY_WEIGHT = 1.0

# Greedy algorithm randomization factor
GREEDY_RANDOMIZATION = 0.5

""" Collection Manipulation Functions """

def compute_total_wc(collection: List[Dict[str, Any]]) -> int:
    """
    Return the total word count of a collection.
    
    Args:
        collection: List of chunk dictionaries
        
    Returns:
        int: Total word count
    """
    return sum(chunk['wc'] for chunk in collection)


def get_collection_index(collection: List[Dict[str, Any]], 
                         collections: List[Dict[str, Any]], 
                         value_extractor: Callable) -> Optional[int]:
    """
    Return the index of the collection into which this collection falls,
    based on the numerical value returned by value_extractor.
    (For example, total word count or number of chunks.)
    
    Args:
        collection: The collection of chunks to categorize
        collections: List of collection specifications
        value_extractor: Function that computes a numerical value for the collection
        
    Returns:
        Optional[int]: Index of the matching collection, or None if no match
    """
    value = value_extractor(collection)
    for i, collection_obj in enumerate(collections):
        low, high = collection_obj['range']
        if low <= value <= high:
            return i
    return None


def valid_collection(collection: List[Dict[str, Any]]) -> bool:
    """
    Validate that a collection has unique topics.
    
    Args:
        collection: List of chunk dictionaries
        
    Returns:
        bool: True if all topics are unique, False otherwise
    """
    topics = [chunk['topic'] for chunk in collection]
    return len(topics) == len(set(topics))


def update_collection_for_collection(collection_obj: Dict[str, Any], 
                                    collections: List[Dict[str, Any]], 
                                    value_extractor: Callable) -> Tuple[Optional[int], Optional[int]]:
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


""" Enhanced Cost Function """

def compute_cost(state: List[Dict[str, Any]], 
                collections: List[Dict[str, Any]], 
                value_extractor: Callable) -> float:
    """
    Improved cost function that balances distribution matching with collection minimization.
    
    Args:
        state: Current state (list of collections)
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        float: Computed cost (lower is better)
    """
    # Get current number of collections
    N = len(state)
    if N == 0:
        return float('inf')  # Invalid state
    
    
    # Calculate distribution matching and out of range penalty
    count_oor = 0
    counts = [0] * len(collections)
    for coll in state:
        idx = coll['collection']
        if idx is not None:
            counts[idx] += 1
        else:
            count_oor += 1
    
    distribution_penalty = 0.0
    for i, collection_obj in enumerate(collections):
        actual_fraction = counts[i] / N if N > 0 else 0
        target_fraction = collection_obj['target_fraction']
        distribution_penalty += (actual_fraction - target_fraction) ** 2
    
    # Calculate collection count penalty
    # Count occurrences of each topic
    topic_counts = {}
    for coll in state:
        for chunk in coll['chunks']:
            topic = chunk['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Minimum collections is determined by the most frequent topic
    min_collections = max(topic_counts.values()) if topic_counts else 0
    
    # Combine penalties with weights
    total_penalty = (
        DISTRIBUTION_WEIGHT * distribution_penalty + 
        COLLECTION_COUNT_WEIGHT * min_collections +
        OOR_PENALTY_WEIGHT * count_oor
    )
    
    return total_penalty


""" Greedy Algorithm Implementation """

def greedy_solution(chunks: List[Dict[str, Any]], 
                   collections: List[Dict[str, Any]], 
                   value_extractor: Callable) -> List[Dict[str, Any]]:
    """
    Create a solution using a greedy algorithm that:
    1. Sorts chunks by rarity of topic
    2. For each chunk, finds the best existing collection or creates a new one
    
    Args:
        chunks: List of chunk dictionaries
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        List[Dict[str, Any]]: The constructed state
    """
    # Start with empty state
    state = []
    
    # Count occurrences of each topic
    topic_counts = {}
    for chunk in chunks:
        topic = chunk['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Sort chunks by frequency of topic (most frequent first) to establish minimum collections
    sorted_chunks = sorted(chunks, key=lambda c: (-topic_counts[c['topic']], -c['wc']))
    
    # Process each chunk
    for chunk in sorted_chunks:
        best_collection = None
        best_cost = float('inf')
        best_index = -1
        
        # Try adding to each existing collection
        for i, coll in enumerate(state):
            # Skip if topic already exists in this collection
            if any(c['topic'] == chunk['topic'] for c in coll['chunks']):
                continue
            
            # Try adding chunk to this collection
            new_chunks = coll['chunks'] + [chunk]
            new_collection = {
                'chunks': new_chunks,
                'collection': get_collection_index(new_chunks, collections, value_extractor)
            }
            
            # Create temporary state to evaluate cost
            temp_state = state.copy()
            temp_state[i] = new_collection
            cost = compute_cost(temp_state, collections, value_extractor)
            
            # If this is better than current best, update best
            if cost < best_cost or (cost == best_cost and random.random() < GREEDY_RANDOMIZATION):
                best_cost = cost
                best_collection = new_collection
                best_index = i
        
        # Create a new collection with just this chunk
        new_coll = {
            'chunks': [chunk],
            'collection': get_collection_index([chunk], collections, value_extractor)
        }
        
        # Evaluate creating a new collection
        temp_state = state.copy()
        temp_state.append(new_coll)
        new_cost = compute_cost(temp_state, collections, value_extractor)
        
        # Compare with best existing collection
        if new_cost < best_cost or (new_cost == best_cost and random.random() < GREEDY_RANDOMIZATION) or best_index < 0:
            # Create new collection
            state.append(new_coll)
        else:
            # Add to best existing collection
            state[best_index] = best_collection
    
    return state

""" Improved Simulated Annealing """

def get_adaptive_move_probs(state: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Determine adaptive probabilities for move types based on current state.
    
    Args:
        state: Current state
        chunks: Original chunks list
        
    Returns:
        Dict[str, float]: Probabilities for each move type
    """
    
    # Count occurrences of each topic
    topic_counts = {}
    for coll in state:
        for chunk in coll['chunks']:
            topic = chunk['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Minimum collections is determined by the most frequent topic
    min_collections = max(topic_counts.values()) if topic_counts else 0
    current_collections = len(state)
    
    # If we're close to minimum collections, favor split move and swap operations
    if current_collections <= min_collections * 1.5:
        return {
            "swap": 0.4,
            "move": 0.3, 
            "merge": 0.2, 
            "split": 0.1
        }
    # If we have more than double the collections necessary, favor merge operations
    if current_collections > min_collections * 2:
        return {
            "move": 0.1, 
            "swap": 0.2, 
            "merge": 0.6,
            "split": 0.1
        }
    # Otherwise, use balanced probabilities
    else:
        return {
            "move": 0.25,
            "swap": 0.25,
            "merge": 0.30,
            "split": 0.20
        }

def propose_neighbor(state: List[Dict[str, Any]], 
                    collections: List[Dict[str, Any]], 
                    value_extractor: Callable,
                    chunks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Propose a neighboring state with adaptive move type selection.
    
    Args:
        state: Current state
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        chunks: Original chunks list for adaptive probabilities
        
    Returns:
        Optional[List[Dict[str, Any]]]: New state or None if no move was possible
    """
    # Create a deep copy of the current state
    new_state = copy.deepcopy(state)
    move_made = False
    
    # Get adaptive probabilities based on current state
    move_probs = get_adaptive_move_probs(state, chunks)
    
    # Select move type based on adaptive probabilities
    move_types = list(move_probs.keys())
    probabilities = list(move_probs.values())
    move_type = random.choices(move_types, weights=probabilities, k=1)[0]
    
    if move_type == "move":
        if len(new_state) == 0:
            return None
        source_idx = random.randint(0, len(new_state) - 1)
        if len(new_state[source_idx]['chunks']) == 0:
            return None
        source_coll = new_state[source_idx]['chunks']
        chunk_idx = random.randint(0, len(source_coll) - 1)
        chunk = source_coll.pop(chunk_idx)
        
        # Find candidate collections (those without this topic)
        target_candidates = [
            i for i in range(len(new_state)) 
            if i != source_idx and not any(c['topic'] == chunk['topic'] for c in new_state[i]['chunks'])
        ]
        
        move_made = False
        if target_candidates:
            # Move to an existing collection
            target_idx = random.choice(target_candidates)
            target_coll = new_state[target_idx]['chunks']
            target_coll.append(chunk)
            update_collection_for_collection(new_state[target_idx], collections, value_extractor)
            move_made = True
        else:
            # Only create a new collection if there are no valid existing collections
            new_state.append({
                'chunks': [chunk], 
                'collection': get_collection_index([chunk], collections, value_extractor)
            })
            move_made = True
            
        # Update source collection
        update_collection_for_collection(new_state[source_idx], collections, value_extractor)
        
        # Remove empty collections
        if len(new_state[source_idx]['chunks']) == 0:
            del new_state[source_idx]

    elif move_type == "swap":
        if len(new_state) < 2:
            return None
        
        # Find collections with at least one chunk
        valid_indices = [i for i, coll in enumerate(new_state) if len(coll['chunks']) > 0]
        if len(valid_indices) < 2:
            return None
            
        idx1, idx2 = random.sample(valid_indices, 2)
        coll1 = new_state[idx1]['chunks']
        coll2 = new_state[idx2]['chunks']
        
        pos1 = random.randint(0, len(coll1) - 1)
        pos2 = random.randint(0, len(coll2) - 1)
        
        # Check if swap would create duplicate topics
        temp1 = coll1.copy()
        temp2 = coll2.copy()
        temp1[pos1], temp2[pos2] = coll2[pos2], coll1[pos1]
        
        if valid_collection(temp1) and valid_collection(temp2):
            # Perform swap
            coll1[pos1], coll2[pos2] = coll2[pos2], coll1[pos1]
            update_collection_for_collection(new_state[idx1], collections, value_extractor)
            update_collection_for_collection(new_state[idx2], collections, value_extractor)
            move_made = True
        else:
            return None

    elif move_type == "merge":
        if len(new_state) < 2:
            return None
            
        # Try up to 3 random pairs before giving up
        for _ in range(3):
            idx1, idx2 = random.sample(range(len(new_state)), 2)
            coll1 = new_state[idx1]['chunks']
            coll2 = new_state[idx2]['chunks']
            merged = coll1 + coll2
            
            if valid_collection(merged):
                new_state[idx1]['chunks'] = merged
                update_collection_for_collection(new_state[idx1], collections, value_extractor)
                del new_state[idx2]
                move_made = True
                break
        
        if not move_made:
            return None

    elif move_type == "split":
        # Only try to split collections with at least 2 chunks
        candidate_indices = [i for i, coll in enumerate(new_state) if len(coll['chunks']) >= 2]
        if not candidate_indices:
            return None
            
        idx = random.choice(candidate_indices)
        coll = new_state[idx]['chunks']
        
        # Try random split points
        for _ in range(3):  # Try up to 3 times
            split_point = random.randint(1, len(coll) - 1)
            new_coll1 = coll[:split_point]
            new_coll2 = coll[split_point:]
            
            if valid_collection(new_coll1) and valid_collection(new_coll2):
                new_state[idx]['chunks'] = new_coll1
                update_collection_for_collection(new_state[idx], collections, value_extractor)
                new_state.append({
                    'chunks': new_coll2, 
                    'collection': get_collection_index(new_coll2, collections, value_extractor)
                })
                move_made = True
                break
                
        if not move_made:
            return None

    return new_state if move_made else None


def improved_simulated_annealing(initial_state: List[Dict[str, Any]], 
                                collections: List[Dict[str, Any]], 
                                value_extractor: Callable,
                                chunks: List[Dict[str, Any]],
                                time_limit: float = 10, 
                                max_iter: int = None) -> List[Dict[str, Any]]:
    """
    Perform improved simulated annealing with adaptive move probabilities 
    and enhanced cost function.
    
    Args:
        initial_state: Starting state
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        chunks: Original chunks list (for adaptive move selection)
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
        neighbor = propose_neighbor(current_state, collections, value_extractor, chunks)
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

""" Two-Phase Approach """

def two_phase_optimization(chunks: List[Dict[str, Any]], 
                          collections: List[Dict[str, Any]], 
                          value_extractor: Callable,
                          time_limit: float = 10) -> List[Dict[str, Any]]:
    """
    Two-phase optimization:
    1. Generate a good initial solution using greedy algorithm
    2. Refine it using simulated annealing
    
    Args:
        chunks: List of chunk dictionaries
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        time_limit: Maximum run time in seconds
        
    Returns:
        List[Dict[str, Any]]: Best state found
    """
    # Phase 1: Greedy initialization
    initial_state = greedy_solution(chunks, collections, value_extractor)
 
    # Phase 2: Refine with simulated annealing
    return improved_simulated_annealing(
        initial_state, 
        collections, 
        value_extractor,
        chunks,
        time_limit=time_limit
    )


""" Main Function """

def aggregate_chunks(chunks: List[Dict[str, Any]], 
                     collections: List[Dict[str, Any]], 
                     collection_mode: str,
                     algorithm: str = "two_phase",
                     time_limit: float = 10, 
                     max_iter: int = None) -> Optional[List[Dict[str, Any]]]:
    """
    Aggregate chunks into collections with three algorithm options:
    - "improved_sa": Improved simulated annealing
    - "greedy": Fast greedy algorithm
    - "two_phase": Combined approach (default)
    
    Args:
        chunks: List of chunk dictionaries
        collections: Collection specifications
        collection_mode: "word" or "chunk" for size calculation
        algorithm: Algorithm choice ("improved_sa", "greedy", or "two_phase")
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

    # Define value extractor based on collection mode
    if collection_mode == "word":
        value_extractor = compute_total_wc
    elif collection_mode == "chunk":
        value_extractor = lambda collection: len(collection)
    else:
        print("aggregate_chunks: Invalid collection_mode (must be 'word' or 'chunk').")
        return None
    
    # Select algorithm and run optimization
    if algorithm == "improved_sa":
        # Create minimal initial solution
        # Start with each chunk in its own collection
        initial_state = []
        for chunk in chunks:
            collection_chunks = [chunk]
            initial_state.append({
                'chunks': collection_chunks, 
                'collection': get_collection_index(collection_chunks, collections, value_extractor)
            })
        # Run improved simulated annealing
        best_state = improved_simulated_annealing(
            initial_state, collections, value_extractor, chunks, time_limit, max_iter
        )
    
    elif algorithm == "greedy":
        # Run greedy algorithm
        best_state = greedy_solution(chunks, collections, value_extractor)
    
    elif algorithm == "two_phase":
        # Run two-phase optimization
        best_state = two_phase_optimization(chunks, collections, value_extractor, time_limit)
    
    else:
        print(f"aggregate_chunks: Invalid algorithm '{algorithm}'")
        print("Valid options are: 'improved_sa', 'greedy', 'two_phase'")
        return None
    
    # Verify solution validity
    for coll in best_state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: Solution violates hard constraint")
            return None
    
    return best_state


""" Utility Functions for Evaluation """

def evaluate_solution(state: List[Dict[str, Any]], 
                     collections: List[Dict[str, Any]], 
                     value_extractor: Callable) -> Dict[str, Any]:
    """
    Evaluate a solution with detailed metrics.
    
    Args:
        state: Solution state
        collections: Collection specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        Dict[str, Any]: Evaluation metrics
    """
    # Number of collections
    N = len(state)
    
    # Calculate distribution statistics
    counts = [0] * len(collections)
    for coll in state:
        idx = coll['collection']
        if idx is not None:
            counts[idx] += 1
    
    distribution_stats = []
    mse = 0.0
    for i, collection_obj in enumerate(collections):
        actual_fraction = counts[i] / N if N > 0 else 0
        target_fraction = collection_obj['target_fraction']
        error = actual_fraction - target_fraction
        mse += error ** 2
        distribution_stats.append({
            "range": collection_obj["range"],
            "target": target_fraction,
            "actual": actual_fraction,
            "error": error
        })
    
    mse /= len(collections)
    
    # Count unique topics
    all_topics = set()
    for chunk in [c for coll in state for c in coll['chunks']]:
        all_topics.add(chunk['topic'])
    
    # Return comprehensive metrics
    return {
        "collection_count": N,
        "unique_topics": len(all_topics),
        "theoretical_min_collections": len(all_topics),
        "distribution_stats": distribution_stats,
        "distribution_mse": mse,
        "collection_size_avg": sum(len(coll['chunks']) for coll in state) / N if N > 0 else 0,
        "cost": compute_cost(state, collections, value_extractor),
    }