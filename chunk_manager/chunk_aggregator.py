"""
Unified Chunk Aggregator Module

This module provides a flexible implementation for solving the chunk aggregation problem,
which aims to:
1. Group chunks into collections with unique topics (hard constraint)
2. Match a target distribution of collection sizes (soft constraint)
3. Minimize the total number of collections (optional objective)

Key features:
- Multiple cost function implementations (basic and enhanced)
- Configurable move selection strategies (static or adaptive)
- Multiple initialization methods (simple or greedy)
- Standardized simulated annealing algorithm with configurable components
"""

import random
import math
import time
import copy
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

""" Tuneable Parameters """

# Simulated annealing parameters
INITIAL_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.1

# Cost function weights (for enhanced cost function)
DISTRIBUTION_WEIGHT = 0.5
COLLECTION_COUNT_WEIGHT = 1.0
OOR_PENALTY_WEIGHT = 1.0

# Greedy algorithm randomization factor
GREEDY_RANDOMIZATION = 0.5

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


def validate_size_ranges(size_ranges):
    """
    Validate the input size_ranges list.
    Each size range must be a dictionary with keys 'range' and 'target_fraction'.
    The 'range' key must contain a tuple of two integers.
    Ranges must not overlap or have gaps between them.
    For collection size size_ranges, ranges like (1, 1) are allowed.
    """
    if not isinstance(size_ranges, list):
        print("validate_size_ranges: Size ranges must be a list.")
        return False
    total_fraction = 0.0
    ranges = []
    for size_range in size_ranges:
        if not isinstance(size_range, dict):
            print("validate_size_ranges: Each size range must be a dict.")
            return False
        for key in ['range', 'target_fraction']:
            if key not in size_range:
                print(f"validate_size_ranges: Each size range must have '{key}'.")
                return False
        if not (isinstance(size_range['range'], (tuple, list)) and len(size_range['range']) == 2):
            print("validate_size_ranges: Size range 'range' must be a tuple of two ints.")
            return False
        low, high = size_range['range']
        if not (isinstance(low, int) and isinstance(high, int)):
            print("validate_size_ranges: Size range values must be integers.")
            return False
        if low > high:
            print("validate_size_ranges: Size range low must be less than or equal to high.")
            return False
        if not isinstance(size_range['target_fraction'], (int, float)):
            print("validate_size_ranges: Size range 'target_fraction' must be numeric.")
            return False
        ranges.append((low, high))
        total_fraction += size_range['target_fraction']
    
    # Check for overlapping ranges
    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            print("validate_size_ranges: Size ranges must not overlap.")
            return False
    
    # Check for gaps in ranges
    for i in range(len(ranges) - 1):
        if ranges[i][1] + 1 < ranges[i + 1][0]:
            print("validate_size_ranges: Size ranges must not have gaps.")
            return False
    
    # Check that fractions sum to approximately 1.0
    if abs(total_fraction - 1.0) > 1e-6:
        print("validate_size_ranges: Sum of target fractions must equal 1.")
        return False
        
    return True

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


def get_size_category_index(collection: List[Dict[str, Any]], 
                          size_ranges: List[Dict[str, Any]], 
                          value_extractor: Callable) -> Optional[int]:
    """
    Return the index of the size category into which this collection falls,
    based on the numerical value returned by value_extractor.
    (For example, total word count or number of chunks.)
    
    Args:
        collection: The collection of chunks to categorize
        size_ranges: List of size range specifications
        value_extractor: Function that computes a numerical value for the collection
        
    Returns:
        Optional[int]: Index of the matching size category, or None if no match
    """
    value = value_extractor(collection)
    for i, size_range in enumerate(size_ranges):
        low, high = size_range['range']
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


def update_size_category(collection_obj: Dict[str, Any], 
                        size_ranges: List[Dict[str, Any]], 
                        value_extractor: Callable) -> Tuple[Optional[int], Optional[int]]:
    """
    Update the size category assignment for a single collection and return a tuple
    (old_category, new_category) so that an incremental cost update can be performed.
    
    Args:
        collection_obj: The collection object to update
        size_ranges: List of size range specifications
        value_extractor: Function to extract value from collection
        
    Returns:
        Tuple[Optional[int], Optional[int]]: Old and new size category indices
    """
    new_category = get_size_category_index(collection_obj['chunks'], size_ranges, value_extractor)
    old_category = collection_obj['size_category']
    collection_obj['size_category'] = new_category
    return old_category, new_category


""" Cost Functions """

def compute_cost_simple(state: List[Dict[str, Any]], size_ranges: List[Dict[str, Any]]) -> float:
    """
    Original simple cost function: Compute the overall penalty as the sum over size categories 
    of the squared difference between the actual fraction of collections in that category
    and the target fraction.
    
    Args:
        state: Current state (list of collections)
        size_ranges: List of size range specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        float: Distribution penalty score (lower is better)
    """
    # Count how many collections we have
    N = len(state)
    
    # Count how many collections belong to each size category
    counts = [0] * len(size_ranges)
    for coll in state:
        idx = coll['size_category']
        if idx is not None:
            counts[idx] += 1
    
    # Calculate penalty using squared difference from targets
    penalty = 0.0
    for i, size_range in enumerate(size_ranges):
        # Calculate actual fraction of collections in this category
        actual_fraction = counts[i] / N if N > 0 else 0
        # Get target fraction
        target_fraction = size_range['target_fraction']
        # Add squared difference to penalty
        penalty += (actual_fraction - target_fraction) ** 2
    
    return penalty


def compute_cost_enhanced(state: List[Dict[str, Any]], size_ranges: List[Dict[str, Any]]) -> float:
    """
    Improved cost function that balances distribution matching with collection minimization.
    
    Args:
        state: Current state (list of collections)
        size_ranges: Size range specifications
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
    counts = [0] * len(size_ranges)
    for coll in state:
        idx = coll['size_category']
        if idx is not None:
            counts[idx] += 1
        else:
            count_oor += 1
    
    distribution_penalty = 0.0
    for i, size_range in enumerate(size_ranges):
        actual_fraction = counts[i] / N if N > 0 else 0
        target_fraction = size_range['target_fraction']
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


""" Move Probability Functions """

def get_move_probs_static() -> Dict[str, float]:
    """
    Return static probabilities for move types (equal probability).
    
    Returns:
        Dict[str, float]: Equal probabilities for each move type
    """
    return {
        "move": 0.25,
        "swap": 0.25,
        "merge": 0.25,
        "split": 0.25
    }


def get_move_probs_adaptive(state: List[Dict[str, Any]]) -> Dict[str, float]:
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


""" Initial Solution Functions """

def simple_solution(chunks: List[Dict[str, Any]], 
                  size_ranges: List[Dict[str, Any]], 
                  value_extractor: Callable) -> List[Dict[str, Any]]:
    """
    Simple initial state: Put each chunk in its own collection.
    (This always respects the hard constraint.)
    
    Args:
        chunks: List of chunk dictionaries
        size_ranges: Size range specifications
        value_extractor: Function to extract value from collections
        
    Returns:
        List[Dict[str, Any]]: Initial state with each chunk in its own collection
    """
    # Start with an empty list of collections
    state = []
    
    # Add each chunk to its own collection
    for chunk in chunks:
        collection_chunks = [chunk]
        # Determine which size category this belongs to
        category_idx = get_size_category_index(collection_chunks, size_ranges, value_extractor)
        # Add to state
        state.append({'chunks': collection_chunks, 'size_category': category_idx})
    
    return state


def greedy_solution(chunks: List[Dict[str, Any]], 
                   size_ranges: List[Dict[str, Any]], 
                   value_extractor: Callable) -> List[Dict[str, Any]]:
    """
    Create a solution using a greedy algorithm that:
    1. Sorts chunks by rarity of topic
    2. For each chunk, finds the best existing collection or creates a new one
    
    Args:
        chunks: List of chunk dictionaries
        size_ranges: Size range specifications
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
                'size_category': get_size_category_index(new_chunks, size_ranges, value_extractor)
            }
            
            # Create temporary state to evaluate cost
            temp_state = state.copy()
            temp_state[i] = new_collection
            cost = compute_cost_enhanced(temp_state, size_ranges)
            
            # If this is better than current best, update best
            if cost < best_cost or (cost == best_cost and random.random() < GREEDY_RANDOMIZATION):
                best_cost = cost
                best_collection = new_collection
                best_index = i
        
        # Create a new collection with just this chunk
        new_coll = {
            'chunks': [chunk],
            'size_category': get_size_category_index([chunk], size_ranges, value_extractor)
        }
        
        # Evaluate creating a new collection
        temp_state = state.copy()
        temp_state.append(new_coll)
        new_cost = compute_cost_enhanced(temp_state, size_ranges)
        
        # Compare with best existing collection
        if new_cost < best_cost or (new_cost == best_cost and random.random() < GREEDY_RANDOMIZATION) or best_index < 0:
            # Create new collection
            state.append(new_coll)
        else:
            # Add to best existing collection
            state[best_index] = best_collection
    
    return state


""" Simulated Annealing Core Functions """

def propose_neighbor(state: List[Dict[str, Any]], 
                    size_ranges: List[Dict[str, Any]], 
                    value_extractor: Callable,
                    chunks: List[Dict[str, Any]],
                    get_move_probs: Callable) -> Optional[List[Dict[str, Any]]]:
    """
    Propose a neighboring state by selecting a move type based on provided probability function,
    and attempting to make that move while maintaining the hard constraint.
    
    Args:
        state: Current state
        size_ranges: Size range specifications
        value_extractor: Function to extract value from collections
        chunks: Original chunks list (for adaptive move probability)
        get_move_probs: Function to get move type probabilities
        
    Returns:
        Optional[List[Dict[str, Any]]]: New state or None if no move was possible
    """
    # Create a deep copy of the current state
    new_state = copy.deepcopy(state)
    move_made = False
    
    # Get move probabilities based on provided function
    move_probs = get_move_probs(state, chunks) if callable(get_move_probs) else get_move_probs
    
    # Select move type based on probabilities
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
            update_size_category(new_state[target_idx], size_ranges, value_extractor)
            move_made = True
        else:
            # Only create a new collection if there are no valid existing collections
            new_state.append({
                'chunks': [chunk], 
                'size_category': get_size_category_index([chunk], size_ranges, value_extractor)
            })
            move_made = True
            
        # Update source collection
        update_size_category(new_state[source_idx], size_ranges, value_extractor)
        
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
            update_size_category(new_state[idx1], size_ranges, value_extractor)
            update_size_category(new_state[idx2], size_ranges, value_extractor)
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
                update_size_category(new_state[idx1], size_ranges, value_extractor)
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
                update_size_category(new_state[idx], size_ranges, value_extractor)
                new_state.append({
                    'chunks': new_coll2, 
                    'size_category': get_size_category_index(new_coll2, size_ranges, value_extractor)
                })
                move_made = True
                break
                
        if not move_made:
            return None

    return new_state if move_made else None


def simulated_annealing(initial_state: List[Dict[str, Any]], 
                      size_ranges: List[Dict[str, Any]], 
                      value_extractor: Callable,
                      chunks: List[Dict[str, Any]],
                      compute_cost: Callable,
                      get_move_probs: Callable,
                      cooling_rate: float = 0.995,
                      time_limit: float = 10, 
                      max_iter: int = None,
                      callback: Callable = None) -> List[Dict[str, Any]]:
    """
    Perform simulated annealing to find an optimal allocation of chunks to collections.
    Uses configurable cost function and move probability function.
    
    Args:
        initial_state: Starting state
        size_ranges: Size range specifications
        value_extractor: Function to extract value from collections
        chunks: Original chunks list (for move probability)
        compute_cost: Function to compute cost of a state
        get_move_probs: Function to determine move probabilities
        time_limit: Maximum run time in seconds
        max_iter: Maximum number of iterations (optional)
        callback: Optional callback function for progress updates
        
    Returns:
        List[Dict[str, Any]]: Best state found
    """
    start_time = time.time()
    current_state = initial_state
    best_state = copy.deepcopy(initial_state)
    current_cost = compute_cost(current_state, size_ranges)
    best_cost = current_cost
    
    # Initialize temperature
    T = INITIAL_TEMPERATURE
    iteration = 0
    stagnant_iterations = 0
    max_stagnant = 1000
    
    if max_iter is None:
        max_iter = float('inf')
        
    # Track progress
    costs = []
    collection_counts = []
    
    while (time.time() - start_time < time_limit) and (iteration < max_iter):
        iteration += 1
        
        # Generate neighbor state
        neighbor = propose_neighbor(current_state, size_ranges, value_extractor, chunks, get_move_probs)
        if neighbor is None:
            continue
            
        # Calculate costs
        new_cost = compute_cost(neighbor, size_ranges)
        delta = new_cost - current_cost
        
        # Accept or reject move
        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_cost = new_cost
            stagnant_iterations = 0
            accepted = True
            
            # Update best if improved
            if new_cost < best_cost:
                best_state = copy.deepcopy(neighbor)
                best_cost = new_cost
        else:
            stagnant_iterations += 1
            
        if callback is not None:
            callback(iteration, T, current_cost, best_cost, current_state, accepted)
        
        # Cool the temperature
        T *= cooling_rate
        
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

def aggregate_chunks(chunks: List[Dict[str, Any]], 
                     size_ranges: List[Dict[str, Any]], 
                     collection_mode: str,
                     initial_solution_fn: str = "simple",
                     cost_function: str = "simple",
                     move_selector: str = "static",
                     cooling_rate: float = 0.995,
                     time_limit: float = 10, 
                     max_iter: int = None,
                     callback: Callable = None) -> Optional[List[Dict[str, Any]]]:
    """
    Aggregate chunks into collections with configurable components:
    - Initial solution: "simple" or "greedy"
    - Cost function: "simple" or "enhanced"
    - Move selector: "static" or "adaptive"
    
    Args:
        chunks: List of chunk dictionaries
        size_ranges: Size range specifications
        collection_mode: "word" or "chunk" for size calculation
        initial_solution_fn: Initial solution method ("simple" or "greedy")
        cost_function: Cost function to use ("simple" or "enhanced")
        move_selector: Move probability method ("static" or "adaptive")
        time_limit: Maximum run time in seconds
        max_iter: Maximum iterations (for simulated annealing)
        callback: Optional callback function for progress updates
        
    Returns:
        Optional[List[Dict[str, Any]]]: The best state found, or None if invalid
    """
    # Validate inputs
    if not validate_chunks(chunks):
        return None
    if not validate_size_ranges(size_ranges):
        return None

    # Define value extractor based on collection mode
    if collection_mode == "word":
        value_extractor = compute_total_wc
    elif collection_mode == "chunk":
        value_extractor = lambda collection: len(collection)
    else:
        print("aggregate_chunks: Invalid collection_mode (must be 'word' or 'chunk').")
        return None
    
    # Select cost function
    if cost_function == "simple":
        cost_fn = compute_cost_simple
    elif cost_function == "enhanced":
        cost_fn = compute_cost_enhanced
    else:
        print(f"aggregate_chunks: Invalid cost function '{cost_function}'")
        print("Valid options are: 'simple', 'enhanced'")
        return None
        
    # Select move probability function
    if move_selector == "static":
        move_probs_fn = get_move_probs_static()
    elif move_selector == "adaptive":
        move_probs_fn = get_move_probs_adaptive
    else:
        print(f"aggregate_chunks: Invalid move selector '{move_selector}'")
        print("Valid options are: 'static', 'adaptive'")
        return None
    
    # Generate initial solution
    if initial_solution_fn == "simple":
        initial_state = simple_solution(chunks, size_ranges, value_extractor)
    elif initial_solution_fn == "greedy":
        initial_state = greedy_solution(chunks, size_ranges, value_extractor)
    else:
        print(f"aggregate_chunks: Invalid initial solution function '{initial_solution_fn}'")
        print("Valid options are: 'simple', 'greedy'")
        return None
    
    # Validate cooling rate
    try:
        if cooling_rate < 0.95 or cooling_rate >= 1.0:
            print("aggregate_chunks: Cooling rate must be in the range [0.95, 1.0).")
            return None
    except ValueError:
        print("aggregate_chunks: Cooling rate must be a float.")
        return None
    
    # Run simulated annealing with configured components
    best_state = simulated_annealing(
        initial_state,
        size_ranges,
        value_extractor,
        chunks,
        cost_fn,
        move_probs_fn,
        cooling_rate,
        time_limit,
        max_iter,
        callback
    )
    
    # Verify solution validity
    for coll in best_state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: Solution violates hard constraint")
            return None
    
    return best_state