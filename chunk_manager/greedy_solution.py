"""
Greedy solution for chunk allocation.

This module provides a greedy algorithm to generate an initial solution
for chunk collection allocation. It estimates the number of collections needed
for each size range based on target proportions and chunk sizes, then allocates chunks
to collections in a way that aims to match the target distribution.
"""  

from collections import defaultdict
from chunk_manager.solution_structure import SolutionStructure
from typing import List, Tuple, Dict, Any

def create_greedy_initial_solution(
    chunks: List[Tuple[str, str, int]],
    size_ranges: List[List[int]],
    target_proportions: List[float],
    mode: str,
    fill_factor: float) -> SolutionStructure | None:
    """
    Creates an initial solution using a greedy allocation algorithm.

    This function estimates the number of collections needed for each size range,
    sorts the chunks by topic frequency and size, and then greedily allocates them
    to collections to approximate the target size distribution.

    Args:
        chunks (List[Tuple[str, str, int]]): List of chunks, each as (topic, sentiment, word_count)
        size_ranges (List[List[int]]): List of [min_size, max_size] for each size range
        target_proportions (List[float]): Desired proportions for each size range
        mode (str): 'word' or 'chunk' to determine size measurement
        fill_factor (float): Factor to estimate average collection size within ranges

    Returns:
        SolutionStructure | None: Initial solution structure with allocated chunks, or None if parameters are invalid
    """
    valid_params = check_greedy_solution_params(chunks, size_ranges, target_proportions, mode, fill_factor)
    if not valid_params:
        return None
    
    # Sort size ranges to ensure they are in ascending order
    size_ranges = sorted(size_ranges, key=lambda x: x[0])
    
    # Initialize solution structure
    solution = SolutionStructure(size_ranges, target_proportions, mode)
    
    # Include out of bounds range sizes created by the solution structure
    size_range_with_oor = solution.size_ranges
    target_proportions_with_orr = solution.target_proportions
    
    # Estimate collection counts
    estimated_collections = estimate_collection_counts(
        chunks, 
        size_range_with_oor, 
        target_proportions_with_orr, 
        mode, fill_factor
    )
    
    # Sort chunks by topic frequency and size
    sorted_chunks = sort_chunks(chunks)
    
    # Allocate chunks to collections
    allocate_chunks(
        solution, 
        sorted_chunks, 
        estimated_collections, 
        size_range_with_oor, 
        mode
    )
    
    return solution

def estimate_collection_counts(
    chunks: List[Tuple[str, str, int]],
    size_ranges: List[List[int]],
    target_proportions: List[float],
    mode: str,
    fill_factor: float) -> Dict[int, int]:
    """
    Estimates the number of collections needed for each size range.

    For each size range, this function estimates the average collection size using the fill_factor,
    then determines how many collections are needed to meet the target proportions of the total size.

    Args:
        chunks (List[Tuple[str, str, int]]): List of chunks to allocate
        size_ranges (List[List[int]]): List of [min_size, max_size] for each size range
        target_proportions (List[float]): Desired proportions for each size range
        mode (str): 'word' or 'chunk' to determine size measurement
        fill_factor (float): Factor to estimate average collection size within ranges
        
    Returns:
        Dict[int, int]: Mapping of size range index to estimated number of collections
    """
    total_size = sum(chunk[2] if mode == "word" else 1 for chunk in chunks)
    
    total_number_of_collection_estimate = 0
    for i, (min_size, max_size) in enumerate(size_ranges):
        # Estimate average size of collections in this range using fill_factor
        estimated_size = max(1, min_size + fill_factor * (max_size - min_size))
        # Calculate size budget based on target proportion
        size_budget = target_proportions[i] * total_size
        # Calculate number of collections needed
        total_number_of_collection_estimate += max(0, round(size_budget / estimated_size))
    
    estimated_collections = {}
    for i in range(len(size_ranges)):
        estimated_collections[i] = round(target_proportions[i] * total_number_of_collection_estimate)
    
    return estimated_collections

def sort_chunks(chunks: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
    """
    Sorts chunks by topic frequency (descending) and then by size (descending).

    Chunks with more frequent topics are prioritized, and within the same topic,
    larger chunks are placed first to improve allocation efficiency.

    Args:
        chunks (List[Tuple[str, str, int]]): List of chunks to sort

    Returns:
        List[Tuple[str, str, int]]: Sorted list of chunks by topic frequency and size
    """
    # Count topic frequencies in a single pass
    topic_counts = defaultdict(int)
    for topic, _, _ in chunks:
        topic_counts[topic] += 1
    
    # Sort chunks in a single operation using a compound key
    # First by topic frequency (descending), then by size (descending)
    sorted_chunks = sorted(chunks, key=lambda chunk: (-topic_counts[chunk[0]], -chunk[2]))
    
    return sorted_chunks

def allocate_chunks(
    solution: SolutionStructure,
    sorted_chunks: List[Tuple[str, str, int]],
    estimated_collections: Dict[int, int],
    size_ranges: List[List[int]],
    mode: str) -> None:
    """
    Allocates chunks to collections using a greedy approach.

    The allocation proceeds as follows:
    - Chunks of the most frequent topic are placed in separate collections.
    - Remaining chunks are added to existing collections if possible, otherwise new collections are created.
    - The allocation aims to match the estimated number of collections per size range.

    Args:
        solution (SolutionStructure): Solution structure to add chunks to
        sorted_chunks (List[Tuple[str, str, int]]): Sorted list of chunks to allocate 
        estimated_collections (Dict[int, int]): Estimated number of collections per size range
        size_ranges (List[List[int]]): List of [min_size, max_size] for each size range
        mode (str): 'word' or 'chunk' to determine size measurement
    """
    if not sorted_chunks:
        return
        
    # Track allocated collections per size range
    allocated_collections = {i: 0 for i in range(len(size_ranges))}
    current_max_range = len(size_ranges) - 2  # Exclude final out of bounds size range
    
    # Start iterator with the sorted chunks
    chunk_iter = iter(sorted_chunks)
    current_chunk = next(chunk_iter, None)
    if not current_chunk:
        return
    first_topic = current_chunk[0]
    
    # Place all chunks of the most frequent topic in separate collections
    while current_chunk is not None and current_chunk[0] == first_topic:
        new_collection_idx = solution.create_new_collection()
        solution.add_chunks_to_collection(new_collection_idx, [current_chunk])
        current_chunk = next(chunk_iter, None)
    
    # Allocate remaining chunks
    while current_chunk is not None:
        added = False
        current_collections = solution.get_active_collection_indices()
        eligible_collections = [idx for idx in current_collections if solution.can_add_chunk_to_collection(idx, current_chunk[0])]
        final_collection_idx = None
        
        # Sort eligible collections by average word count (or chunk count)
        if eligible_collections and current_max_range >= 0:
            eligible_collections.sort(key=lambda idx: solution.get_collection_avg_word_count(idx) if solution.get_all_chunks(idx) else 0)
            
            # Try to add to the most suitable collection without exceeding size range limits
            for collection_idx in eligible_collections:
                if can_add_to_collection(solution, collection_idx, current_chunk, size_ranges, current_max_range, mode):
                    solution.add_chunks_to_collection(collection_idx, [current_chunk])
                    final_collection_idx = collection_idx
                    added = True
                    break
        
        # If not added to existing collection based on size constraints
        if not added:
            # Calculate the cost of adding to each eligible collection or creating a new one
            best_cost = float('inf')
            best_collection_idx = None
            
            # Check cost for adding to each eligible collection
            for collection_idx in eligible_collections:
                solution.add_chunks_to_collection(collection_idx, [current_chunk])
                cost = solution.get_total_absolute_deviation()
                if cost < best_cost:
                    best_cost = cost
                    best_collection_idx = collection_idx
                solution.remove_chunks_from_collection(collection_idx, [current_chunk[0]])
            
            # Check cost for creating a new collection
            new_collection_idx = solution.create_new_collection()
            solution.add_chunks_to_collection(new_collection_idx, [current_chunk])
            new_cost = solution.get_total_absolute_deviation()
            
            # Compare costs and decide whether to keep the new collection or not
            if solution.get_collection_range_idx(best_collection_idx) == len(size_ranges) - 1:
                # If the best collection is already out of range, keep the new collection
                final_collection_idx = new_collection_idx
            elif new_cost < best_cost:
                # If the new collection is better, keep it
                final_collection_idx = new_collection_idx
            else:
                # Otherwise, keep the best collection
                final_collection_idx = best_collection_idx
                solution.add_chunks_to_collection(final_collection_idx, [current_chunk])
                solution.remove_collection(new_collection_idx)

        # Update allocated collections count and adjust current max range if needed
        collection_range = solution.get_collection_range_idx(final_collection_idx)
        if collection_range >= current_max_range:
            allocated_collections[collection_range] += 1
            if allocated_collections[collection_range] >= estimated_collections[collection_range]:
                current_max_range = next_valid_range(allocated_collections, estimated_collections, current_max_range)
        
        # Get the next chunk
        current_chunk = next(chunk_iter, None)

def can_add_to_collection(
    solution: SolutionStructure,
    collection_idx: int,
    chunk: Tuple[str, str, int],
    size_ranges: List[List[int]],
    current_max_range: int,
    mode: str) -> bool:
    """
    Checks if a chunk can be added to a collection without exceeding the current max size range.

    Ensures that adding the chunk will not move the collection outside the allowed size range.

    Args:
        solution (SolutionStructure): The current solution
        collection_idx (int): Index of the collection to check
        chunk (Tuple[str, str, int]): The chunk to potentially add
        size_ranges (List[List[int]]): List of [min_size, max_size] for each size range
        current_max_range (int): Maximum allowed size range index
        mode (str): 'word' or 'chunk' to determine size measurement

    Returns:
        bool: True if the chunk can be added, False otherwise
    """
    current_size = solution.get_collection_size(collection_idx)
    new_size = current_size + (chunk[2] if mode == "word" else 1)
    
    # Test if the new size exceeds the max size of the current range
    collection_current_range = solution.get_collection_range_idx(collection_idx)
    if new_size <= size_ranges[collection_current_range][1]:
        return True  # Adding this chunk will not change the range size for this collection
    
    # Check if the new size would place this collection in an acceptable range
    for range_idx in range(current_max_range + 1):
        min_size, max_size = size_ranges[range_idx]
        if min_size <= new_size <= max_size:
            return True
    
    return False

def next_valid_range(
    allocated_collections: Dict[int, int],
    estimated_collections: Dict[int, int],
    current_max_range: int) -> int:
    """
    Finds the next valid size range to fill.

    When a size range has met its target number of collections, this function
    finds the next largest size range that still needs collections.

    Args:
        allocated_collections (Dict[int, int]): Number of collections allocated to each range
        estimated_collections (Dict[int, int]): Target number of collections for each range
        current_max_range (int): Current maximum size range index

    Returns:
        int: Index of the next valid range, or -1 if no more valid ranges
    """
    for range_idx in range(current_max_range - 1, -1, -1):
        if allocated_collections[range_idx] < estimated_collections[range_idx]:
            return range_idx
    return -1  # No more valid ranges

def check_greedy_solution_params(
    chunks: List[Tuple[str, str, int]],
    size_ranges: List[List[int]],
    target_proportions: List[float],
    mode: str,
    fill_factor: float) -> bool:
    """
    Validates the parameters for create_greedy_initial_solution.

    Args:
        chunks (List[Tuple[str, str, int]]): List of chunks
        size_ranges (List[List[int]]): List of [min_size, max_size] pairs
        target_proportions (List[float]): List of proportions for each size range
        mode (str): 'word' or 'chunk'
        fill_factor (float): Fill factor between 0 and 1

    Returns:
        bool: True if all parameters are valid, False otherwise.
    """
    
    # Check chunks
    if not isinstance(chunks, list) or not all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks):
        print("create_greedy_initial_solution: 'chunks' must be a list of (topic, sentiment, wc) tuples.")
        return False

    # Check size_ranges
    if not isinstance(size_ranges, list) or not all(isinstance(r, (list, tuple)) and len(r) == 2 for r in size_ranges):
        print("create_greedy_initial_solution: 'size_ranges' must be a list of [min_size, max_size] pairs.")
        return False
    for r in size_ranges:
        if not (isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)) and r[0] <= r[1]):
            print("create_greedy_initial_solution: Each size range must be [min_size, max_size] with min_size <= max_size.")
            return False

    # Check target_proportions
    if not isinstance(target_proportions, list) or len(target_proportions) != len(size_ranges):
        print("create_greedy_initial_solution: 'target_proportions' must be a list with the same length as 'size_ranges'.")
        return False
    if not all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in target_proportions):
        print("create_greedy_initial_solution: All target proportions must be numbers between 0 and 1.")
        return False
    if abs(sum(target_proportions) - 1.0) > 1e-6:
        print("create_greedy_initial_solution: The sum of target proportions must be 1.")
        return False

    # Check mode
    if mode not in ("word", "chunk"):
        print("create_greedy_initial_solution: 'mode' must be either 'word' or 'chunk'.")
        return False

    # Check fill_factor
    if not isinstance(fill_factor, (int, float)) or not (0 <= fill_factor <= 1):
        print("create_greedy_initial_solution: 'fill_factor' must be a float between 0 and 1.")
        return False

    return True
