from collections import defaultdict
from chunk_manager.solution_structure import SolutionStructure

def create_greedy_initial_solution(chunks, size_ranges, target_proportions, mode, fill_factor):
    """
    Creates an initial solution using a greedy algorithm.

    Args:
    chunks (list): List of chunks, each as (topic: str, sentiment: str, word_count: int)
    size_ranges (list): List of [min_size, max_size] for each size range
    target_proportions (list): List of desired proportions for each size range
    mode (str): 'word_count' or 'chunk_count' to determine size measurement
    fill_factor (float): Factor to estimate average collection size within ranges

    Returns:
    SolutionStructure: Initial solution structure with allocated chunks
    """
    
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
    
    # Sort chunks
    sorted_chunks = sort_chunks(chunks)
    
    # Allocate chunks
    allocate_chunks(
        solution, 
        sorted_chunks, 
        estimated_collections, 
        size_range_with_oor, 
        mode
    )
    
    return solution

def estimate_collection_counts(chunks, size_ranges, target_proportions, mode, fill_factor):
    """
    Estimates the number of collections needed for each size range.

    Returns:
    dict: Mapping of size range index to estimated number of collections
    """
    total_size = sum(chunk[2] if mode == 'word_count' else 1 for chunk in chunks)
    
    estimated_collections = {}
    for i, (min_size, max_size) in enumerate(size_ranges):
        estimated_size = min_size + fill_factor * (max_size - min_size)
        size_budget = target_proportions[i] * total_size
        estimated_collections[i] = max(0, round(size_budget / estimated_size))
    
    return estimated_collections

def sort_chunks(chunks):
    """
    Sorts chunks by topic frequency (descending) and then by size (descending).
    
    Returns:
    list: Sorted list of chunks
    """
    # Count topic frequencies in a single pass
    topic_counts = defaultdict(int)
    for topic, _, _ in chunks:
        topic_counts[topic] += 1
    
    # Sort chunks in a single operation using a compound key
    # First by topic frequency (descending), then by size (descending)
    sorted_chunks = sorted(chunks, key=lambda chunk: (-topic_counts[chunk[0]], -chunk[2]))
    
    return sorted_chunks

def allocate_chunks(solution, sorted_chunks, estimated_collections, size_ranges, mode):
    """
    Allocates chunks to collections using the greedy algorithm.
    """
    if not sorted_chunks:
        return
        
    # Track allocated collections per size range
    allocated_collections = {i: 0 for i in range(len(size_ranges))}
    current_max_range = len(size_ranges) - 2 # Exclude final out of bounds size range
    
    # Start iterator with the sorted chunks
    chunk_iter = iter(sorted_chunks)
    current_chunk = next(chunk_iter, None)
    if not current_chunk:
        return
    first_topic = current_chunk[0]
    
    # Process chunks with first_topic
    while current_chunk is not None and current_chunk[0] == first_topic:
        # Create a new collection for chunk of this topic as there is no other solution
        new_collection_idx = solution.create_new_collection()
        solution.add_chunks_to_collection(new_collection_idx, [current_chunk])
        current_chunk = next(chunk_iter, None)
    
    # Subsequent batch allocation - continue from where we left off
    while current_chunk is not None:
        added = False
        eligible_collections = get_eligible_collections(solution, current_chunk[0])
        final_collection_idx = None
        
        # Sort eligible collections by average word count (or chunk count)
        if eligible_collections and current_max_range >= 0:
            eligible_collections.sort(key=lambda idx: solution.get_collection_avg_word_count(idx) if solution.get_all_chunks(idx) else 0)
            
            # Try to add to the most suitable collection
            for collection_idx in eligible_collections:
                if can_add_to_collection(solution, collection_idx, current_chunk, size_ranges, current_max_range, mode):
                    solution.add_chunks_to_collection(collection_idx, [current_chunk])
                    final_collection_idx = collection_idx
                    added = True
                    break
        
        # If not added, create a new collection
        if not added:
            new_collection_idx = solution.create_new_collection()
            solution.add_chunks_to_collection(new_collection_idx, [current_chunk])
            final_collection_idx = new_collection_idx
            
        # Update allocated collections count
        collection_range = solution.get_collection_range_idx(final_collection_idx)
        if collection_range >= current_max_range:
            allocated_collections[collection_range] += 1
            if allocated_collections[collection_range] >= estimated_collections[collection_range]:
                current_max_range = next_valid_range(allocated_collections, estimated_collections, current_max_range)
        
        # Get the next chunk
        current_chunk = next(chunk_iter, None)

def get_eligible_collections(solution, topic):
    """
    Returns a list of collection indices that don't contain the given topic.
    """
    return [idx for idx in solution.get_active_collection_indices() if solution.can_add_chunk_to_collection(idx, topic)]

def can_add_to_collection(solution, collection_idx, chunk, size_ranges, current_max_range, mode):
    """
    Checks if a chunk can be added to a collection without exceeding the current max size range.
    """
    current_size = solution.get_collection_size(collection_idx)
    new_size = current_size + (chunk[2] if mode == 'word_count' else 1)
    
    # Test if the new size exceeds the max size of the current range
    collection_current_range = solution.get_collection_range_idx(collection_idx)
    if new_size <= size_ranges[collection_current_range][1]:
        return True # Adding this chunk will not change the range size for this collection
    
    for range_idx in range(current_max_range + 1):
        min_size, max_size = size_ranges[range_idx]
        if min_size <= new_size <= max_size:
            return True
    
    return False

def next_valid_range(allocated_collections, estimated_collections, current_max_range):
    """
    Finds the next valid size range to fill.
    """
    for range_idx in range(current_max_range - 1, -1, -1):
        if allocated_collections[range_idx] < estimated_collections[range_idx]:
            return range_idx
    return -1  # No more valid ranges
