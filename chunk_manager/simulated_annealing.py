import random
import math
import copy

# Global constants for simulated annealing algorithm
INITIAL_TEMPERATURE = 100.0
COOLING_RATE = 0.99
MAX_ITERATIONS = 10000

# Penalty factor for out-of-range collections
#   Factor to apply to the deviation of collections 
#   that are out of range as temperature decreases
OOR_PENALTY_FACTOR = 3.0

# Selection bias for chunk selection
# = 0.0: uniform selection
# = 1.0: linear weighting
# > 1.0: exponential weighting
SELECTION_BIAS = 0.0

def optimize_collections_with_simulated_annealing(
    initial_solution, max_iterations=MAX_ITERATIONS, 
    initial_temperature=INITIAL_TEMPERATURE, 
    cooling_rate=COOLING_RATE,
    oor_penalty_factor=OOR_PENALTY_FACTOR,
    selection_bias=SELECTION_BIAS,
    callback=None):
    """
    Optimize collection distribution using simulated annealing.
    
    Args:
        initial_solution: Initial solution structure
        max_iterations: Maximum number of iterations
        initial_temperature: Starting temperature
        cooling_rate: Temperature reduction factor per iteration
        callback: Optional callback function for progress reporting
        
    Returns:
        Optimized solution structure
    """
    valid_params = check_simulated_annealing_params(
        initial_solution,
        max_iterations,
        initial_temperature,
        cooling_rate,
        oor_penalty_factor,
        selection_bias,
        callback
    )
    if not valid_params:
        return None
    
    # Make a copy of the initial solution
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(current_solution)
    
    # Calculate initial cost
    current_cost = calculate_cost(current_solution, oor_penalty_factor)
    best_cost = current_cost
    
    # Initialize parameters
    temperature = initial_temperature
    iteration = 0
    no_improvement_count = 0
    max_no_improvement = max_iterations // 10
    
    # Main simulated annealing loop
    while iteration < max_iterations and no_improvement_count < max_no_improvement:
        
        # Apply a move to current solution
        move_applied, move_info = apply_random_move(current_solution, selection_bias)
        
        # If no valid move could be applied, try again
        if not move_applied:
            iteration += 1
            continue
        
        # Calculate new cost
        new_cost = calculate_cost(current_solution, oor_penalty_factor)
        
        # Decide whether to accept the move
        delta_cost = new_cost - current_cost
        accepted = False
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            # Accept the move
            current_cost = new_cost
            accepted = True
            
            # Check if this is the best solution found so far
            if new_cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = new_cost
                no_improvement_count = 0
                
            else:
                no_improvement_count += 1
        else:
            # Reject the move - revert the changes
            revert_move(current_solution, move_info)
            no_improvement_count += 1
            
        # Report progress via callback if provided
        if callback:
            num_collections = len(current_solution.get_active_collection_indices())
            callback(iteration, temperature, current_cost, best_cost, num_collections, accepted)
            
        # Prepare for next iteration
        temperature *= cooling_rate
        iteration += 1
    
    return best_solution

def calculate_cost(solution, oor_penalty_factor):
    """
    Calculate the cost function based on deviation from target proportions.
    
    Args:
        solution: Solution to evaluate
        normalized_temperature: Temperature normalized to [0,1]
        
    Returns:
        Cost value (lower is better)
    """
    current_distribution = solution.calculate_size_distribution()
    total_cost = 0.0
    
    for i, (current, target) in enumerate(zip(current_distribution, solution.target_proportions)):
        deviation = abs(current - target)
        
        # Apply extra penalty for out-of-range collections
        if i == solution.below_min_range_idx:
            deviation *= oor_penalty_factor
        if i == solution.above_max_range_idx:
            deviation *= oor_penalty_factor
        
        total_cost += deviation
        
    return total_cost

def apply_random_move(solution, selection_bias):
    """
    Apply a random move to the solution.
    
    Args:
        solution: Solution to modify
        normalized_temp: Temperature normalized to [0,1]
        
    Returns:
        (success, move_info)
    """
    # Get current distribution status
    overpopulated = solution.get_overpopulated_ranges()
    underpopulated = solution.get_underpopulated_ranges()
    
    if not overpopulated or not underpopulated:
        return False, None
    
    # Select a move type based on temperature
    move_types = ["transfer_chunk", "swap_chunks", "split_collection"]
    
    # Choose move type
    move_type = random.choices(move_types)[0]
    
    # Apply the selected move
    if move_type == "transfer_chunk":
        return_val = transfer_chunk(solution, overpopulated, underpopulated, selection_bias)
    elif move_type == "swap_chunks":
        return_val = swap_chunks(solution, overpopulated, underpopulated, selection_bias)
    else:
        return_val = split_collection(solution, overpopulated, underpopulated, selection_bias)
    
    return return_val
        
def transfer_chunk(solution, overpopulated, underpopulated, selection_bias):
    """
    Transfer chunks from a collection in an overpopulated range to move it to an underpopulated range.
    
    Strategy:
    1. Remove smallest chunks until the collection falls into an underpopulated range
    2. Optimally redistribute removed chunks to collections in underpopulated ranges
    """
    # Select source range and collection with a bias towards more overpopulated ranges
    range_indices = [idx for idx, _ in overpopulated]
    overpopulation_values = [value ** selection_bias for _, value in overpopulated]
    source_range_idx = random.choices(range_indices, weights=overpopulation_values, k=1)[0]
    
    source_collections = solution.get_collections_by_size_range(source_range_idx)
    
    if not source_collections:
        return False, None
    
    # Choose a collection to remove chunks from with a bias towards larger collections
    collection_weights = [len(solution.get_all_chunks(idx)) ** selection_bias for idx in source_collections]
    source_collection_idx = random.choices(source_collections, weights=collection_weights, k=1)[0]
    
    # Get chunks sorted by size (smallest first)
    source_chunks = solution.get_chunks_by_size_order(source_collection_idx)
    source_chunks.sort(key=lambda x: x[2])  # Sort by word count ascending
    
    if not source_chunks:
        return False, None
    
    # Get underpopulated size ranges
    target_ranges = [idx for idx, _ in underpopulated]
    
    # Find how many chunks to remove to reach an underpopulated range
    chunks_to_remove = []
    current_size = solution.get_collection_size(source_collection_idx)
    remaining_size = current_size
    
    for chunk in source_chunks:
        chunk_size = chunk[2] if solution.mode == "word" else 1
        remaining_size -= chunk_size
        chunks_to_remove.append(chunk)
        
        # Check if new size falls into any underpopulated range
        for range_idx in target_ranges:
            min_size, max_size = solution.size_ranges[range_idx]
            if min_size <= remaining_size <= max_size:
                # Found target size, stop removing chunks
                break
        else:
            # Continue if no underpopulated range matches
            continue
        break
    
    # If no suitable size found, consider removing all chunks
    complete_removal = len(chunks_to_remove) == len(source_chunks)
    
    # Track move info
    move_info = {
        "type": "transfer",
        "from": source_collection_idx,
        "chunks": [c[0] for c in chunks_to_remove],
        "destinations": {},
        "complete_removal": complete_removal
    }
    
    # Remove chunks from source collection
    success = solution.remove_chunks_from_collection(
        source_collection_idx, 
        [c[0] for c in chunks_to_remove]
    )
    
    if not success:
        return False, None
    
    # If complete removal, remove the collection
    if complete_removal:
        solution.remove_collection(source_collection_idx)
        move_info["removed_source"] = True
    
    # Create new collections as needed
    new_collections = []
    
    # Assign destinations for removed chunks
    for chunk in chunks_to_remove:
        # Find eligible collections
        eligible_collections = []
        for coll_idx in solution.get_active_collection_indices():
            if solution.can_add_chunk_to_collection(coll_idx, chunk[0]):
                eligible_collections.append(coll_idx)
        
        # Find optimal destination that results in any underpopulated range
        best_dest = None
        for coll_idx in eligible_collections:
            current_size = solution.get_collection_size(coll_idx)
            new_size = current_size + (chunk[2] if solution.mode == "word" else 1)
            
            # Check if new size falls into any underpopulated range
            for range_idx in target_ranges:
                min_size, max_size = solution.size_ranges[range_idx]
                if min_size <= new_size <= max_size:
                    best_dest = coll_idx
                    break
            
            if best_dest:
                break
        
        # If no optimal destination, use any eligible collection
        if best_dest is None and eligible_collections:
            best_dest = eligible_collections[0]
        
        # If no eligible collections, create new one
        if best_dest is None:
            best_dest = solution.create_new_collection()
            new_collections.append(best_dest)
        
        # Add chunk to destination
        success = solution.add_chunks_to_collection(best_dest, [chunk])
        
        if not success:
            # If addition fails, try to revert
            revert_info = {
                "type": "transfer",
                "from": source_collection_idx,
                "chunks": [c[0] for c in chunks_to_remove],
                "destinations": move_info["destinations"],
                "complete_removal": complete_removal,
                "removed_source": move_info.get("removed_source", False),
                "new_collections": new_collections
            }
            revert_move(solution, revert_info)
            return False, None
        
        # Record destination for potential reversion
        move_info["destinations"][chunk[0]] = best_dest
    
    move_info["new_collections"] = new_collections
    return True, move_info

def swap_chunks(solution, overpopulated, underpopulated, selection_bias):
    """
    Swap chunks between collections to move them toward target ranges.
    
    Strategy:
    1. If overpopulated range is smaller than underpopulated: swap small chunks with large
    2. If overpopulated range is larger: swap large chunks with small
    3. Only confirm swap if it moves at least one collection to a better range
    """
    # Select ranges using weighted random selection based on deviation values
    over_range_indices = [idx for idx, _ in overpopulated]
    over_values = [value ** selection_bias for _, value in overpopulated]
    over_range_idx = random.choices(over_range_indices, weights=over_values, k=1)[0]
    
    under_range_indices = [idx for idx, _ in underpopulated]
    under_values = [value ** selection_bias for _, value in underpopulated]
    under_range_idx = random.choices(under_range_indices, weights=under_values, k=1)[0]
    
    # Determine if overpopulated range is smaller than underpopulated
    over_min, over_max = solution.size_ranges[over_range_idx]
    under_min, under_max = solution.size_ranges[under_range_idx]
    overpopulated_is_smaller = over_max < under_min
    
    # Get collections from both ranges
    over_collections = solution.get_collections_by_size_range(over_range_idx)
    under_collections = solution.get_collections_by_size_range(under_range_idx)
    
    if not over_collections or not under_collections:
        return False, None
    
    # Choose collections with weighted bias towards those with more chunks
    over_weights = [len(solution.get_all_chunks(idx)) ** selection_bias for idx in over_collections]
    collection1_idx = random.choices(over_collections, weights=over_weights, k=1)[0]
    
    under_weights = [len(solution.get_all_chunks(idx)) ** selection_bias for idx in under_collections]
    collection2_idx = random.choices(under_collections, weights=under_weights, k=1)[0]
    
    # Get and sort chunks based on strategy
    chunks1 = solution.get_chunks_by_size_order(collection1_idx)
    chunks2 = solution.get_chunks_by_size_order(collection2_idx)
    
    if not chunks1 or not chunks2:
        return False, None
    
    if overpopulated_is_smaller:
        # Swap small from overpopulated with large from underpopulated
        chunks1.sort(key=lambda x: x[2])  # Smallest first
        chunks2.sort(key=lambda x: x[2], reverse=True)  # Largest first
    else:
        # Swap large from overpopulated with small from underpopulated
        chunks1.sort(key=lambda x: x[2], reverse=True)  # Largest first
        chunks2.sort(key=lambda x: x[2])  # Smallest first
        
    # Find a valid swap that moves collection1 to an underpopulated range
    for chunk1 in chunks1:
        if not solution.can_add_chunk_to_collection(collection2_idx, chunk1[0]):
            continue
        
        for chunk2 in chunks2:
            if not solution.can_add_chunk_to_collection(collection1_idx, chunk2[0]):
                continue
            
            # Calculate new sizes after swap
            coll1_size = solution.get_collection_size(collection1_idx)
            coll2_size = solution.get_collection_size(collection2_idx)
            
            chunk1_size = chunk1[2] if solution.mode == "word" else 1
            chunk2_size = chunk2[2] if solution.mode == "word" else 1
            
            new_coll1_size = coll1_size - chunk1_size + chunk2_size
            
            # Check if swap moves collection1 to an underpopulated range
            for range_idx, _ in underpopulated:
                min_size, max_size = solution.size_ranges[range_idx]
                if min_size <= new_coll1_size <= max_size:
                    # Perform the swap
                    move_info = {
                        "type": "swap",
                        "chunk1": chunk1[0], "collection1": collection1_idx,
                        "chunk2": chunk2[0], "collection2": collection2_idx
                    }
                    
                    # Remove both chunks
                    removed1 = solution.remove_chunks_from_collection(collection1_idx, [chunk1[0]])
                    if not removed1:
                        return False, None
                    
                    removed2 = solution.remove_chunks_from_collection(collection2_idx, [chunk2[0]])
                    if not removed2:
                        # Restore chunk1 (operation is atomic, so we know this will succeed)
                        solution.add_chunks_to_collection(collection1_idx, [chunk1])
                        return False, None
                    
                    # Add chunks to opposite collections
                    added1 = solution.add_chunks_to_collection(collection2_idx, [chunk1])
                    if not added1:
                        # Restore both chunks to original collections
                        solution.add_chunks_to_collection(collection1_idx, [chunk1])
                        solution.add_chunks_to_collection(collection2_idx, [chunk2])
                        return False, None
                    
                    added2 = solution.add_chunks_to_collection(collection1_idx, [chunk2])
                    if not added2:
                        # Restore all chunks
                        solution.remove_chunks_from_collection(collection2_idx, [chunk1[0]])
                        solution.add_chunks_to_collection(collection1_idx, [chunk1])
                        solution.add_chunks_to_collection(collection2_idx, [chunk2])
                        return False, None
                    
                    return True, move_info
    
    return False, None

def split_collection(solution, overpopulated, underpopulated, selection_bias):
    """
    Split a collection from an overpopulated range to create collections in underpopulated ranges.
    """
    # Select range using weighted random selection based on overpopulation values
    range_indices = [idx for idx, _ in overpopulated]
    overpopulation_values = [value ** selection_bias for _, value in overpopulated]
    source_range_idx = random.choices(range_indices, weights=overpopulation_values, k=1)[0]
    
    source_collections = solution.get_collections_by_size_range(source_range_idx)
    
    if not source_collections:
        return False, None
    
    # Choose collection with weighted bias towards those with more chunks
    collection_weights = [len(solution.get_all_chunks(idx)) ** selection_bias for idx in source_collections]
    source_collection_idx = random.choices(source_collections, weights=collection_weights, k=1)[0]
    
    # Get all chunks
    chunks = solution.get_all_chunks(source_collection_idx)
    
    if len(chunks) < 2:  # Need at least 2 chunks to split
        return False, None
    
    # Try different split points to find one that creates collections in underpopulated ranges
    target_ranges = [idx for idx, _ in underpopulated]
    best_split = None
    best_score = -1
    
    # Try a reasonable number of splits (adjust based on collection size)
    max_attempts = min(10, len(chunks) - 1)
    
    # Try random split points
    for _ in range(max_attempts):
        split_point = random.randint(1, len(chunks) - 1)
        
        # Calculate sizes of both parts
        part1 = chunks[:split_point]
        part2 = chunks[split_point:]
        
        if solution.mode == "word":
            size1 = sum(chunk[2] for chunk in part1)
            size2 = sum(chunk[2] for chunk in part2)
        else:
            size1 = len(part1)
            size2 = len(part2)
        
        # Check which ranges these sizes would fall into
        range1 = None
        range2 = None
        
        for r_idx, (min_size, max_size) in enumerate(solution.size_ranges):
            if min_size <= size1 <= max_size:
                range1 = r_idx
            if min_size <= size2 <= max_size:
                range2 = r_idx
        
        # Score based on how well it matches underpopulated ranges
        score = 0
        if range1 in target_ranges:
            score += 1
        if range2 in target_ranges:
            score += 1
        
        # Extra points if both parts are in underpopulated ranges
        if range1 in target_ranges and range2 in target_ranges:
            score += 1
        
        if score > best_score:
            best_score = score
            best_split = (split_point, part1, part2)
    
    # If no good split found, don't proceed
    if best_split is None or best_score == 0:
        return False, None
    
    split_point, part1, part2 = best_split
    
    # Create a new collection and perform the split
    move_info = {
        "type": "split",
        "source": source_collection_idx,
        "chunks1": [c[0] for c in part1],
        "chunks2": [c[0] for c in part2],
    }
    
    new_collection_idx = solution.create_new_collection()
    move_info["new_collection"] = new_collection_idx
    
    # Remove all chunks from source
    removed = solution.remove_chunks_from_collection(
        source_collection_idx, 
        [c[0] for c in chunks]
    )
    
    if not removed:
        solution.remove_collection(new_collection_idx)
        return False, None
    
    # Add chunks back according to the split
    added1 = solution.add_chunks_to_collection(source_collection_idx, part1)
    if not added1:
        # Move failed - revert by restoring all chunks to original collection
        solution.add_chunks_to_collection(source_collection_idx, chunks)
        solution.remove_collection(new_collection_idx)
        return False, None
    
    added2 = solution.add_chunks_to_collection(new_collection_idx, part2)
    if not added2:
        # Move failed - revert the split
        solution.remove_chunks_from_collection(source_collection_idx, [c[0] for c in part1])
        solution.add_chunks_to_collection(source_collection_idx, chunks)
        solution.remove_collection(new_collection_idx)
        return False, None
    
    return True, move_info

def revert_move(solution, move_info):
    """
    Revert a move that was rejected.
    """
    if not move_info:
        return
    
    move_type = move_info.get("type")
    
    if move_type == "transfer":
        # Revert a chunk transfer
        from_idx = move_info["from"]
        chunks_to_return = []
        
        # Get chunks from their destinations
        for chunk_topic, dest_idx in move_info["destinations"].items():
            chunk_data = solution.get_chunk_by_topic(dest_idx, chunk_topic)
            if chunk_data:
                solution.remove_chunks_from_collection(dest_idx, [chunk_topic])
                if chunk_data:
                    chunks_to_return.append(chunk_data)
        
        # If source was removed, recreate it
        if move_info.get("removed_source"):
            new_idx = solution.create_new_collection()
            solution.add_chunks_to_collection(new_idx, chunks_to_return)
        else:
            # Add chunks back to source
            solution.add_chunks_to_collection(from_idx, chunks_to_return)
        
        # Clean up any created collections
        if move_info.get("new_collections"):
            for coll_idx in move_info["new_collections"]:
                solution.remove_collection(coll_idx)
    
    elif move_type == "swap":
        
        # Revert a chunk swap
        chunk1 = move_info["chunk1"]
        collection1 = move_info["collection1"]
        chunk2 = move_info["chunk2"]
        collection2 = move_info["collection2"]
        
        chunk1_data = solution.get_chunk_by_topic(collection2, chunk1)
        chunk2_data = solution.get_chunk_by_topic(collection1, chunk2)
        
        if chunk1_data and chunk2_data:
            solution.remove_chunks_from_collection(collection2, [chunk1])
            solution.remove_chunks_from_collection(collection1, [chunk2])
            
            solution.add_chunks_to_collection(collection1, [chunk1_data])
            solution.add_chunks_to_collection(collection2, [chunk2_data])

    
    elif move_type == "split":
        # Revert a collection split
        source_idx = move_info["source"]
        new_collection_idx = move_info["new_collection"]
        
        # Get all chunks from both collections
        chunks1 = [solution.get_chunk_by_topic(source_idx, topic) for topic in move_info["chunks1"]]
        chunks2 = [solution.get_chunk_by_topic(new_collection_idx, topic) for topic in move_info["chunks2"]]
        
        # Remove from both collections
        solution.remove_chunks_from_collection(source_idx, move_info["chunks1"])
        solution.remove_chunks_from_collection(new_collection_idx, move_info["chunks2"])
        
        # Add all chunks back to source
        all_chunks = [chunk for chunk in chunks1 + chunks2 if chunk is not None]
        solution.add_chunks_to_collection(source_idx, all_chunks)
        
        # Remove the new collection
        solution.remove_collection(new_collection_idx)

def check_simulated_annealing_params(
    initial_solution,
    max_iterations,
    initial_temperature,
    cooling_rate,
    oor_penalty_factor,
    selection_bias,
    callback):
    """
    Validate parameters for simulated annealing optimization.
    Prints error and returns False if any parameter is invalid, otherwise returns True.
    """
    if initial_solution is None:
        print("optimize_collections_with_simulated_annealing: initial_solution must not be None.")
        return False
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        print("optimize_collections_with_simulated_annealing: max_iterations must be a positive integer.")
        return False
    if not isinstance(initial_temperature, (int, float)) or initial_temperature <= 0:
        print("optimize_collections_with_simulated_annealing: initial_temperature must be a positive number.")
        return False
    if not isinstance(cooling_rate, (int, float)) or not (0 < cooling_rate < 1):
        print("optimize_collections_with_simulated_annealing: cooling_rate must be a float between 0 and 1 (exclusive).")
        return False
    if not isinstance(oor_penalty_factor, (int, float)) or oor_penalty_factor < 0:
        print("optimize_collections_with_simulated_annealing: oor_penalty_factor must be a non-negative number.")
        return False
    if not isinstance(selection_bias, (int, float)) or selection_bias < 0:
        print("optimize_collections_with_simulated_annealing: selection_bias must be a non-negative number.")
        return False
    if callback is not None and not callable(callback):
        print("optimize_collections_with_simulated_annealing: callback must be a callable or None.")
        return False
    return True