import random
import math
import time
import copy

# ============================
# Input Validation Functions
# ============================

def validate_chunks(chunks):
    if not isinstance(chunks, list):
        print("allocate_chunks: Chunks must be a list.")
        return False
    for chunk in chunks:
        if not isinstance(chunk, dict):
            print("allocate_chunks: Each chunk must be a dict.")
            return False
        for key in ['topic', 'sentiment', 'wc']:
            if key not in chunk:
                print(f"allocate_chunks: Each chunk must have key '{key}'.")
                return False
        if not isinstance(chunk['topic'], str):
            print("allocate_chunks: Chunk 'topic' must be a string.")
            return False
        if not (isinstance(chunk['sentiment'], str) and len(chunk['sentiment']) == 3):
            print("allocate_chunks: Chunk 'sentiment' must be a three-letter string.")
            return False
        if not isinstance(chunk['wc'], int):
            print("allocate_chunks: Chunk 'wc' must be an integer.")
            return False
    return True

def validate_buckets(buckets):
    if not isinstance(buckets, list):
        print("allocate_chunks: Buckets must be a list.")
        return False
    
    total_fraction = 0.0
    ranges = []
    min_value = float('inf')
    max_value = float('-inf')
    
    for bucket in buckets:
        if not isinstance(bucket, dict):
            print("allocate_chunks: Each bucket must be a dict.")
            return False
        for key in ['range', 'target_fraction']:
            if key not in bucket:
                print(f"allocate_chunks: Each bucket must have '{key}'.")
                return False
        if not (isinstance(bucket['range'], tuple) and len(bucket['range']) == 2):
            print("allocate_chunks: Bucket 'range' must be a tuple of two ints.")
            return False
        low, high = bucket['range']
        if not (isinstance(low, int) and isinstance(high, int)):
            print("allocate_chunks: Bucket range values must be integers.")
            return False
        if low >= high:
            print("allocate_chunks: Bucket range low must be less than high.")
            return False
        if not isinstance(bucket['target_fraction'], (int, float)):
            print("allocate_chunks: Bucket 'target_fraction' must be numeric.")
            return False
            
        # Track min and max values
        min_value = min(min_value, low)
        max_value = max(max_value, high)
        ranges.append((low, high))
        total_fraction += bucket['target_fraction']
    
    # Check for overlapping ranges
    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            print("allocate_chunks: Bucket ranges must not overlap.")
            return False
    
    # Check for gaps in ranges
    for i in range(len(ranges) - 1):
        if ranges[i][1] + 1 < ranges[i + 1][0]:
            print("allocate_chunks: Bucket ranges must not have gaps.")
            return False
    
    if abs(total_fraction - 1.0) > 1e-6:
        print("allocate_chunks: Sum of target fractions must equal 1.")
        return False
        
    return True

# =========================================
# Functions for Bucket Assignment & Metrics
# =========================================

def compute_total_wc(collection):
    return sum(chunk['wc'] for chunk in collection)

def get_bucket_index(collection, buckets):
    """Return the index of the bucket into which this collection falls,
    based on its total word count. Returns None if no bucket matches."""
    total_wc = compute_total_wc(collection)
    for i, bucket in enumerate(buckets):
        low, high = bucket['range']
        if low <= total_wc <= high:
            return i
    return None

# =========================================
# State Initialization and Hard Constraint Check
# =========================================

def valid_collection(collection):
    """A collection is valid if it does not contain duplicate topics."""
    topics = [chunk['topic'] for chunk in collection]
    return len(topics) == len(set(topics))

def initial_solution(chunks, buckets):
    """
    Simple initial state: Put each chunk in its own collection.
    (This always respects the hard constraint.)
    """
    state = []
    for chunk in chunks:
        collection = [chunk]
        state.append({'chunks': collection, 'bucket': get_bucket_index(collection, buckets)})
    return state

# =========================================
# Heuristic / Penalty Calculation
# =========================================

def compute_cost(state, buckets):
    """
    Compute the overall penalty as the sum over buckets of the squared
    difference between the actual fraction of collections in that bucket
    and the target fraction.
    """
    N = len(state)
    counts = [0] * len(buckets)
    for coll in state:
        idx = coll['bucket']
        if idx is not None:
            counts[idx] += 1
    penalty = 0.0
    for i, bucket in enumerate(buckets):
        actual_fraction = counts[i] / N if N > 0 else 0
        target_fraction = bucket['target_fraction']
        penalty += (actual_fraction - target_fraction) ** 2
    return penalty

def update_bucket_for_collection(collection_obj, buckets):
    """
    Update the bucket assignment for a single collection and return a tuple
    (old_bucket, new_bucket) so that an incremental cost update can be performed.
    """
    new_bucket = get_bucket_index(collection_obj['chunks'], buckets)
    old_bucket = collection_obj['bucket']
    collection_obj['bucket'] = new_bucket
    return old_bucket, new_bucket

# =========================================
# Neighborhood Moves for Simulated Annealing
# =========================================

def propose_neighbor(state, buckets):
    """
    Propose a neighboring state by randomly selecting one of the following moves:
      - move: Remove a chunk from one collection and add it to another (or new) collection.
      - swap: Swap a chunk between two collections.
      - merge: Merge two collections (if valid).
      - split: Split one collection into two.
    The move is only accepted if it maintains the hard constraint.
    Returns a new state (deep copy) if a move was made, or None if no move was possible.
    """
    new_state = copy.deepcopy(state)
    move_made = False

    move_type = random.choice(["move", "swap", "merge", "split"])

    if move_type == "move":
        # Choose a random source collection with at least one chunk.
        if len(new_state) == 0:
            return None
        source_idx = random.randint(0, len(new_state) - 1)
        if len(new_state[source_idx]['chunks']) == 0:
            return None
        source_coll = new_state[source_idx]['chunks']
        chunk_idx = random.randint(0, len(source_coll) - 1)
        chunk = source_coll.pop(chunk_idx)
        # Randomly decide to move to an existing collection or create a new one.
        if new_state and random.random() < 0.5 and len(new_state) > 1:
            target_candidates = [i for i in range(len(new_state)) if i != source_idx]
            target_idx = random.choice(target_candidates)
            target_coll = new_state[target_idx]['chunks']
            # Check hard constraint: chunk's topic must not be present.
            if any(c['topic'] == chunk['topic'] for c in target_coll):
                # Revert move
                source_coll.insert(chunk_idx, chunk)
                return None
            target_coll.append(chunk)
            update_bucket_for_collection(new_state[target_idx], buckets)
            move_made = True
        else:
            # Create a new collection with the chunk.
            new_state.append({'chunks': [chunk], 'bucket': get_bucket_index([chunk], buckets)})
            move_made = True
        update_bucket_for_collection(new_state[source_idx], buckets)
        if len(new_state[source_idx]['chunks']) == 0:
            # Remove the empty collection.
            del new_state[source_idx]

    elif move_type == "swap":
        if len(new_state) < 2:
            return None
        idx1, idx2 = random.sample(range(len(new_state)), 2)
        coll1 = new_state[idx1]['chunks']
        coll2 = new_state[idx2]['chunks']
        if len(coll1) == 0 or len(coll2) == 0:
            return None
        pos1 = random.randint(0, len(coll1) - 1)
        pos2 = random.randint(0, len(coll2) - 1)
        chunk1 = coll1[pos1]
        chunk2 = coll2[pos2]
        # Check if swapping preserves the constraint.
        temp1 = coll1.copy()
        temp2 = coll2.copy()
        temp1[pos1] = chunk2
        temp2[pos2] = chunk1
        if valid_collection(temp1) and valid_collection(temp2):
            coll1[pos1], coll2[pos2] = coll2[pos2], coll1[pos1]
            update_bucket_for_collection(new_state[idx1], buckets)
            update_bucket_for_collection(new_state[idx2], buckets)
            move_made = True
        else:
            return None

    elif move_type == "merge":
        if len(new_state) < 2:
            return None
        idx1, idx2 = random.sample(range(len(new_state)), 2)
        coll1 = new_state[idx1]['chunks']
        coll2 = new_state[idx2]['chunks']
        merged = coll1 + coll2
        if valid_collection(merged):
            new_state[idx1]['chunks'] = merged
            update_bucket_for_collection(new_state[idx1], buckets)
            del new_state[idx2]
            move_made = True
        else:
            return None

    elif move_type == "split":
        # Attempt to split a collection with at least 2 chunks.
        candidate_indices = [i for i, coll in enumerate(new_state) if len(coll['chunks']) >= 2]
        if not candidate_indices:
            return None
        idx = random.choice(candidate_indices)
        coll = new_state[idx]['chunks']
        split_point = random.randint(1, len(coll) - 1)
        new_coll1 = coll[:split_point]
        new_coll2 = coll[split_point:]
        if valid_collection(new_coll1) and valid_collection(new_coll2):
            new_state[idx]['chunks'] = new_coll1
            new_state[idx]['bucket'] = get_bucket_index(new_coll1, buckets)
            new_state.append({'chunks': new_coll2, 'bucket': get_bucket_index(new_coll2, buckets)})
            move_made = True
        else:
            return None

    return new_state if move_made else None

# =========================================
# Simulated Annealing Search
# =========================================

def simulated_annealing(initial_state, buckets, time_limit=10, max_iter=10000):
    start_time = time.time()
    current_state = initial_state
    best_state = copy.deepcopy(initial_state)
    current_cost = compute_cost(current_state, buckets)
    best_cost = current_cost
    T = 1.0
    cooling_rate = 0.999  # Adjust as needed

    iteration = 0
    while (time.time() - start_time < time_limit) and (iteration < max_iter):
        iteration += 1
        neighbor = propose_neighbor(current_state, buckets)
        if neighbor is None:
            continue  # No valid move found; try another iteration.
        new_cost = compute_cost(neighbor, buckets)
        delta = new_cost - current_cost
        # Accept if cost is lower or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_cost = new_cost
            if new_cost < best_cost:
                best_state = copy.deepcopy(neighbor)
                best_cost = new_cost
        T *= cooling_rate

    # Check if time limit or iteration limit was reached.
    if time.time() - start_time >= time_limit or iteration >= max_iter:
        print("allocate_chunks: Search time limit has been hit - Returning best found solution.")
    return best_state

# =========================================
# Main Function
# =========================================

def allocate_chunks(chunks, buckets, time_limit=10, max_iter=10000):
    """
    Allocate the list of chunks into collections such that:
      - No collection contains duplicate topics (hard constraint)
      - The overall distribution (by total word count per collection) approximates
        the target fractions provided in the buckets (soft constraint, with penalty).
    If no valid solution is found, print an error and return None.
    """
    # Validate inputs.
    if not validate_chunks(chunks):
        return None
    if not validate_buckets(buckets):
        return None

    # Generate an initial solution (each chunk in its own collection).
    state = initial_solution(chunks, buckets)
    # Verify the hard constraint.
    for coll in state:
        if not valid_collection(coll['chunks']):
            print("allocate_chunks: No solution has been found")
            return None

    # Use simulated annealing to search for an optimal allocation.
    best_state = simulated_annealing(state, buckets, time_limit, max_iter)

    # Final check: ensure no collection violates the hard constraint.
    for coll in best_state:
        if not valid_collection(coll['chunks']):
            print("allocate_chunks: No solution has been found")
            return None

    return best_state

# =========================================
# Example Usage
# =========================================


if __name__ == "__main__":
    chunks = [
        {'topic': 'A', 'sentiment': 'pos', 'wc': 60},
        {'topic': 'B', 'sentiment': 'neu', 'wc': 40},
        {'topic': 'C', 'sentiment': 'neg', 'wc': 120},
        {'topic': 'D', 'sentiment': 'pos', 'wc': 80},
        {'topic': 'E', 'sentiment': 'neu', 'wc': 220},
        {'topic': 'F', 'sentiment': 'neg', 'wc': 210},
        # ... add more chunks as needed
    ]
    
    buckets = [
        {'range': (50, 100), 'target_fraction': 0.4},
        {'range': (101, 200), 'target_fraction': 0.1},
        {'range': (201, 250), 'target_fraction': 0.5},
    ]

    solution = allocate_chunks(chunks, buckets, time_limit=5, max_iter=5000)
    if solution is not None:
        print("Solution found:")
        for coll in solution:
            topics = [chunk['topic'] for chunk in coll['chunks']]
            total_wc = compute_total_wc(coll['chunks'])
            print(f"Collection: {topics}, Total WC: {total_wc}")
    else:
        print("No valid allocation found.")
