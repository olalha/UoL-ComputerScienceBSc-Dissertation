import random
import math
import time
import copy
from typing import Optional
import matplotlib.pyplot as plt

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
        if not (isinstance(collection['range'], tuple) and len(collection['range']) == 2):
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
    
    # Check for overlapping ranges.
    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            print("validate_collections: Collection ranges must not overlap.")
            return False
    
    # Check for gaps in ranges.
    for i in range(len(ranges) - 1):
        if ranges[i][1] + 1 < ranges[i + 1][0]:
            print("validate_collections: Collection ranges must not have gaps.")
            return False
    
    if abs(total_fraction - 1.0) > 1e-6:
        print("validate_collections: Sum of target fractions must equal 1.")
        return False
        
    return True

""" Collection Manipulation Functions """

def compute_total_wc(collection):
    """Return the total word count of a collection."""
    return sum(chunk['wc'] for chunk in collection)

def get_collection_index(collection, collections, value_extractor):
    """
    Return the index of the collection into which this collection falls,
    based on the numerical value returned by value_extractor.
    (For example, total word count or number of chunks.)
    Returns None if no collection matches.
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
    """
    topics = [chunk['topic'] for chunk in collection]
    return len(topics) == len(set(topics))

def initial_solution(chunks, collections, value_extractor):
    """
    Simple initial state: Put each chunk in its own collection.
    (This always respects the hard constraint.)
    """
    state = []
    for chunk in chunks:
        collection_chunks = [chunk]
        state.append({'chunks': collection_chunks, 'collection': get_collection_index(collection_chunks, collections, value_extractor)})
    return state

""" Heuristic / Penalty Functions """

def compute_cost(state, collections, value_extractor):
    """
    Compute the overall penalty as the sum over collections of the squared
    difference between the actual fraction of collections in that collection
    and the target fraction.
    """
    N = len(state)
    counts = [0] * len(collections)
    for coll in state:
        idx = coll['collection']
        if idx is not None:
            counts[idx] += 1
    penalty = 0.0
    for i, collection_obj in enumerate(collections):
        actual_fraction = counts[i] / N if N > 0 else 0
        target_fraction = collection_obj['target_fraction']
        penalty += (actual_fraction - target_fraction) ** 2
    return penalty

def update_collection_for_collection(collection_obj, collections, value_extractor):
    """
    Update the collection assignment for a single collection and return a tuple
    (old_collection, new_collection) so that an incremental cost update can be performed.
    """
    new_collection = get_collection_index(collection_obj['chunks'], collections, value_extractor)
    old_collection = collection_obj['collection']
    collection_obj['collection'] = new_collection
    return old_collection, new_collection

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
    """
    new_state = copy.deepcopy(state)
    move_made = False

    move_type = random.choice(["move", "swap", "merge", "split"])

    if move_type == "move":
        if len(new_state) == 0:
            return None
        source_idx = random.randint(0, len(new_state) - 1)
        if len(new_state[source_idx]['chunks']) == 0:
            return None
        source_coll = new_state[source_idx]['chunks']
        chunk_idx = random.randint(0, len(source_coll) - 1)
        chunk = source_coll.pop(chunk_idx)
        if new_state and random.random() < 0.5 and len(new_state) > 1:
            target_candidates = [i for i in range(len(new_state)) if i != source_idx]
            target_idx = random.choice(target_candidates)
            target_coll = new_state[target_idx]['chunks']
            if any(c['topic'] == chunk['topic'] for c in target_coll):
                source_coll.insert(chunk_idx, chunk)
                return None
            target_coll.append(chunk)
            update_collection_for_collection(new_state[target_idx], collections, value_extractor)
            move_made = True
        else:
            new_state.append({'chunks': [chunk], 'collection': get_collection_index([chunk], collections, value_extractor)})
            move_made = True
        update_collection_for_collection(new_state[source_idx], collections, value_extractor)
        if len(new_state[source_idx]['chunks']) == 0:
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
        temp1 = coll1.copy()
        temp2 = coll2.copy()
        temp1[pos1], temp2[pos2] = coll2[pos2], coll1[pos1]
        if valid_collection(temp1) and valid_collection(temp2):
            coll1[pos1], coll2[pos2] = coll2[pos2], coll1[pos1]
            update_collection_for_collection(new_state[idx1], collections, value_extractor)
            update_collection_for_collection(new_state[idx2], collections, value_extractor)
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
            update_collection_for_collection(new_state[idx1], collections, value_extractor)
            del new_state[idx2]
            move_made = True
        else:
            return None

    elif move_type == "split":
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
            new_state[idx]['collection'] = get_collection_index(new_coll1, collections, value_extractor)
            new_state.append({'chunks': new_coll2, 'collection': get_collection_index(new_coll2, collections, value_extractor)})
            move_made = True
        else:
            return None

    return new_state if move_made else None

def simulated_annealing(initial_state, collections, value_extractor, time_limit=10, max_iter=None):
    """
    Perform simulated annealing to find an optimal allocation of chunks to collections.
    The algorithm runs for a specified time limit or maximum number of iterations.
    Returns the best state found during the search.
    """
    start_time = time.time()
    current_state = initial_state
    best_state = copy.deepcopy(initial_state)
    current_cost = compute_cost(current_state, collections, value_extractor)
    best_cost = current_cost
    
    T = 1.0
    cooling_rate = 0.999
    iteration = 0
    if max_iter is None:
        print(f"simulated_annealing: Starting search for {time_limit} seconds.")
        max_iter = float('inf')
    else:
        print(f"simulated_annealing: Starting search for {time_limit} seconds or {max_iter} iterations.")
    while (time.time() - start_time < time_limit) and (iteration < max_iter):
        iteration += 1
        neighbor = propose_neighbor(current_state, collections, value_extractor)
        if neighbor is None:
            continue
        new_cost = compute_cost(neighbor, collections, value_extractor)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = neighbor
            current_cost = new_cost
            if new_cost < best_cost:
                best_state = copy.deepcopy(neighbor)
                best_cost = new_cost
        T *= cooling_rate
    if time.time() - start_time >= time_limit or iteration >= max_iter:
        print("simulated_annealing: Search time limit has been hit - Returning best found solution.")
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
    The parameter 'collection_mode' should be either "word" or "chunk".
    """
    if not validate_chunks(chunks):
        return None
    if not validate_collections(collections):
        return None

    # Define value_extractor based on collection_mode.
    if collection_mode == "word":
        value_extractor = compute_total_wc
    elif collection_mode == "chunk":
        value_extractor = lambda collection: len(collection)
    else:
        print("aggregate_chunks: Invalid collection_mode (must be 'word' or 'chunk').")
        return None

    state = initial_solution(chunks, collections, value_extractor)
    for coll in state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: Initial solution violates hard constraint")
            return None

    best_state = simulated_annealing(state, collections, value_extractor, time_limit, max_iter)

    for coll in best_state:
        if not valid_collection(coll['chunks']):
            print("aggregate_chunks: No solution has been found")
            return None

    return best_state

""" Visualization Functions """

def visualize_chunk_aggregation(state):
    """
    Visualize chunk aggregation as a stacked bar chart.

    Collections are sorted by total word count, and each collection's chunks 
    are ordered by sentiment ('negative', 'netrual', 'positive'). The chart displays vertical 
    stacked bars for each collection.

    Args:
        state (List[dict]): A list of collections, 
            where each collection is a dict with a 'chunks' key containing a list of 
            chunk dicts. Each chunk must have 'wc' (int) and 'sentiment' (str).
    """
    sentiment_colors = {'negative': '#ff9999', 'netrual': '#ffff99', 'positive': '#99ccff'}

    # Sort collections by total word count and chunks by sentiment.
    all_collections = [c['chunks'] for c in state]
    all_collections = sorted(all_collections, key=lambda col: sum(chunk['wc'] for chunk in col))
    sentiment_order = {'negative': 0, 'netrual': 1, 'positive': 2}

    # Sort chunks by sentiment within each collection.
    for collection in all_collections:
        collection.sort(key=lambda chunk: sentiment_order.get(chunk['sentiment'], 99))

    # Plot the stacked bar chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8
    for i, collection in enumerate(all_collections):
        bottom = 0
        for chunk in collection:
            wc, s = chunk['wc'], chunk['sentiment']
            ax.bar(i, wc, bottom=bottom, width=bar_width,
                   color=sentiment_colors.get(s, 'gray'), edgecolor='none')
            ax.hlines([bottom, bottom + wc], i - bar_width/2, i + bar_width/2, colors='black', linewidth=1)
            bottom += wc
    ax.set(xlabel='Collection', ylabel='Word Count', title='Stacked Bar Chart of Collections by Sentiment')
    plt.tight_layout()
    plt.show()

""" Example Usage """

if __name__ == "__main__":
    chunks = [
        {'topic': 'A', 'sentiment': 'positive', 'wc': 60},
        {'topic': 'B', 'sentiment': 'netrual', 'wc': 40},
        {'topic': 'C', 'sentiment': 'negative', 'wc': 120},
        {'topic': 'D', 'sentiment': 'positive', 'wc': 80},
        {'topic': 'E', 'sentiment': 'netrual', 'wc': 220},
        {'topic': 'F', 'sentiment': 'negative', 'wc': 210},
        # ... add more chunks as needed
    ]
    
    # Example collections for "chunk" mode: (min, max, fraction)
    # 40% of collections must have exactly 1 chunk,
    # 30% must have between 2 and 3 chunks,
    # 30% must have between 4 and 6 chunks.
    collections_chunk = [
        {'range': (1, 1), 'target_fraction': 0.40},
        {'range': (2, 3), 'target_fraction': 0.30},
        {'range': (4, 6), 'target_fraction': 0.30},
    ]
    
    # Example collections for "word" mode (if desired):
    collections_word = [
        {'range': (50, 100), 'target_fraction': 0.40},
        {'range': (101, 200), 'target_fraction': 0.30},
        {'range': (201, 250), 'target_fraction': 0.30},
    ]

    # Choose mode: "chunk" or "word"
    mode = "chunk"
    collections = collections_chunk if mode == "chunk" else collections_word

    solution = aggregate_chunks(chunks, collections, collection_mode=mode, time_limit=5)
    if solution is not None:
        print("Solution found:")
        for coll in solution:
            topics = [chunk['topic'] for chunk in coll['chunks']]
            if mode == "chunk":
                size = len(coll['chunks'])
                print(f"Collection: {topics}, Size: {size}")
            else:
                total_wc = compute_total_wc(coll['chunks'])
                print(f"Collection: {topics}, Total WC: {total_wc}")
        visualize_chunk_aggregation(solution)
    else:
        print("No valid allocation found.")
