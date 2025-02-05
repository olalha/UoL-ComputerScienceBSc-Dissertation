import math
import random

def _random_partition(N, min_val, max_val, i):
    """
    Partitions a postive int N into i random chunks within specified minimum and maximum values.
    Args:
        N (int): The total value to be partitioned
        min_val (int): Minimum allowed chunk size
        max_val (int): Maximum allowed chunk size
        i (int): The number of chunks to create
    Returns:
        list: A list of i positive integers that sum to N, each between min_val and max_val
    Note:
        - The order of the chunks is randomized
    """

    # Start with minimum values
    chunks = [min_val] * i
    # Calculate remainder to distribute
    R = N - i * min_val
    # Max additional per chunk
    max_extra = max_val - min_val

    # Distribute remainder
    for j in range(i):
        remaining_chunks = i - j - 1
        
        # Calculate min allowed for this chunk (consider remaining chunks)
        allowed_min = max(0, R - remaining_chunks * max_extra)
        # Calculate max allowed for this chunk
        allowed_max = min(max_extra, R)
        
        # Add random amount to chunk and subtract from remainder
        extra = random.randint(allowed_min, allowed_max)
        chunks[j] += extra
        R -= extra

    # Randomize order
    random.shuffle(chunks)
    return chunks

def get_chunks(N, min_val, max_val):
    """
    Find optimal number of chunks to split a postive int N into given min and max chunk size constraints.
    Args:
        N (int): Total number to split into chunks
        min_val (int): Minimum allowed chunk size
        max_val (int): Maximum allowed chunk size
    Returns:
        int: Optimal number of chunks, or None if no valid split exists
    """
    
    # Check if the inputs are valid
    if not isinstance(N, int) or N <= 0:
        return None
    if min_val > max_val:
        return None

    # Compute the feasible range for the number of chunks.
    i_lower = math.ceil(N / max_val)
    i_upper = N // min_val

    # Check if a legal partition exists.
    if i_lower > i_upper:
        return None

    # Find the number of chunks that minimizes the
    # different between avg chunk size and target avg
    target_avg = (min_val + max_val) / 2.0
    best_i = None
    best_diff = float('inf')
    for i in range(i_lower, i_upper + 1):
        diff = abs(N / i - target_avg)
        if diff < best_diff:
            best_diff = diff
            best_i = i

    return _random_partition(N, min_val, max_val, best_i)
