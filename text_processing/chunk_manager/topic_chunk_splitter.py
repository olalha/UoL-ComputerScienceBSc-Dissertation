import math
import random
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

def get_chunks(N: int, 
               min_wc: int, 
               max_wc: int, 
               chunk_count_pref: float = 0.5, 
               dirichlet_a: float = 5.0) -> Optional[List[int]]:
    """
    Partition the total word count N into a list of chunk sizes, each between min_wc and max_wc.
    
    The number of chunks is determined by a preference value (chunk_count_pref) between 0 and 1:
      - 0 corresponds to the minimum possible number of chunks.
      - 1 corresponds to the maximum possible number of chunks.
      - 0.5 corresponds to the average (middle) number of chunks.
    
    The extra words beyond the minimum allocation are distributed among the chunks using a Dirichlet distribution
    with concentration parameter `dirichlet_a`. If any chunk would exceed max_wc, the extra is capped and any
    leftover is redistributed among chunks that have not reached their max.
    
    Args:
        N (int): Total word count. Must be a positive integer.
        min_wc (int): Minimum words per chunk. Must be a positive integer.
        max_wc (int): Maximum words per chunk. Must be a positive integer and >= min_wc.
        chunk_count_pref (float): A value in [0, 1] indicating the desired chunk count relative to the feasible range.
                                  0 selects the minimum number of chunks; 1 selects the maximum number.
        dirichlet_a (float): Dirichlet concentration parameter for extra word allocation. Higher values yield
                             more uniform extras.
    
    Returns:
        Optional[List[int]]: A list of chunk sizes that sum to N, or None if there is an input error or no valid partition.
    """
    
    """" Input validation """
    
    if not isinstance(N, int) or N <= 0:
        print("get_chunks: N must be a positive integer")
        return None
    if not isinstance(min_wc, int) or not isinstance(max_wc, int):
        print("get_chunks: min_wc and max_wc must be integers")
        return None
    if min_wc <= 0 or max_wc <= 0:
        print("get_chunks: min_wc and max_wc must be positive")
        return None
    if min_wc > max_wc:
        print("get_chunks: min_wc cannot be greater than max_wc")
        return None
    if not isinstance(chunk_count_pref, float) or not (0.0 <= chunk_count_pref <= 1.0):
        print("get_chunks: chunk_count_pref must be a float between 0 and 1")
        return None
    
    """ Obtain number of chunks """

    # Compute feasible range for number of chunks
    i_lower = math.ceil(N / max_wc)
    i_upper = N // min_wc
    if i_lower > i_upper:
        print("get_chunks: no possible number of chunks for the given N, min_wc, and max_wc")
        return None

    # Linear interpolation: 0 -> i_lower, 1 -> i_upper.
    chosen_i = i_lower + round(chunk_count_pref * (i_upper - i_lower))

    """ Partition N into chosen_i chunks with min_wc """
    
    # Give each chunk the minimum word count
    base = [min_wc] * chosen_i
    # Calculate remaining words to allocate
    R = N - chosen_i * min_wc
    # Maximum extra allowed per chunk
    extra_max = max_wc - min_wc
    
    """" Allocate the remainder using Dirichlet distribution """

    # Obtain a vector of floats that sum to R
    extras_float = np.random.dirichlet([dirichlet_a] * chosen_i) * R
    # Round each extra, capping at extra_max.
    extras = [min(int(round(x)), extra_max) for x in extras_float]
    
    # Adjust the extras to ensure the total sum is N.
    current_sum = sum(extras)
    while current_sum != R:
        if current_sum < R:
            candidates = [j for j, val in enumerate(extras) if val < extra_max]
            delta = 1
        else:
            candidates = [j for j, val in enumerate(extras) if val > 0]
            delta = -1
        
        if not candidates:
            break
        idx = random.choice(candidates)
        extras[idx] += delta
        current_sum += delta
    
    # Combine the base allocation with the extra allocation.
    chunks = [base[j] + extras[j] for j in range(chosen_i)]
    return chunks

def plot_chunks(chunks: List[int], N: int, min_wc: int, max_wc: int) -> None:
    """
    Plots the list of chunk sizes as a sorted bar chart.
    
    Args:
        chunks (List[int]): The list of chunk sizes.
        N (int): Total word count.
        min_wc (int): Minimum word count per chunk.
        max_wc (int): Maximum word count per chunk.
    """
    if not chunks:
        print("plot_chunks: Empty chunk list provided")
        return

    # Sort the chunks for a clearer view.
    sorted_chunks = sorted(chunks)
    
    # Plot the chunks distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_chunks)), sorted_chunks, color='skyblue')
    plt.xlabel("Chunk Index (sorted)")
    plt.ylabel("Word Count")
    plt.title(f"Chunks Distribution\nTotal N = {N}, Min = {min_wc}, Max = {max_wc}, Chunks = {len(chunks)}")
    plt.tight_layout()
    plt.show()

""" Example Usage """

if __name__ == "__main__":
    
    N = 10000
    min_wc = 50
    max_wc = 300

    chunks = get_chunks(N, min_wc, max_wc, chunk_count_pref=0.5, dirichlet_a=100.0)
    if chunks is not None:
        plot_chunks(chunks, N, min_wc, max_wc)
    else:
        print("No valid partition found.")
