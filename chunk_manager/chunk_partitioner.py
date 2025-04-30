"""
Chunk partitioning functions for the chunk manager.

This module contains functions to partition a total word count into chunks,
ensuring that each chunk adheres to specified minimum and maximum word counts. 
The distribution of extra words among the chunks is done using a Dirichlet 
distribution, allowing for a more natural allocation of words across the chunks.
"""

import math
import random
from typing import Optional, List
import numpy as np

def _allocate_extras(num_chunks: int, R: int, extra_max: int, dirichlet_a: float) -> List[int]:
    """
    Helper function to distribute a total of R extra words among a number of chunks,
    ensuring that no chunk receives more than extra_max extra words.

    This function uses a Dirichlet distribution with concentration parameter `dirichlet_a`
    to initially partition R among the chunks. If rounding causes any chunk to exceed the
    per-chunk extra limit, the allocation is adjusted (with any surplus redistributed) until
    the total extra exactly matches R.

    Args:
        num_chunks (int): Number of chunks.
        R (int): Total extra words to distribute.
        extra_max (int): Maximum extra words allowed per chunk.
        dirichlet_a (float): Concentration parameter for the Dirichlet distribution.

    Returns:
        List[int]: A list of extra word counts for each chunk.
    """
    extras_float = np.random.dirichlet([dirichlet_a] * num_chunks) * R
    extras = [min(int(round(x)), extra_max) for x in extras_float]
    
    current_sum = sum(extras)
    while current_sum != R:
        if current_sum < R:
            # If the current sum is too low, add one word to a chunk not at its cap.
            candidates = [j for j, val in enumerate(extras) if val < extra_max]
            delta = 1
        else:
            # If the current sum is too high, remove one word from a chunk that has a nonzero extra.
            candidates = [j for j, val in enumerate(extras) if val > 0]
            delta = -1
        
        if not candidates:
            break
        idx = random.choice(candidates)
        extras[idx] += delta
        current_sum += delta
        
    return extras

def split_wc_into_chunks(
    total_wc: int, 
    min_wc: int, 
    max_wc: int, 
    chunk_count_pref: float = 0.5, 
    dirichlet_a: float = 5.0) -> Optional[List[int]]:
    """
    Split a total word count into chunks, with each chunk receiving a word count between min_wc and max_wc.
    
    The number of chunks is determined by linearly interpolating between the minimum and maximum feasible number 
    of chunks (given by total_wc, min_wc, and max_wc) using the preference parameter chunk_count_pref 
    (where 0 selects the fewest chunks and 1 selects the most).
    
    After initially assigning each chunk min_wc words, the remaining words (total_wc minus the base allocation)
    are distributed among the chunks using a Dirichlet distribution with concentration parameter `dirichlet_a`.
    Any extra allocation for a chunk is capped so that its total does not exceed max_wc.
    
    Args:
        total_wc (int): Total word count to partition.
        min_wc (int): Minimum words per chunk.
        max_wc (int): Maximum words per chunk (must be >= min_wc).
        chunk_count_pref (float): Preference in [0, 1] for selecting the number of chunks within the feasible range.
        dirichlet_a (float): Dirichlet concentration parameter for extra word allocation.
    
    Returns:
        Optional[List[int]]: A list of word counts for each chunk that sums to total_wc, or None if inputs are invalid.
    """
    if not isinstance(total_wc, int) or total_wc <= 0:
        print("split_wc_into_chunks: total_wc must be a positive integer")
        return None
    if not isinstance(min_wc, int) or not isinstance(max_wc, int):
        print("split_wc_into_chunks: min_wc and max_wc must be integers")
        return None
    if min_wc <= 0 or max_wc <= 0:
        print("split_wc_into_chunks: min_wc and max_wc must be positive")
        return None
    if min_wc > max_wc:
        print("split_wc_into_chunks: min_wc cannot be greater than max_wc")
        return None
    if not isinstance(chunk_count_pref, float) or not (0.0 <= chunk_count_pref <= 1.0):
        print("split_wc_into_chunks: chunk_count_pref must be a float between 0 and 1")
        return None

    # Determine the feasible range for the number of chunks.
    min_chunks = math.ceil(total_wc / max_wc)
    max_chunks = total_wc // min_wc
    if min_chunks > max_chunks:
        print("split_wc_into_chunks: no possible number of chunks for the given total_wc, min_wc, and max_wc")
        return None

    # Choose a number of chunks based on the linear interpolation.
    chosen_chunks = min_chunks + round(chunk_count_pref * (max_chunks - min_chunks))

    base_allocation = [min_wc] * chosen_chunks
    remaining_words = total_wc - chosen_chunks * min_wc
    extra_max = max_wc - min_wc

    extras = _allocate_extras(chosen_chunks, remaining_words, extra_max, dirichlet_a)
    chunks = [base_allocation[i] + extras[i] for i in range(chosen_chunks)]
    return chunks

def allocate_wc_to_chunks(
    num_chunks: int, 
    min_wc: int, 
    max_wc: int, 
    chunk_size_pref: float = 0.5,
    dirichlet_a: float = 5.0) -> Optional[List[int]]:
    """
    Allocate word counts to a fixed number of chunks, ensuring each chunk receives between min_wc and max_wc words.
    
    This function assumes that the number of chunks is predetermined and does not operate under a fixed total word count
    budget. Each chunk is initially assigned min_wc words. Then, a total extra word count is randomly selected between 0 
    and the maximum possible extra (i.e. (max_wc - min_wc) * num_chunks), and these extra words are distributed among 
    the chunks using a Dirichlet distribution with concentration parameter `dirichlet_a`. Any allocation exceeding the 
    per-chunk extra limit of (max_wc - min_wc) is capped, with any remainder redistributed until the total extra allocation 
    matches the target.
    
    Args:
        num_chunks (int): The fixed number of chunks for allocation.
        min_wc (int): Minimum words per chunk.
        max_wc (int): Maximum words per chunk (must be >= min_wc).
        chunk_size_pref (float): Preference in [0, 1] for selecting the total extra word count to be allocated.
        dirichlet_a (float): Dirichlet concentration parameter; higher values yield more uniform extra allocations.
    
    Returns:
        Optional[List[int]]: A list of allocated word counts for each chunk, or None if inputs are invalid.
    """
    if not isinstance(num_chunks, int) or num_chunks <= 0:
        print("allocate_chunk_wc: num_chunks must be a positive integer")
        return None
    if not isinstance(min_wc, int) or not isinstance(max_wc, int):
        print("allocate_chunk_wc: min_wc and max_wc must be integers")
        return None
    if min_wc <= 0 or max_wc <= 0:
        print("allocate_chunk_wc: min_wc and max_wc must be positive")
        return None
    if min_wc > max_wc:
        print("allocate_chunk_wc: min_wc cannot be greater than max_wc")
        return None

    base_allocation = [min_wc] * num_chunks
    extra_max = max_wc - min_wc
    max_total_extra = extra_max * num_chunks

    # Randomly choose a total extra word count between 0 and the maximum possible.
    total_extra = int(round(chunk_size_pref * max_total_extra))
    extras = _allocate_extras(num_chunks, total_extra, extra_max, dirichlet_a)
    chunks = [base_allocation[i] + extras[i] for i in range(num_chunks)]
    return chunks

def get_chunks(rulebook: dict) -> Optional[List[dict]]:
    """
    Generate chunks based on the provided rulebook.
    
    This function takes a rulebook dictionary that specifies the total word count,
    collection mode, and content rules for different topics and sentiments. It generates
    a list of chunks, each represented as a dictionary containing the topic name,
    sentiment, and word count.
    
    Args:
        rulebook (dict): A dictionary containing the rulebook information, including:
            - 'total': Total word count to be partitioned.
            - 'collection_mode': Mode of collection ('word' or 'chunk').
            - 'content_rules': A dictionary of content rules for each topic, including:
                - 'total_proportion': Proportion of total word count for the topic.
                - 'sentiment_proportion': List of proportions for positive, neutral, and negative sentiments.
                - 'chunk_min_wc': Minimum word count per chunk.
                - 'chunk_max_wc': Maximum word count per chunk.
                - 'chunk_pref': Preference for chunk size distribution.
                - 'chunk_wc_distribution': Dirichlet concentration parameter for chunk size distribution.
    Returns:
        Optional[List[dict]]: A list of dictionaries representing the generated chunks, or None if partitioning fails.
    """
    all_chunks_dicts = []
    TOTAL = rulebook['total']
    MODE = rulebook['collection_mode']
    
    for topic_name, topic_dict in rulebook['content_rules'].items():
            
        # Get topic budget
        topic_budget = int(TOTAL * topic_dict['total_proportion'])
        
        # Get sentiment budgets
        for index, sentiment in enumerate(["positive", "neutral", "negative"]):
            topic_sentiment_budget = int(topic_budget * topic_dict['sentiment_proportion'][index])
            
            # Skip if no word count
            if topic_sentiment_budget == 0:
                continue
            
            # Collection mode: 'word'
            if MODE == 'word':
                # Partition topic-sentiment word count into chunks
                chunks = split_wc_into_chunks(
                    total_wc=topic_sentiment_budget, 
                    min_wc=topic_dict['chunk_min_wc'], 
                    max_wc=topic_dict['chunk_max_wc'], 
                    chunk_count_pref=topic_dict['chunk_pref'], 
                    dirichlet_a=topic_dict['chunk_wc_distribution'])
                
            # Collection mode: 'chunk'
            else:
                # Allocate word count to topic-sentiment chunks
                chunks = allocate_wc_to_chunks(
                    num_chunks=topic_sentiment_budget, 
                    min_wc=topic_dict['chunk_min_wc'], 
                    max_wc=topic_dict['chunk_max_wc'], 
                    chunk_size_pref=topic_dict['chunk_pref'], 
                    dirichlet_a=topic_dict['chunk_wc_distribution'])

            # Check if partitioning failed
            if not chunks:
                print(f"get_chunks: Partitioning fail - '{topic_name}' - '{sentiment}' - wc:{topic_sentiment_budget}.")
                return None
                
            # Add chunks to all_chunks if partitioning succeeded
            all_chunks_dicts.extend([{'topic': topic_name, 'sentiment': sentiment, 'wc': i} for i in chunks])
            
    return all_chunks_dicts
