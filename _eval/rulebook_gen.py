import json
import os
import random
import numpy as np
from typing import List, Dict, Optional

# Constants for all hard-coded values
DECIMAL_PLACES = 2

# Validation limits
MIN_CHUNK_SIZE_AVG = 20
MIN_CHUNK_WC = 20
MIN_RANGE_FACTOR = 0.1
MAX_RANGE_FACTOR = 0.9

# Content rule generation constants
MIN_CHUNK_PREF = 0.2
MAX_CHUNK_PREF = 0.8
MIN_CHUNK_WC_DISTRIBUTION = 1.0
MAX_CHUNK_WC_DISTRIBUTION = 5.0

def generate_rulebook(
    mode: str,
    content_title: str,
    total: int,
    topics: List[str],
    # Topic distribution control (higher = more balanced, lower = more skewed)
    topic_concentration: float,
    # Sentiment distribution control (higher = more balanced, lower = more skewed)  
    sentiment_concentration: float,
    # Chunk size parameters
    chunk_size_avg: int,
    chunk_size_max_deviation: int,
    chunk_size_range_factor: float,
    # Collection range parameters
    collection_ranges_count: int,
    collection_ranges_max_val: int,
    collection_ranges_min_val: int,
    # Collection range distribution control (higher = more balanced, lower = more skewed)  
    collection_distribution_concentration: float,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Generates a valid rulebook for testing purposes with customizable characteristics.
    
    Args:
        mode: Either "word" or "chunk"
        content_title: Title/name for the review content
        total: Total word count or chunk count (depending on mode)
        topics: List of topic names to include
        
        topic_concentration: Controls topic distribution (higher = more balanced)
            - 0.8: Highly skewed distribution (few dominant topics)
            - 2.0: Moderately skewed distribution
            - 5.0: Balanced distribution
            - 10.0: Very even distribution
            
        sentiment_concentration: Controls sentiment distribution (higher = more balanced)
            - 0.5: Highly skewed sentiment (one dominant sentiment)
            - 1.5: Moderately skewed sentiment
            - 5.0: Balanced sentiment distribution
            
        chunk_size_avg: Average words per chunk
        
        chunk_size_max_deviation: Controls the maximum deviation from average chunk size
        
        chunk_size_range_factor: Controls the range from min to max word count
            - 0.2: Small range (tight min-max bounds)
            - 0.5: Medium range (moderate min-max bounds)
            - 0.8: Large range (wide min-max bounds)
            
        collection_ranges_count: Number of collection ranges to generate
        
        collection_ranges_max_val: Maximum value of the highest collection range interval
        
        collection_ranges_min_val: Minimum value of the lowest collection range interval
        
        collection_distribution_concentration: Controls distribution of collection ranges
            - 1.0: Highly skewed range distribution
            - 3.0: Moderately skewed range distribution
            - 10.0: Balanced range distribution
            
        random_seed: Optional seed for reproducibility
        
    Returns:
        A valid rulebook dictionary conforming to the expected structure
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if mode not in ["word", "chunk"]:
        raise ValueError("Mode must be either 'word' or 'chunk'")
    
    if total <= 0:
        raise ValueError("Total must be a positive integer")
    
    if not topics or len(topics) == 0:
        raise ValueError("At least one topic must be provided")
    
    if collection_ranges_count < 2:
        raise ValueError("Must have at least 2 collection ranges")
    
    if chunk_size_avg <= MIN_CHUNK_SIZE_AVG:
        raise ValueError(f"chunk_size_avg must be greater than {MIN_CHUNK_SIZE_AVG}")
    
    if not (MIN_RANGE_FACTOR <= chunk_size_range_factor <= MAX_RANGE_FACTOR):
        raise ValueError(f"chunk_size_range_factor must be between {MIN_RANGE_FACTOR} and {MAX_RANGE_FACTOR}")
    
    # Generate content rules
    content_rules = _generate_content_rules(
        topics, 
        mode,
        topic_concentration, 
        sentiment_concentration,
        chunk_size_avg, 
        chunk_size_max_deviation,
        chunk_size_range_factor
    )
    
    # Generate collection ranges
    collection_ranges = _generate_collection_ranges(
        collection_ranges_count,
        collection_ranges_max_val,
        collection_ranges_min_val,
        collection_distribution_concentration,
    )
    
    # Create the rulebook
    rulebook = {
        "collection_mode": mode,
        "content_title": content_title,
        "total": total,
        "content_rules": content_rules,
        "collection_ranges": collection_ranges
    }
    
    # Save the rulebook to a JSON file
    from view_components.file_loader import validate_and_save_rulebook
    import time
    formatted_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    rulebook_name = "Generated Rulebook - " + formatted_time + ".json"
    validate_and_save_rulebook(rulebook_name, rulebook, overwrite=True)
    
    return rulebook

def _round_and_ensure_sum_to_one(values: List[float]) -> List[float]:
    """
    Round values to DECIMAL_PLACES while ensuring they still sum to exactly 1.
    
    This is accomplished by rounding each value and then adjusting the largest value
    to compensate for any rounding errors.
    """
    if not values:
        return []
    
    # Round all values to the specified decimal places
    rounded_values = [round(val, DECIMAL_PLACES) for val in values]
    
    # Calculate the rounding error
    sum_rounded = sum(rounded_values)
    error = 1.0 - sum_rounded
    
    if abs(error) > 1e-10:  # If there's a meaningful error
        # Find the index of the largest value to adjust
        largest_idx = rounded_values.index(max(rounded_values))
        
        # Add the error to the largest value and re-round
        rounded_values[largest_idx] = round(rounded_values[largest_idx] + error, DECIMAL_PLACES)
    
    return rounded_values

def _generate_dirichlet_values(count: int, concentration: float) -> List[float]:
    """
    Generate values using Dirichlet distribution that sum to 1
    
    Args:
        count: Number of values to generate
        concentration: Controls the distribution (higher = more balanced)
    
    Returns:
        List of values that sum to 1, rounded to DECIMAL_PLACES
    """
    # Generate values using Dirichlet distribution
    values = np.random.dirichlet(np.ones(count) * concentration).tolist()
    
    # Round and ensure they sum to exactly 1
    values_rounded = _round_and_ensure_sum_to_one(values)
    
    return values_rounded

def _generate_content_rules(
    topics: List[str],
    mode: str,
    topic_concentration: float,
    sentiment_concentration: float,
    chunk_size_avg: int,
    chunk_size_max_deviation: int,
    chunk_size_range_factor: float
) -> Dict:
    """Generate content rules for each topic based on provided parameters"""
    content_rules = {}
    
    # Generate topic proportions based on concentration parameter
    proportions = _generate_dirichlet_values(len(topics), topic_concentration)
    
    # Generate rules for each topic
    for i, topic in enumerate(topics):
        # Generate sentiment distribution
        sentiment = _generate_dirichlet_values(3, sentiment_concentration)
        
        # Calculate average chunk size for the topic
        min_avg = max(MIN_CHUNK_SIZE_AVG, chunk_size_avg - chunk_size_max_deviation)
        max_avg = chunk_size_avg + chunk_size_max_deviation
        topic_avg = random.randint(min_avg, max_avg)
        
        # Calculate chunk min and max based on average and range factor
        range_size = int(topic_avg * chunk_size_range_factor)
        min_wc = max(MIN_CHUNK_WC, topic_avg - range_size)
        max_wc = topic_avg + range_size
        
        # Get random chunk preference value between MIN_CHUNK_PREF and MAX_CHUNK_PREF
        chunk_pref = round(random.uniform(MIN_CHUNK_PREF, MAX_CHUNK_PREF), DECIMAL_PLACES)
        
        # Get random chunk word count distribution between MIN_CHUNK_WC_DISTRIBUTION and MAX_CHUNK_WC_DISTRIBUTION
        chunk_wc_distribution = round(random.uniform(
            MIN_CHUNK_WC_DISTRIBUTION, MAX_CHUNK_WC_DISTRIBUTION), DECIMAL_PLACES)
        
        content_rules[topic] = {
            "total_proportion": proportions[i],
            "sentiment_proportion": sentiment,
            "chunk_min_wc": min_wc,
            "chunk_max_wc": max_wc,
            "chunk_pref": chunk_pref,
            "chunk_wc_distribution": chunk_wc_distribution
        }
    
    return content_rules

def _generate_collection_ranges(
    count: int, 
    ranges_max: int,
    ranges_min: int,
    distribution_concentration: float,
) -> List[Dict]:
    """Generate collection ranges based on parameters"""
    collection_ranges = []
    
    # Generate target fractions based on distribution concentration
    target_fractions = _generate_dirichlet_values(count, distribution_concentration)
    
    ranges_max = int(ranges_max)
    ranges_min = int(ranges_min)

    range_span = max(count, int(ranges_max - ranges_min + 1))
    
    # Calculate range size to distribute evenly across the range_span
    range_size = max(1, range_span // count) - 1 # Subtract 1 to account for overlap
    
    # Create the ranges
    current_start = ranges_min
    currend_end = current_start + range_size
    for i in range(count):
        collection_ranges.append({
            "range": [current_start, currend_end],
            "target_fraction": target_fractions[i]
        })
        current_start = currend_end + 1
        currend_end = current_start + range_size
    
    return collection_ranges
