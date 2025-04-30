"""
Module for analyzing and extracting metrics from dataset structures.

This module provides functions to calculate various metrics from dataset structures
without modifying the original dataset. It's designed to replace in-dataset metadata
storage with on-demand calculations.
"""

from typing import Dict, Any, List, Set

def get_basic_counts(dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Get basic count metrics from a dataset.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Dictionary with total word count, chunk count, collections count
    """
    if not dataset or 'collections' not in dataset:
        return {"total_wc": 0, "total_cc": 0, "collections_count": 0}
    
    collections = dataset.get('collections', [])
    total_wc = 0
    total_cc = 0
    
    for collection in collections:
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            total_wc += chunk_dict.get('wc', 0)
            total_cc += 1
    
    return {
        "total_wc": total_wc,
        "total_cc": total_cc,
        "collections_count": len(collections)
    }


def get_text_presence_percentages(dataset: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the percentage of chunks and collections that have text.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Dictionary with percentages of chunks and collections containing text
    """
    basic_counts = get_basic_counts(dataset)
    total_cc = basic_counts.get("total_cc", 0)
    collections_count = basic_counts.get("collections_count", 0)
    
    if not dataset or 'collections' not in dataset:
        return {"chunks_with_text_percent": 0.0, "collections_with_text_percent": 0.0}
    
    collections = dataset.get('collections', [])
    chunks_with_text = 0
    collections_with_text = 0
    
    for collection in collections:
        collection_has_text = collection.get('collection_text') is not None
        chunk_text_count = 0
        
        for chunk in collection.get('chunks', []):
            if chunk.get('chunk_text') is not None:
                chunk_text_count += 1
        
        chunks_with_text += chunk_text_count
        if collection_has_text:
            collections_with_text += 1
    
    chunks_with_text_percent = (chunks_with_text / total_cc) * 100 if total_cc > 0 else 0.0
    collections_with_text_percent = (collections_with_text / collections_count) * 100 if collections_count > 0 else 0.0
    
    return {
        "chunks_text_percent": chunks_with_text_percent,
        "collections_text_percent": collections_with_text_percent
    }


def get_collection_distribution(dataset: Dict[str, Any]) -> Dict[int, int]:
    """
    Get the distribution of chunks across collections.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Dictionary mapping chunk counts to the number of collections with that count
    """
    if not dataset or 'collections' not in dataset:
        return {}
    
    collection_cc_distribution = {}
    
    for collection in dataset.get('collections', []):
        collection_cc = len(collection.get('chunks', []))
        collection_cc_distribution[collection_cc] = collection_cc_distribution.get(collection_cc, 0) + 1
        
    return collection_cc_distribution


def get_sentiment_distribution(dataset: Dict[str, Any], mode: str) -> Dict[str, int]:
    """
    Get the distribution of chunks or words by sentiment.
    
    Args:
        dataset: The dataset dictionary
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        Dictionary mapping sentiments to their counts
    """
    if not dataset or 'collections' not in dataset:
        return {}
    if mode not in ['chunk', 'word']:
        return {}
    
    sentiment_distribution = {}
    
    for collection in dataset.get('collections', []):
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            sentiment = chunk_dict.get('sentiment', 'Unknown')
            
            if mode == 'chunk':
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            else:
                wc = chunk_dict.get('wc', 0)
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + wc
    
    return sentiment_distribution


def get_topic_distribution(dataset: Dict[str, Any], mode: str) -> Dict[str, int]:
    """
    Get the distribution of chunks or words by topic.
    
    Args:
        dataset: The dataset dictionary
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        Dictionary mapping topics to their counts
    """
    if not dataset or 'collections' not in dataset:
        return {}
    if mode not in ['chunk', 'word']:
        return {}
    
    topic_distribution = {}
    
    for collection in dataset.get('collections', []):
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            topic = chunk_dict.get('topic', 'Unknown')
            
            if mode == 'chunk':
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
            else:
                wc = chunk_dict.get('wc', 0)
                topic_distribution[topic] = topic_distribution.get(topic, 0) + wc
    
    return topic_distribution


def get_topic_sentiment_distribution(dataset: Dict[str, Any], mode: str) -> Dict[str, int]:
    """
    Get the distribution of chunks or words by topic-sentiment combinations.
    
    Args:
        dataset: The dataset dictionary
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        Dictionary mapping 'topic - sentiment' to their counts
    """
    if not dataset or 'collections' not in dataset:
        return {}
    if mode not in ['chunk', 'word']:
        return {}
    
    ts_distribution = {}
    
    for collection in dataset.get('collections', []):
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            topic = chunk_dict.get('topic', 'Unknown')
            sentiment = chunk_dict.get('sentiment', 'Unknown')
            ts_key = f"{topic} - {sentiment}"
            
            if mode == 'chunk':
                ts_distribution[ts_key] = ts_distribution.get(ts_key, 0) + 1
            else:
                wc = chunk_dict.get('wc', 0)
                ts_distribution[ts_key] = ts_distribution.get(ts_key, 0) + wc
    
    return ts_distribution


def get_collection_metrics(dataset: Dict[str, Any], 
                          collection_idx: int) -> Dict[str, Any]:
    """
    Get metrics for a specific collection by index.
    
    Args:
        dataset: The dataset dictionary
        collection_idx: Zero-based index of the collection
        
    Returns:
        Dictionary with metrics for the specified collection
    """
    if not dataset or 'collections' not in dataset:
        return {}
    
    collections = dataset.get('collections', [])
    if collection_idx < 0 or collection_idx >= len(collections):
        return {}
    
    collection = collections[collection_idx]
    
    collection_wc = 0
    collection_cc = len(collection.get('chunks', []))
    sentiment_counts = {}
    topic_counts = {}
    
    for chunk in collection.get('chunks', []):
        chunk_dict = chunk.get('chunk_dict', {})
        wc = chunk_dict.get('wc', 0)
        collection_wc += wc
        
        sentiment = chunk_dict.get('sentiment', 'Unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        topic = chunk_dict.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    return {
        "collection_wc": collection_wc,
        "collection_cc": collection_cc,
        "sentiment_counts": sentiment_counts,
        "topic_counts": topic_counts,
        "has_text": collection.get('collection_text') is not None
    }


def get_collection_sentiment_data(dataset: Dict[str, Any], mode: str) -> Dict[str, Dict[str, int]]:
    """
    Get sentiment distribution for each collection.
    
    Args:
        dataset: The dataset dictionary
        mode: Either "chunk" or "word" to specify count type
        
    Returns:
        Dictionary mapping collection IDs to sentiment distributions
    """
    if not dataset or 'collections' not in dataset:
        return {}
    if mode not in ['chunk', 'word']:
        return {}
    
    collection_sentiment = {}
    
    for i, collection in enumerate(dataset.get('collections', [])):
        collection_id = f"Collection {i+1}"
        sentiment_data = {"positive": 0, "neutral": 0, "negative": 0, "Unknown": 0}
        
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            sentiment = chunk_dict.get('sentiment', 'Unknown')
            
            if mode == 'chunk':
                sentiment_data[sentiment] = sentiment_data.get(sentiment, 0) + 1
            else:
                wc = chunk_dict.get('wc', 0)
                sentiment_data[sentiment] = sentiment_data.get(sentiment, 0) + wc
                
        collection_sentiment[collection_id] = sentiment_data
    
    return collection_sentiment

def get_chunk_data_for_analysis(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract chunk data in a format suitable for analysis.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        List of dictionaries with chunk data
    """
    if not dataset or 'collections' not in dataset:
        return []
    
    chunks_data = []
    
    for i, collection in enumerate(dataset.get('collections', [])):
        for j, chunk in enumerate(collection.get('chunks', [])):
            chunk_dict = chunk.get('chunk_dict', {})
            
            chunks_data.append({
                'collection_idx': i,
                'chunk_idx': j,
                'wc': chunk_dict.get('wc', 0),
                'sentiment': chunk_dict.get('sentiment', 'Unknown'),
                'topic': chunk_dict.get('topic', 'Unknown'),
                'has_text': chunk.get('chunk_text') is not None
            })
    
    return chunks_data

def get_unique_topics(dataset: Dict[str, Any]) -> Set[str]:
    """
    Get the set of all unique topics in the dataset.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Set of unique topic strings
    """
    if not dataset or 'collections' not in dataset:
        return set()
    
    topics = set()
    
    for collection in dataset.get('collections', []):
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            topic = chunk_dict.get('topic', 'Unknown')
            topics.add(topic)
    
    return topics


def get_unique_sentiments(dataset: Dict[str, Any]) -> Set[str]:
    """
    Get the set of all unique sentiments in the dataset.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Set of unique sentiment strings
    """
    if not dataset or 'collections' not in dataset:
        return set()
    
    sentiments = set()
    
    for collection in dataset.get('collections', []):
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            sentiment = chunk_dict.get('sentiment', 'Unknown')
            sentiments.add(sentiment)
    
    return sentiments


def get_min_max_counts(dataset: Dict[str, Any]) -> Dict[str, int]:
    """
    Get minimum and maximum counts for chunks and collections.
    
    Args:
        dataset: The dataset dictionary
        
    Returns:
        Dictionary with min/max word counts for chunks and collection counts
    """
    if not dataset or 'collections' not in dataset:
        return {
            "min_chunk_wc": 0,
            "max_chunk_wc": 0,
            "min_collection_wc": 0,
            "max_collection_wc": 0,
            "min_collection_cc": 0,
            "max_collection_cc": 0
        }
    
    collections = dataset.get('collections', [])

    min_chunk_wc = 0
    max_chunk_wc = 0
    min_collection_wc = 0
    max_collection_wc = 0
    min_collection_cc = 0
    max_collection_cc = 0
    
    for collection in collections:
        collection_wc = 0
        collection_cc = len(collection.get('chunks', []))
        
        for chunk in collection.get('chunks', []):
            chunk_dict = chunk.get('chunk_dict', {})
            wc = chunk_dict.get('wc', 0)
            collection_wc += wc
            
            if wc < min_chunk_wc:
                min_chunk_wc = wc
            if wc > max_chunk_wc:
                max_chunk_wc = wc
        
        if collection_wc < min_collection_wc:
            min_collection_wc = collection_wc
        if collection_wc > max_collection_wc:
            max_collection_wc = collection_wc
        
        if collection_cc < min_collection_cc:
            min_collection_cc = collection_cc
        if collection_cc > max_collection_cc:
            max_collection_cc = collection_cc
            
    return {
        "min_chunk_wc": min_chunk_wc,
        "max_chunk_wc": max_chunk_wc,
        "min_collection_wc": min_collection_wc,
        "max_collection_wc": max_collection_wc,
        "min_collection_cc": min_collection_cc,
        "max_collection_cc": max_collection_cc
    }

def filter_collections(dataset, min_wc=0, max_wc=float('inf'), min_cc=0, max_cc=float('inf'), 
                      topic=None, sentiment=None, has_text=None):
    """
    Filter collections based on specified criteria.
    
    Args:
        dataset: The dataset dictionary
        min_wc: Minimum word count
        max_wc: Maximum word count
        min_cc: Minimum chunk count
        max_cc: Maximum chunk count
        topic: Filter by topic (None for all)
        sentiment: Filter by sentiment (None for all)
        has_text: Filter by text presence (None for all, True for has text, False for no text)
        
    Returns:
        List of indices of collections matching the criteria
    """
    if not dataset or 'collections' not in dataset:
        return []
    
    matching_indices = []
    collections = dataset.get('collections', [])
    
    for i, collection in enumerate(collections):
        collection_counts = get_collection_metrics(dataset, collection_idx=i)
        wc = collection_counts.get("collection_wc", 0)
        cc = collection_counts.get("collection_cc", 0)
        collection_has_text = collection.get("collection_text") is not None
        
        # Check word count and chunk count
        if wc < min_wc or wc > max_wc or cc < min_cc or cc > max_cc:
            continue
            
        # Check text presence
        if has_text is True and not collection_has_text:
            continue
        elif has_text is False and collection_has_text:
            continue
            
        # Check topic and sentiment
        if topic or sentiment:
            has_matching_topic = topic is None
            has_matching_sentiment = sentiment is None
            
            for chunk in collection.get("chunks", []):
                chunk_dict = chunk.get("chunk_dict", {})
                
                if topic and chunk_dict.get("topic") == topic:
                    has_matching_topic = True
                    
                if sentiment and chunk_dict.get("sentiment") == sentiment:
                    has_matching_sentiment = True
                    
                if has_matching_topic and has_matching_sentiment:
                    break
                    
            if not (has_matching_topic and has_matching_sentiment):
                continue
                
        # If we got here, the collection matches all criteria
        matching_indices.append(i)
        
    return matching_indices

def compare_topic_proportions(dataset: dict) -> float:
    """
    Compare required topic proportions vs actual in the dataset.
    Returns a percentage match (0-100).
    """
    rulebook = dataset.get("rulebook", {})
    content_rules = rulebook.get("content_rules", {})
    mode = rulebook.get("collection_mode", "word")
    total = rulebook.get("total", 1)

    # Calculate required proportions
    required = {topic: rules.get("total_proportion", 0) for topic, rules in content_rules.items()}

    # Calculate actual proportions
    actual_counts = {}
    actual_total = 0
    for collection in dataset.get("collections", []):
        for chunk in collection.get("chunks", []):
            chunk_dict = chunk.get("chunk_dict", {})
            topic = chunk_dict.get("topic", "Unknown")
            if mode == "word":
                wc = chunk_dict.get("wc", 0)
                actual_counts[topic] = actual_counts.get(topic, 0) + wc
                actual_total += wc
            else:
                actual_counts[topic] = actual_counts.get(topic, 0) + 1
                actual_total += 1

    actual = {topic: (actual_counts.get(topic, 0) / actual_total) if actual_total > 0 else 0 for topic in required}

    # Calculate match as 1 - sum of absolute differences divided by 2 (to normalize)
    diff_sum = sum(abs(required[topic] - actual.get(topic, 0)) for topic in required)
    match = max(0.0, 1.0 - diff_sum / 2.0)
    return round(match * 100, 2)

def compare_global_sentiment_proportions(dataset: dict) -> float:
    """
    Compare required global sentiment proportions vs actual in the dataset.
    Returns a percentage match (0-100).
    """
    rulebook = dataset.get("rulebook", {})
    content_rules = rulebook.get("content_rules", {})
    mode = rulebook.get("collection_mode", "word")
    total = rulebook.get("total", 1)

    # Aggregate required sentiment proportions (weighted by topic proportions)
    sentiment_labels = ["positive", "neutral", "negative"]
    required = {s: 0.0 for s in sentiment_labels}
    for topic, rules in content_rules.items():
        topic_prop = rules.get("total_proportion", 0)
        sentiment_prop = rules.get("sentiment_proportion", [0, 0, 0])
        for i, s in enumerate(sentiment_labels):
            required[s] += topic_prop * sentiment_prop[i]

    # Calculate actual sentiment proportions
    actual_counts = {s: 0 for s in sentiment_labels}
    actual_total = 0
    for collection in dataset.get("collections", []):
        for chunk in collection.get("chunks", []):
            chunk_dict = chunk.get("chunk_dict", {})
            sentiment = chunk_dict.get("sentiment", "Unknown")
            if sentiment not in sentiment_labels:
                continue
            if mode == "word":
                wc = chunk_dict.get("wc", 0)
                actual_counts[sentiment] += wc
                actual_total += wc
            else:
                actual_counts[sentiment] += 1
                actual_total += 1

    actual = {s: (actual_counts[s] / actual_total) if actual_total > 0 else 0 for s in sentiment_labels}

    # Calculate match as 1 - sum of absolute differences divided by 2 (to normalize)
    diff_sum = sum(abs(required[s] - actual.get(s, 0)) for s in sentiment_labels)
    match = max(0.0, 1.0 - diff_sum / 2.0)
    return round(match * 100, 2)

def compare_topic_sentiment_pair_proportions(dataset: dict) -> float:
    """
    Compare required topic-sentiment pair proportions vs actual in the dataset.
    Returns a percentage match (0-100).
    """
    rulebook = dataset.get("rulebook", {})
    content_rules = rulebook.get("content_rules", {})
    mode = rulebook.get("collection_mode", "word")
    total = rulebook.get("total", 1)
    sentiment_labels = ["positive", "neutral", "negative"]

    # Required proportions for each topic-sentiment pair
    required = {}
    for topic, rules in content_rules.items():
        topic_prop = rules.get("total_proportion", 0)
        sentiment_prop = rules.get("sentiment_proportion", [0, 0, 0])
        for i, s in enumerate(sentiment_labels):
            required[(topic, s)] = topic_prop * sentiment_prop[i]

    # Actual proportions for each topic-sentiment pair
    actual_counts = {k: 0 for k in required}
    actual_total = 0
    for collection in dataset.get("collections", []):
        for chunk in collection.get("chunks", []):
            chunk_dict = chunk.get("chunk_dict", {})
            topic = chunk_dict.get("topic", "Unknown")
            sentiment = chunk_dict.get("sentiment", "Unknown")
            if (topic, sentiment) not in required:
                continue
            if mode == "word":
                wc = chunk_dict.get("wc", 0)
                actual_counts[(topic, sentiment)] += wc
                actual_total += wc
            else:
                actual_counts[(topic, sentiment)] += 1
                actual_total += 1

    actual = {k: (actual_counts[k] / actual_total) if actual_total > 0 else 0 for k in required}

    # Calculate match as 1 - sum of absolute differences divided by 2 (to normalize)
    diff_sum = sum(abs(required[k] - actual.get(k, 0)) for k in required)
    match = max(0.0, 1.0 - diff_sum / 2.0)
    return round(match * 100, 2)

def compare_collection_size_range_distribution(dataset: dict) -> float:
    """
    Compare required collection size range distributions vs actual in the dataset.
    Returns a percentage match (0-100).
    """
    rulebook = dataset.get("rulebook", {})
    collection_ranges = rulebook.get("collection_ranges", [])
    mode = rulebook.get("collection_mode", "word")

    # Prepare required ranges and fractions
    required_ranges = []
    required_fractions = []
    for r in collection_ranges:
        required_ranges.append(tuple(r["range"]))
        required_fractions.append(r["target_fraction"])

    # Count actual collections in each range
    actual_counts = [0 for _ in required_ranges]
    total_collections = 0
    for collection in dataset.get("collections", []):
        if mode == "word":
            size = sum(chunk.get("chunk_dict", {}).get("wc", 0) for chunk in collection.get("chunks", []))
        else:
            size = len(collection.get("chunks", []))
        for i, (low, high) in enumerate(required_ranges):
            if low <= size <= high:
                actual_counts[i] += 1
                break
        total_collections += 1

    actual_fractions = [(count / total_collections) if total_collections > 0 else 0 for count in actual_counts]

    # Calculate match as 1 - sum of absolute differences divided by 2 (to normalize)
    diff_sum = sum(abs(required_fractions[i] - actual_fractions[i]) for i in range(len(required_fractions)))
    match = max(0.0, 1.0 - diff_sum / 2.0)
    return round(match * 100, 2)
