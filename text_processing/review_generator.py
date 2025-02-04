import math
import random

# -----------------------------
# Global constraints (examples)
# Adjust these as needed
topic_sentiment_chunk_wc_min = 20
topic_sentiment_chunk_wc_max = 100

total_review_wc_min = 50
total_review_wc_max = 300
# -----------------------------

def get_div_5(x: int) -> int:
    """
    Returns the closest nonzero multiple of 5 to x.
    If x <= 2, returns 5 (as per examples).
    Example:
      1002 -> 1000
      534  -> 535
      1    -> 5
      0    -> 5
    """
    if x <= 0:
        return 5  # nonzero minimum is 5
    
    remainder = x % 5
    lower = x - remainder
    upper = lower + 5
    
    # If remainder < 2.5, we go lower, otherwise upper
    # But if lower is 0, we choose 5.
    if remainder == 0:
        candidate = x
    elif remainder < 2.5:
        candidate = lower if lower > 0 else 5
    else:
        candidate = upper
    
    return max(candidate, 5)


def greedy_fill(total_word_count: int, rulebook: dict) -> list:
    """
    Greedy random fill of topic-sentiment word allocations.
    
    :param total_word_count: The approximate total words for the corpus.
    :param rulebook: Dictionary of { topic: (content_prop, pos_prop, neu_prop, neg_prop) }.
                    Each proportion in (0,1], sum(pos, neu, neg) = 1, and sum of content_prop across topics = 1.
    :return: A list of reviews, where each review is a list of (topic, sentiment, word_count).
             Example of a review: [("Delivery", "Neg", 55), ("Quality", "Pos", 80)]
    """
    # 1. Compute the word budget for each (topic, sentiment) pair
    #    e.g., if Quality=10% and Positive=50% => 5% of total words goes to (Quality, Positive)
    topic_sent_map = {}
    for topic, (topic_prop, pos_p, neu_p, neg_p) in rulebook.items():
        topic_total_words = topic_prop * total_word_count
        
        # Round each sub-proportion to a multiple of 5
        pos_words = get_div_5(int(math.floor(topic_total_words * pos_p)))
        neu_words = get_div_5(int(math.floor(topic_total_words * neu_p)))
        neg_words = get_div_5(int(math.floor(topic_total_words * neg_p)))
        
        topic_sent_map[(topic, "Pos")] = pos_words
        topic_sent_map[(topic, "Neu")] = neu_words
        topic_sent_map[(topic, "Neg")] = neg_words
        
    # 2. Convert them into a structure we can track leftover budgets
    #    leftover_map: (topic, sentiment) -> words left to allocate
    leftover_map = dict(topic_sent_map)
    
    # We'll store final reviews as a list of lists
    all_reviews = []
    
    # 3. Greedy approach: while there's leftover, create new reviews
    #    We randomly pick topic-sentiment pairs to fill a new review, chunk by chunk.
    
    # Helper to check if we can add this chunk to a given review
    def can_add_chunk(review, topic, chunk_size):
        # 1) no repeated topic in the same review
        if any(t == topic for (t, _, _) in review):
            return False
        # 2) total word count must be within [min, max]
        current_size = sum(wc for (_, _, wc) in review)
        if current_size + chunk_size > total_review_wc_max:
            return False
        return True
    
    # Step-by-step create new reviews until leftover is (mostly) allocated
    # We'll put a cap on how many new reviews we create in this loop to avoid infinite loops
    # in pathological edge cases.
    
    attempts = 0
    max_attempts = 2 * len(leftover_map)  # heuristic
    
    while any(leftover_map.values()) and attempts < max_attempts:
        attempts += 1
        
        # Start a new review
        new_review = []
        review_size = 0
        
        # We keep trying to add random chunks until we approach total_review_wc_max
        # or can't add anything else
        for _ in range(50):  # limit tries per review
            # Filter out (t,s) that still have leftover
            candidates = [(ts, leftover_map[ts]) for ts in leftover_map if leftover_map[ts] > topic_sentiment_chunk_wc_min]
            if not candidates:
                break
            
            # Shuffle candidates to randomize
            random.shuffle(candidates)
            
            allocated_any = False
            for (topic_s, leftover_amount) in candidates:
                (topic, sentiment) = topic_s
                
                # Generate a random chunk that is:
                #   - multiple of 5
                #   - >= topic_sentiment_chunk_wc_min
                #   - <= topic_sentiment_chunk_wc_max
                #   - <= leftover_amount
                #   - fits the current review
                possible_sizes = []
                
                # TODO: this can be precomputed for before the main loop
                for size_5 in range(
                    topic_sentiment_chunk_wc_min,
                    topic_sentiment_chunk_wc_max + 1,
                    5
                ):
                    if size_5 <= leftover_amount:
                        possible_sizes.append(size_5)
                
                if not possible_sizes:
                    # can't allocate from this leftover
                    continue
                
                random_size = random.choice(possible_sizes)
                
                # Check if we can add this chunk to new_review
                if can_add_chunk(new_review, topic, sentiment, random_size):
                    new_review.append((topic, sentiment, random_size))
                    leftover_map[(topic, sentiment)] -= random_size
                    review_size += random_size
                    allocated_any = True
                    break  # break out of the for-loop to pick a new chunk in the next iteration
            
            # If we failed to allocate any chunk in this pass, we stop adding to this review
            if not allocated_any:
                break
            
            # If we are past the minimum size, we reduce the chance of adding more chunks
            if review_size >= total_review_wc_min:
                # Calculate interval size
                interval = (total_review_wc_max - total_review_wc_min) / 5
                
                # Find which interval the current size falls into (0-4)
                current_interval = min(4, int((review_size - total_review_wc_min) / interval))
                
                # Probability increases by 0.2 for each interval (0.2 to 1.0)
                break_probability = (current_interval + 1) * 0.2
                
                if random.random() < break_probability:
                    break
        
        # Final check: if the new_review meets the minimum review size, accept it
        if review_size >= total_review_wc_min:
            all_reviews.append(new_review)
        else:
            # If we didn't meet the min size, try to push leftover back (undo)
            for (topic, sentiment, wc) in new_review:
                leftover_map[(topic, sentiment)] += wc
            # break or continue? We'll continue to see if the next iteration does better.
    
    # 4. Attempt leftover re-allocation to existing reviews
    #    We try to place leftover in reviews that already mention that (topic,sent) or can mention it
    for (topic, sentiment), leftover_words in list(leftover_map.items()):
        if leftover_words <= 0:
            continue
        
        # TODO: For each leftover (topic, sentiment) go through all the reviews that contain it
        #       and try to allocate leftover_words to them in chunks of 5 while still keeping
        #       within max wc limits for reviews totals and chunks. Move on to the next candiate if
        #       we can't allocate any more to the current review. If we can't allocate all of the
        #       remaining word count to the existing reviews, print a message and move on.
        
        # We'll try to chunk leftover in steps of 5, or up to leftover_words, whichever is smaller
        while leftover_words > 0:
            chunk = min( leftover_words, topic_sentiment_chunk_wc_max )
            chunk = get_div_5(chunk)
            if chunk < topic_sentiment_chunk_wc_min:
                chunk = topic_sentiment_chunk_wc_min
            
            allocated_success = False
            for review in all_reviews:
                # Check if we can add chunk to the review
                # 1) If the review already has the same sentiment for the same topic => skip (only 1 mention of each topic per review).
                if any(r_topic == topic for (r_topic, r_s, r_wc) in review):
                    # This review already has that topic in some sentiment => skip
                    continue
                
                # 2) Check total word limit
                current_size = sum(r_wc for (_, _, r_wc) in review)
                if current_size + chunk <= total_review_wc_max:
                    # Good, we can add
                    review.append((topic, sentiment, chunk))
                    leftover_words -= chunk
                    allocated_success = True
                    break
            
            if not allocated_success:
                # We couldn't allocate chunk to any review
                # If chunk < min size, we skip to avoid infinite loop
                if chunk < topic_sentiment_chunk_wc_min:
                    break
                # else we skip to next chunk
                break
        
        leftover_map[(topic, sentiment)] = leftover_words
    
    # 5. Print leftover if any
    for (topic, sentiment), lw in leftover_map.items():
        if lw > 0:
            print(f"Could not allocate {lw} leftover words for ({topic}, {sentiment}).")
    
    return all_reviews


def verify_integrity(all_reviews: list, total_word_count: int, rulebook: dict) -> bool:
    """
    Checks:
      1) Each review's total word count is within [total_review_wc_min, total_review_wc_max].
      2) Each topic-sentiment chunk is within [topic_sentiment_chunk_wc_min, topic_sentiment_chunk_wc_max].
      3) No review mentions the same topic more than once.
      4) The total topic-sentiment word counts adhere to the rulebook proportions (± some tolerance).
         Because we're rounding to multiples of 5, we'll compare exact sums to the expected budgets.
    Returns True if all conditions are met, else False.
    """
    
    # 1) Review-level checks
    for review in all_reviews:
        review_size = sum(x[2] for x in review)
        if not (total_review_wc_min <= review_size <= total_review_wc_max):
            print(f"Review size {review_size} out of bounds for review {review}.")
            return False
        
        # track topics to ensure we only mention each topic once
        seen_topics = set()
        for (topic, sentiment, wc) in review:
            if wc < topic_sentiment_chunk_wc_min or wc > topic_sentiment_chunk_wc_max:
                print(f"Chunk size {wc} out of bounds in {topic, sentiment}.")
                return False
            if topic in seen_topics:
                print(f"Topic {topic} mentioned more than once in the same review.")
                return False
            seen_topics.add(topic)
    
    # 2) Summation check for topic-sentiment distribution
    # Recompute the target budgets
    ts_target = {}
    for topic, (topic_prop, pos_p, neu_p, neg_p) in rulebook.items():
        topic_words = total_word_count * topic_prop
        ts_target[(topic, "Pos")] = topic_words * pos_p
        ts_target[(topic, "Neu")] = topic_words * neu_p
        ts_target[(topic, "Neg")] = topic_words * neg_p
    
    # Accumulate actual
    ts_actual = {}
    for review in all_reviews:
        for (topic, sentiment, wc) in review:
            ts_actual[(topic, sentiment)] = ts_actual.get((topic, sentiment), 0) + wc
    
    # Compare actual with target. Because we round to multiples of 5, expect some discrepancy
    # We'll define a tolerance, e.g. +/- 50 words or 5% of the target, whichever is bigger.
    for ts_pair, target_val in ts_target.items():
        actual_val = ts_actual.get(ts_pair, 0)
        # define tolerance
        tolerance = max(0.05 * target_val, 50)  # 5% or 50 words
        if abs(actual_val - target_val) > tolerance:
            print(f"For {ts_pair}, actual={actual_val} vs target={target_val} (tolerance={tolerance}).")
            return False
    
    return True


# -------------
# Example usage
if __name__ == "__main__":
    # Suppose we want ~10,000 words total
    total_words = 10000
    
    # Example rulebook
    # topic -> (content proportion, pos, neu, neg)
    # sum of content proportions = 1
    # sum(pos, neu, neg) = 1
    sample_rulebook = {
        "Delivery": (0.3, 0.3, 0.2, 0.5),   # 30% of total corpus to Delivery
        "Quality":  (0.2, 0.5, 0.25, 0.25),# 20% of total corpus to Quality
        "Price":    (0.2, 0.2, 0.5, 0.3),  # 20% of total corpus to Price
        "Support":  (0.3, 0.3, 0.3, 0.4)   # 30% of total corpus to Support
    }
    
    reviews = greedy_fill(total_words, sample_rulebook)
    valid = verify_integrity(reviews, total_words, sample_rulebook)
    
    print(f"Generated {len(reviews)} reviews. Valid? {valid}")
    # Optionally inspect the first few reviews
    for i, r in enumerate(reviews[:5]):
        print(f"Review #{i+1}: {r}")
