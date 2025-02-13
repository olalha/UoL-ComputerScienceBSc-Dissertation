
import os
import json
from pathlib import Path

from utils.api_request import prompt_llm_single, prompt_llm_parallel
from utils.settings_manager import get_setting
from text_processing.prompt_builder import render_prompt
from text_processing.chunk_manager.topic_data_parser import parse_topic_sentiment_distribution_Excel
from text_processing.chunk_manager.topic_chunk_splitter import get_chunks
from text_processing.chunk_manager.chunk_group_allocator import allocate_chunks, visualize_chunk_allocation

if __name__ == "__main__":
    
    """ Environment setup """
    # Get absolute path to .env file
    current_dir = Path(__file__).resolve().parent
    env_path = current_dir / '.env'

    # Load environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    
    """ Review content generation """
    
    TOTAL_WC = get_setting("TOTAL_WC")
    CHUNK_WC_MIN = get_setting("CHUNK_WC_MIN")
    CHUNK_WC_MAX = get_setting("CHUNK_WC_MAX")
    RULEBOOK_NAME = "TEMPLATE.xlsx"
    
    # Parse rulebook
    rulebook = parse_topic_sentiment_distribution_Excel(RULEBOOK_NAME)
    if not rulebook:
        raise ValueError(f"Error: Parsing rulebook {RULEBOOK_NAME} failed.")
    review_item = rulebook['review_item']
    
    # Generate chunks
    all_chunks = []
    for topic_name, topic_values in rulebook['content'].items():
        # Get topic word count
        topic_wc = int(TOTAL_WC * topic_values[0])
        
        # Get sentiment word count
        for index, sentiment in enumerate(["positive", "neutral", "negative"]):
            topic_sentiment_wc = int(topic_wc * topic_values[1][index])
            
            # Skip if no word count
            if topic_sentiment_wc == 0:
                continue
            chunks = get_chunks(topic_sentiment_wc, CHUNK_WC_MIN, CHUNK_WC_MAX, chunk_count_pref=0.2, dirichlet_a=5.0)

            # Check if partitioning failed
            if not chunks:
                raise ValueError(f"Error: Partitioning fail - topic:'{topic_name}' sentiment:'{sentiment}' wc:{topic_sentiment_wc}.")
            
            # Add chunks to all_chunks if partitioning succeeded
            all_chunks.extend([{'topic': topic_name, 'sentiment': sentiment, 'wc': i} for i in chunks])
        
    # Define bucket ranges and target fractions
    BUCKETS = [
        {'range': (20, 50), 'target_fraction': 0.33},
        {'range': (51, 150), 'target_fraction': 0.34},
        {'range': (151, 300), 'target_fraction': 0.33},
    ]
    solution = allocate_chunks(all_chunks, BUCKETS, time_limit=5, max_iter=100000)
    
    # Check if a solution was found
    if not solution:
        raise ValueError("Error: Failed to allocate chunks to collections.")

    """ Review text generation """
    
    # Select test reviews
    selected_reviews = []
    reviews_per_bucket = 1
    bucket_counts = [0] * len(BUCKETS)
    for i in solution:
        if i['bucket'] is not None and 0 <= i['bucket'] < len(BUCKETS): 
            if bucket_counts[i['bucket']] < reviews_per_bucket:
                bucket_counts[i['bucket']] += 1
                selected_reviews.append(i['chunks'])
                
    # Get model string
    model = get_setting('MODELS','GPT4o')
    
    # Render individual chunk prompts
    selected_reviews_text_snippets = []
    for review in selected_reviews:
        
        # Get messages for each chunk
        chunk_messages = []
        for chunk_dict in review:
            prompt_context = {
                'review_item': review_item,
                'topic': chunk_dict['topic'],
                'sentiment': chunk_dict['sentiment'],
                'word_count': chunk_dict['wc']
            }
            prompt = render_prompt("usr_chunk_gen.html", prompt_context)
            messages = [{'role': 'user', 'content': prompt}]
            chunk_messages.append({'chunk_dict': chunk_dict, 'messages': messages})
        
        # Generate text for each chunk
        messages = [i['messages'] for i in chunk_messages]
        responses = prompt_llm_parallel(model=model, messages=messages)
        
        # Extract generated text
        review_text_snippets = []
        for r in responses:
            chunk_dict = chunk_messages[r['idx']]['chunk_dict']
            chunk_text = f"{chunk_dict['topic']} - {chunk_dict['sentiment']} - {chunk_dict['wc']} - \n"
            if r['success']:
                chunk_text += r['response']['choices'][0]['message']['content']
            else:
                chunk_text += "NOT GENERATED"
                print(f"Failed to generate chunk: {chunk_dict}")
            review_text_snippets.append(chunk_text)
        selected_reviews_text_snippets.append(review_text_snippets)
        
    """
    At this point we have a list of reviews, each containing a list of text snippets.
    The text snippets need be combined to form the full review text with the prompt usr_review_gen.html.
    """
        
    # Print generated text
    for idx, review_text_snippets in enumerate(selected_reviews_text_snippets):
        print(f"\nReview {idx+1}:")
        for snippet in review_text_snippets:
            print(f"\n{snippet}")
