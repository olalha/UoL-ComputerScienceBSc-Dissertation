
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
    
    """ Review content generation """
    
    TOTAL_WC = get_setting("TOTAL_WC")
    CHUNK_WC_MIN = get_setting("CHUNK_WC_MIN")
    CHUNK_WC_MAX = get_setting("CHUNK_WC_MAX")
    
    RULEBOOK_NAME = "TEMPLATE.xlsx"
    
    topic_distribution_percentage = parse_topic_sentiment_distribution_Excel(RULEBOOK_NAME)
    if not topic_distribution_percentage:
        raise ValueError(f"Error: Parsing rulebook {RULEBOOK_NAME} failed.")
    
    all_chunks = []
    for topic_name, topic_values in topic_distribution_percentage.items():
        # Get topic word count
        topic_wc = int(TOTAL_WC * topic_values[0])
        
        # Get sentiment word count
        for index, sentiment in enumerate(["pos", "neu", "neg"]):
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
        if 0 <= i['bucket'] < len(BUCKETS) and bucket_counts[i['bucket']] < reviews_per_bucket:
            bucket_counts[i['bucket']] += 1
            selected_reviews.append(i['chunks'])
            
    # Get absolute path to .env file
    current_dir = Path(__file__).resolve().parent
    env_path = current_dir / '.env'

    # Load environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    
    # Render system prompt
    review_item = "Hotel"
    review_gen_sys_prompt = render_prompt(template_name="sys_review_gen.html", context={'review_item': review_item})
    
    # Render individual review prompts
    messages = []
    for review in selected_reviews:
        all_chunks_word_count = sum([i['wc'] for i in review])
        padding_word_count = round(max(10, 0.2 * all_chunks_word_count))
        total_review_word_count = all_chunks_word_count + padding_word_count
        context = { 
            'review_item': review_item,
            'total_word_count': total_review_word_count,
            'items': review,
            'padding_word_count': padding_word_count,
        }
        review_gen_usr_prompt = render_prompt(template_name="usr_review_gen.html", context=context)
        messages.append([
            {"role": "user", "content": review_gen_usr_prompt},
            {"role": "system", "content": review_gen_sys_prompt},
        ])
    
    # Generate reviews
    model = get_setting('MODELS','GENERATE-MINI')
    responses = prompt_llm_parallel(model=model, messages=messages)
    
    # Print responses
    for r in responses:
        if r['success']:
            print("Defined Chunks:")
            review = selected_reviews[r['prompt_idx']]
            for chunk in review:
                print(f"- {chunk['topic']} {chunk['sentiment']} {chunk['wc']}")
            chunks_total_wc = sum([i['wc'] for i in review])
            padding_word_count = round(max(10, 0.2 * chunks_total_wc))
            print(f"Expected Word Count: {chunks_total_wc + padding_word_count}\n")
            generated_review = r['response']['choices'][0]['message']['content']
            print(f"Generated Review ({len(generated_review.split())} words):")
            print(generated_review)
            print("\n---\n")
        else:
            print(f"Failed to generate review for prompt: {r['prompt_idx']}")
    