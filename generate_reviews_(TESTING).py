
import os
import json
from pathlib import Path

from utils.api_request import prompt_llm_single, prompt_llm_parallel
from utils.settings_manager import get_setting
from prompt_manager.prompt_builder import render_prompt
from chunk_manager.topic_data_parser import parse_topic_sentiment_distribution_Excel
from chunk_manager.topic_chunk_splitter import get_chunks
from chunk_manager.chunk_group_allocator import allocate_chunks, visualize_chunk_allocation

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
    
    print("Parsing rulebook...")
    
    RULEBOOK_NAME = "TEMPLATE.xlsx"
    
    # Parse rulebook
    rulebook = parse_topic_sentiment_distribution_Excel(RULEBOOK_NAME)
    if not rulebook:
        raise ValueError(f"Error: Parsing rulebook {RULEBOOK_NAME} failed.")
    
    REVIEW_ITEM = rulebook['review_item']
    TOTAL_WC = rulebook['total_wc']
    
    print("Generating topic-sentiment chunks...")
    
    # Generate chunks
    all_chunks = []
    for topic_name, topic_dict in rulebook['content_rules'].items():
        # Get topic word count
        topic_wc = int(TOTAL_WC * topic_dict['total_proportion'])
        
        # Get sentiment word count
        for index, sentiment in enumerate(["positive", "neutral", "negative"]):
            topic_sentiment_wc = int(topic_wc * topic_dict['sentiment_proportion'][index])
            
            # Skip if no word count
            if topic_sentiment_wc == 0:
                continue
            
            # Partition topic-sentiment word count into chunks
            chunks = get_chunks(
                N=topic_sentiment_wc, 
                min_wc=topic_dict['chunk_min_wc'], 
                max_wc=topic_dict['chunk_max_wc'], 
                chunk_count_pref=topic_dict['chunk_count_pref'], 
                dirichlet_a=topic_dict['chunk_wc_distribution'])

            # Check if partitioning failed
            if not chunks:
                raise ValueError(f"Error: Partitioning fail - topic:'{topic_name}' sentiment:'{sentiment}' wc:{topic_sentiment_wc}.")
            
            # Add chunks to all_chunks if partitioning succeeded
            all_chunks.extend([{'topic': topic_name, 'sentiment': sentiment, 'wc': i} for i in chunks])
    
    print("Creating individual reviews...")
    
    # Define bucket ranges and target fractions
    BUCKETS = [
        {'range': (30, 100), 'target_fraction': 1},
    ]
    solution = allocate_chunks(all_chunks, BUCKETS, time_limit=30)
    
    # Check if a solution was found
    if not solution:
        raise ValueError("Error: Failed to allocate chunks to collections.")
    
    # Visualize chunk allocation
    visualize_chunk_allocation(solution)

    # """ Review text generation """
    
    # print("Generating text for selected test reviews...")
    
    # # Select test reviews
    # selected_reviews = []
    # reviews_per_bucket = 1
    # bucket_counts = [0] * len(BUCKETS)
    # for i in solution:
    #     if i['bucket'] is not None and 0 <= i['bucket'] < len(BUCKETS): 
    #         if bucket_counts[i['bucket']] < reviews_per_bucket:
    #             bucket_counts[i['bucket']] += 1
    #             selected_reviews.append(i['chunks'])
                
    # # Get model string
    # model = get_setting('MODELS','GPT4o-mini')
    
    # # Render individual chunk prompts
    # selected_reviews_text_snippets = []
    # for review in selected_reviews:
        
    #     # Get messages for each chunk
    #     chunk_messages = []
    #     for chunk_dict in review:
    #         prompt_context = {
    #             'review_item': REVIEW_ITEM,
    #             'topic': chunk_dict['topic'],
    #             'sentiment': chunk_dict['sentiment'],
    #             'word_count': chunk_dict['wc']
    #         }
    #         prompt = render_prompt("usr_chunk_gen.html", prompt_context)
    #         messages = [{'role': 'user', 'content': prompt}]
    #         chunk_messages.append({'chunk_dict': chunk_dict, 'messages': messages})
        
    #     # Generate text for each chunk
    #     messages = [i['messages'] for i in chunk_messages]
    #     responses = prompt_llm_parallel(model=model, messages=messages)
        
    #     # Extract generated text
    #     review_text_snippets = []
    #     for r in responses:
    #         chunk_dict = chunk_messages[r['idx']]['chunk_dict']
    #         chunk_text = f"{chunk_dict['topic']} - {chunk_dict['sentiment']} - {chunk_dict['wc']} - \n"
    #         if r['success']:
    #             chunk_text += r['response']['choices'][0]['message']['content']
    #         else:
    #             chunk_text += "NOT GENERATED"
    #             print(f"Failed to generate chunk: {chunk_dict}")
    #         review_text_snippets.append(chunk_text)
    #     selected_reviews_text_snippets.append(review_text_snippets)
        
    # """
    # At this point we have a list of reviews, each containing a list of text snippets.
    # The text snippets need be combined to form the full review text with the prompt usr_review_gen.html.
    # """
        
    # # Print generated text
    # for idx, review_text_snippets in enumerate(selected_reviews_text_snippets):
    #     print(f"\nReview {idx+1}:")
    #     for snippet in review_text_snippets:
    #         print(f"\n{snippet}")
