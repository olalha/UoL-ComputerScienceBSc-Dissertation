
from utils.settings_manager import get_setting
from text_processing.chunk_manager.topic_data_parser import parse_topic_sentiment_distribution_Excel
from text_processing.chunk_manager.topic_chunk_splitter import get_chunks
from text_processing.chunk_manager.chunk_group_allocator import allocate_chunks, visualize_chunk_allocation

if __name__ == "__main__":
    
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
        {'range': (20, 50), 'target_fraction': 0.66},
        {'range': (51, 200), 'target_fraction': 0.01},
        {'range': (201, 300), 'target_fraction': 0.11},
        {'range': (301, 400), 'target_fraction': 0.11},
        {'range': (401, 500), 'target_fraction': 0.11},
    ]
    solution = allocate_chunks(all_chunks, BUCKETS, time_limit=20, max_iter=100000)
    
    # Check if a solution was found
    if not solution:
        raise ValueError("Error: Faild to Allocating chunks to collections.")

    # Visualize collection allocation
    visualize_chunk_allocation(solution)
    