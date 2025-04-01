
from _eval.rulebook_gen import generate_rulebook
from chunk_manager.chunk_partitioner import get_chunks
from dataset_manager.dataset_visualizer import plot_collection_distribution

from chunk_manager.greedy_solution import create_greedy_initial_solution

# Rulebook generation parameters
RULEBOOK_PARAMS = {
        "mode": "word",
        "content_title": "EVAL - RULEBOOK 1",
        "total": 20000,
        "topics": [
            "Quality", "Price", "Design", "Performance", "Support",
            "Reliability", "Innovation", "Ergonomics", "Value", "Features",
        ],
        "topic_concentration": 2.0,
        "sentiment_concentration": 2.0,
        "chunk_size_avg": 60,
        "chunk_size_max_deviation": 20,
        "chunk_size_range_factor": 0.6,
        "collection_ranges_count": 3,
        "collection_ranges_max_val": 180,
        "collection_ranges_min_val": 120,
        "collection_distribution_concentration": 4.0,
        "random_seed": 1234
}

rulebook = generate_rulebook(**RULEBOOK_PARAMS)
if not rulebook:
    raise ValueError("Rulebook generation failed. Please check the parameters.")

chunks = get_chunks(rulebook)
if not chunks:
    raise ValueError("No chunks generated. Please check the rulebook parameters.")

mode = rulebook["collection_mode"]

size_ranges = []
target_proportions = []
for rng in rulebook["collection_ranges"]:
    size_ranges.append(rng["range"])
    target_proportions.append(rng["target_fraction"])
    
print("SIZE RANGES")
for i in range(len(size_ranges)):
    print(f"Range {i}: {size_ranges[i]} - {target_proportions[i]}")

# Convert chunks from list of dicts to list of tuples
chunks_tuples = []
for chunk in chunks:
    chunks_tuples.append(tuple(chunk.values()))
chunks = chunks_tuples

solution = create_greedy_initial_solution(
    chunks=chunks,
    size_ranges=size_ranges, 
    target_proportions=target_proportions, 
    mode=mode, 
    fill_factor=0.8
)

# Convert solution to dataset format for visualization
collections = []
for coll_idx in solution.get_active_collection_indices():
    collection = {'chunks': [], 'collection_text': None}
    for chunk in solution.get_all_chunks(coll_idx):
        chunk_dict = {
            'topic': chunk[0],
            'sentiment': chunk[1],
            'wc': chunk[2],
        }
        collection['chunks'].append({'chunk_dict': chunk_dict, 'chunk_text': None})
    collections.append(collection)

dataset = {'content_title': rulebook['content_title'], 'collections': collections}

plot = plot_collection_distribution(dataset, mode)
plot.savefig("greedy_solution_distribution.png")
