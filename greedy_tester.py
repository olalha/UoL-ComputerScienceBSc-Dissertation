
from _eval.rulebook_gen import generate_rulebook
from chunk_manager.chunk_partitioner import get_chunks
from dataset_manager.dataset_visualizer import plot_collection_distribution

from chunk_manager.greedy_solution import create_greedy_initial_solution
from chunk_manager.simulated_annealing import optimize_collections_with_simulated_annealing

# Rulebook generation parameters
RULEBOOK_PARAMS = {
        "mode": "word",
        "content_title": "EVAL - RULEBOOK 1",
        "total": 30000,
        "topics": [
            "Quality", "Price", "Design", "Performance", "Support",
            "Reliability", "Innovation", "Ergonomics", "Value", "Features",
            "Usability", "Compatibility", "Durability", "Flexibility", "Aesthetics",
        ],
        "topic_concentration": 2.0,
        "sentiment_concentration": 2.0,
        "chunk_size_avg": 60,
        "chunk_size_max_deviation": 20,
        "chunk_size_range_factor": 0.6,
        "collection_ranges_count": 3,
        "collection_ranges_max_val": 160,
        "collection_ranges_min_val": 100,
        "collection_distribution_concentration": 50.0,
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
    print(f"    Range: {rng['range']}, Target fraction: {rng['target_fraction']}")

# Convert chunks from list of dicts to list of tuples
chunks_tuples = []
for chunk in chunks:
    chunks_tuples.append(tuple(chunk.values()))
chunks = chunks_tuples

import time
start_time = time.time()

solution = create_greedy_initial_solution(
    chunks=chunks,
    size_ranges=size_ranges, 
    target_proportions=target_proportions, 
    mode=mode, 
    fill_factor=0.75
)

# from chunk_manager.solution_structure import SolutionStructure

# solution = SolutionStructure(
#     size_ranges=size_ranges,
#     target_proportions=target_proportions,
#     mode=mode
# )

# for chunk in chunks:
#     topic, sentiment, word_count = chunk
#     collection_idx = solution.create_new_collection()
#     solution.add_chunks_to_collection(collection_idx, [(topic, sentiment, word_count)])

end_time = time.time()
print(f"Greedy solution created in {end_time - start_time:.2f} seconds.")

final_chunks = sum(len(solution.get_all_chunks(idx)) for idx in solution.get_active_collection_indices())
if final_chunks != len(chunks):
    print("Warning: The number of chunks in the GREEDY solution does not match the original chunks.")
    print(f"Original chunks: {len(chunks)}, Greedy chunks: {final_chunks}")
else:
    print("The number of chunks in the GREEDY solution matches the original chunks.")
    
solution.visualize_solution()

start_time = time.time()
print("Optimizing collections with simulated annealing...")

solution = optimize_collections_with_simulated_annealing(solution)

end_time = time.time()
print(f"Simulated annealing optimization completed in {end_time - start_time:.2f} seconds.")

final_chunks = sum(len(solution.get_all_chunks(idx)) for idx in solution.get_active_collection_indices())
if final_chunks != len(chunks):
    print("Warning: The number of chunks in the OPTIMIZED solution does not match the original chunks.")
    print(f"Original chunks: {len(chunks)}, Optimized chunks: {final_chunks}")
else:
    print("The number of chunks in the OPTIMIZED solution matches the original chunks.")
    
final_cost = solution.get_total_absolute_deviation()
print(f"Final cost of the solution: {final_cost:.4f}")
deviation = solution.get_size_range_deviation()
for idx, dev in deviation:
    print(f"Range {idx} Deviation: {abs(dev):.2f}")

solution.visualize_solution()