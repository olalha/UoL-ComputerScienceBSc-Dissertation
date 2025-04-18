import random
import copy
import numpy as np
import pytest

from chunk_manager import greedy_solution, simulated_annealing, chunk_partitioner
from input_manager import rulebook_generator

@pytest.fixture(autouse=True)
def test_setup_random_seed():
    random.seed(123)
    np.random.seed(123)

@pytest.fixture
def test_simple_rulebook():
    # Generate a simple rulebook for testing
    topics = ["A", "B", "C"]
    return rulebook_generator.generate_rulebook(
        mode="word",
        content_title="Test Content",
        total=2000,
        topics=topics,
        topic_concentration=5.0,
        sentiment_concentration=5.0,
        chunk_size_avg=50,
        chunk_size_max_deviation=10,
        chunk_size_range_factor=0.3,
        collection_ranges_count=3,
        collection_ranges_max_val=300,
        collection_ranges_min_val=50,
        collection_distribution_concentration=5.0,
        random_seed=123
    )

@pytest.fixture
def test_partitioned_chunks(test_simple_rulebook):
    # Partition the rulebook into chunks
    chunk_dicts = chunk_partitioner.get_chunks(test_simple_rulebook)
    # Convert to (topic, sentiment, word_count) tuples
    return [(c['topic'], c['sentiment'], c['wc']) for c in chunk_dicts]

@pytest.fixture
def test_size_ranges_and_targets(test_simple_rulebook):
    # Extract size ranges and targets from rulebook
    ranges = []
    targets = []
    for r in test_simple_rulebook['collection_ranges']:
        ranges.append([r['range'][0], r['range'][1]])
        targets.append(r['target_fraction'])
    return ranges, targets

def test_greedy_solution_integrity(test_partitioned_chunks, test_size_ranges_and_targets):
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    
    # Create a greedy initial solution
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    # Retrieve all chunks from the solution
    all_chunks = []
    for idx in solution.get_active_collection_indices():
        all_chunks.extend(solution.get_all_chunks(idx))
    
    # All chunks must still be present, none duplicated or missing
    input_set = set(test_partitioned_chunks)
    output_set = set(all_chunks)
    assert input_set == output_set
    assert len(test_partitioned_chunks) == len(all_chunks)

def test_simulated_annealing_runs(test_partitioned_chunks, test_size_ranges_and_targets):
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    
    # Create a greedy initial solution
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    # Run simulated annealing
    optimized = simulated_annealing.optimize_collections_with_simulated_annealing(solution)
    
    # Retrieve all chunks from the optimized solution
    all_chunks = []
    for idx in optimized.get_active_collection_indices():
        all_chunks.extend(optimized.get_all_chunks(idx))
    
    # All chunks must still be present, none duplicated or missing
    input_set = set(test_partitioned_chunks)
    output_set = set(all_chunks)
    assert input_set == output_set
    assert len(test_partitioned_chunks) == len(all_chunks)

def test_optimize_collections_with_simulated_annealing_param_check(test_partitioned_chunks, test_size_ranges_and_targets):
    from chunk_manager import simulated_annealing
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    # Invalid initial_solution
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        None, 100, 10, 0.9, 2, 0.0
    ) is None
    # Invalid max_iterations
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, -1, 10, 0.9, 2, 0.0
    ) is None
    # Invalid initial_temperature
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, 100, 0, 0.9, 2, 0.0
    ) is None
    # Invalid cooling_rate
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, 100, 10, 1.0, 2, 0.0
    ) is None
    # Invalid oor_penalty_factor
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, 100, 10, 0.9, -1, 0.0
    ) is None
    # Invalid selection_bias
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, 100, 10, 0.9, 2, -1
    ) is None
    # Invalid callback
    assert simulated_annealing.optimize_collections_with_simulated_annealing(
        solution, 100, 10, 0.9, 2, 0.0, callback=123
    ) is None

def test_create_greedy_initial_solution_param_check(test_partitioned_chunks, test_size_ranges_and_targets):
    from chunk_manager import greedy_solution
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    # Invalid chunks
    assert greedy_solution.create_greedy_initial_solution(
        "not a list", size_ranges, targets, mode, fill_factor
    ) is None
    # Invalid size_ranges
    assert greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, "not a list", targets, mode, fill_factor
    ) is None
    # Invalid target_proportions
    assert greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, [0.5], mode, fill_factor
    ) is None
    # Invalid mode
    assert greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, "invalid", fill_factor
    ) is None
    # Invalid fill_factor
    assert greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, -0.1
    ) is None

def test_transfer_chunk_move(test_partitioned_chunks, test_size_ranges_and_targets):
    from chunk_manager import greedy_solution, simulated_annealing
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    over = solution.get_overpopulated_ranges()
    under = solution.get_underpopulated_ranges()
    if over and under:
        before = copy.deepcopy(solution)
        success, move_info = simulated_annealing.transfer_chunk(solution, over, under, 0.0)
        assert isinstance(success, bool)
        if success:
            assert move_info is not None
            simulated_annealing.revert_move(solution, move_info)
            def get_collection_set(sol):
                return set(
                    frozenset(sol.get_all_chunks(idx))
                    for idx in sol.get_active_collection_indices()
                )
            assert get_collection_set(solution) == get_collection_set(before)
            assert solution.size_ranges == before.size_ranges
            assert solution.target_proportions == before.target_proportions

def test_swap_chunks_move(test_partitioned_chunks, test_size_ranges_and_targets):
    from chunk_manager import greedy_solution, simulated_annealing
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    over = solution.get_overpopulated_ranges()
    under = solution.get_underpopulated_ranges()
    if over and under:
        before = copy.deepcopy(solution)
        success, move_info = simulated_annealing.swap_chunks(solution, over, under, 0.0)
        assert isinstance(success, bool)
        if success:
            assert move_info is not None
            simulated_annealing.revert_move(solution, move_info)
            def get_collection_set(sol):
                return set(
                    frozenset(sol.get_all_chunks(idx))
                    for idx in sol.get_active_collection_indices()
                )
            assert get_collection_set(solution) == get_collection_set(before)
            assert solution.size_ranges == before.size_ranges
            assert solution.target_proportions == before.target_proportions

def test_split_collection_move(test_partitioned_chunks, test_size_ranges_and_targets):
    from chunk_manager import greedy_solution, simulated_annealing
    size_ranges, targets = test_size_ranges_and_targets
    mode = "word"
    fill_factor = 0.8
    solution = greedy_solution.create_greedy_initial_solution(
        test_partitioned_chunks, size_ranges, targets, mode, fill_factor
    )
    over = solution.get_overpopulated_ranges()
    under = solution.get_underpopulated_ranges()
    if over and under:
        before = copy.deepcopy(solution)
        success, move_info = simulated_annealing.split_collection(solution, over, under, 0.0)
        assert isinstance(success, bool)
        if success:
            assert move_info is not None
            simulated_annealing.revert_move(solution, move_info)
            def get_collection_set(sol):
                return set(
                    frozenset(sol.get_all_chunks(idx))
                    for idx in sol.get_active_collection_indices()
                )
            assert get_collection_set(solution) == get_collection_set(before)
            assert solution.size_ranges == before.size_ranges
            assert solution.target_proportions == before.target_proportions

def test_solution_structure_basic():
    from chunk_manager.solution_structure import SolutionStructure
    # Simple ranges and proportions
    size_ranges = [[10, 20], [21, 40]]
    target_proportions = [0.5, 0.5]
    sol = SolutionStructure(size_ranges, target_proportions, mode="word")
    idx1 = sol.create_new_collection()
    idx2 = sol.create_new_collection()
    # Add chunks
    assert sol.add_chunks_to_collection(idx1, [("A", "positive", 15)])
    assert sol.add_chunks_to_collection(idx2, [("B", "negative", 25)])
    # Can't add duplicate topic
    assert not sol.add_chunks_to_collection(idx1, [("A", "neutral", 5)])
    # Remove chunk
    assert sol.remove_chunks_from_collection(idx1, ["A"])
    # Remove collection
    assert sol.remove_collection(idx2)
    # Out of range collections
    idx3 = sol.create_new_collection()
    assert sol.add_chunks_to_collection(idx3, [("C", "neutral", 100)])
    out_of_range = sol.get_out_of_range_collections()
    assert idx3 in out_of_range
    # Distribution
    dist = sol.calculate_size_distribution()
    assert isinstance(dist, list)
