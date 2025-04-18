"""
Chunk Aggregator Evaluation Script

This script provides a comprehensive framework for evaluating and comparing
different configurations of the chunk aggregator algorithm.

It generates test rulebooks, runs multiple experiments with different configurations,
collects metrics, and visualizes the results for comparison.
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import copy

# Import required modules from the project
from input_manager.rulebook_generator import generate_rulebook
from chunk_manager.chunk_partitioner import get_chunks
from input_manager.rulebook_parser import validate_rulebook_values
from chunk_manager.solution_structure import SolutionStructure
from chunk_manager.greedy_solution import create_greedy_initial_solution
from chunk_manager.simulated_annealing import optimize_collections_with_simulated_annealing

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Experiment configurations to test
INITIAL_SOLUTION_METHODS = ["simple", "greedy"]
COOLING_RATES = [0.9, 0.95, 0.99, 0.999]
NUM_OF_ITERATIONS = [2000, 5000, 10000]
OOR_PENALTY_FACTOR = [0.0, 2.0, 4.0]
SELECTION_BIAS = [1.0, 2.0, 4.0]

# Test run parameters
NUM_RUNS_PER_CONFIG = 5  # Number of times to run each configuration
RECORD_ITERATION_INTERVAL = 10  # Record data every N iterations

# Multiprocessing settings
USE_MULTIPROCESSING = True  # Set to False to disable multiprocessing
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Use all but one CPU core

# Rulebook generation parameters
RULEBOOK_PARAMS = [
    {
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
    "collection_ranges_count": 4,
    "collection_ranges_min_val": 120,
    "collection_ranges_max_val": 200,
    "collection_distribution_concentration": 50.0,
    "random_seed": 1234
    }
]

# Visualization options
CREATE_VISUALIZATIONS = False
VISUALIZATION_FORMATS = ['png']  # Options: 'png', 'svg', 'pdf'
PLOT_DPI = 300

# Output options
OUTPUT_DIR = "_eval_results"
SAVE_RESULTS_TO_CSV = True
SAVE_RESULTS_TO_JSON = True
DETAILED_LOGGING = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_output_directory() -> str:
    """Create timestamp-based output directory for results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path

def generate_test_rulebooks() -> List[Dict[str, Any]]:
    """Generate rulebooks for testing based on configured parameters."""
    rulebooks = []
    
    print(f"Generating {len(RULEBOOK_PARAMS)} test rulebooks...")
    for params in RULEBOOK_PARAMS:
        try:
            # Create a deep copy to avoid modifying original
            rb_params = copy.deepcopy(params)
            rulebook = generate_rulebook(**rb_params)
            
            # Validate the generated rulebook
            if validate_rulebook_values(rulebook):
                rulebooks.append(rulebook)
                print(f"  ✓ Generated rulebook: {rulebook['content_title']}")
            else:
                print(f"  ✗ Invalid rulebook generated: {rb_params['content_title']}")
                
        except Exception as e:
            print(f"  ✗ Error generating rulebook: {str(e)}")
    
    return rulebooks

def get_configuration_name(config: Dict[str, Any]) -> str:
    """Generate a descriptive name for a configuration."""
    return (f"Init_sl-{config['initial_solution']} - "
            f"Cool_rt-{config['cooling_rate']:.3f} - "
            f"Oor_pn-{config['oor_penalty']:.1f} - "
            f"Sel_bs-{config['selection_bias']:.1f} - "
            f"Num_it-{config['num_iterations']:.0f}")

def get_all_configurations() -> List[Dict[str, Any]]:
    """Generate all combinations of configuration parameters."""
    configs = []

    for initial_solution in INITIAL_SOLUTION_METHODS:
        for cooling_rate in COOLING_RATES:
            for oor_penalty in OOR_PENALTY_FACTOR:
                for selection_bias in SELECTION_BIAS:
                    for num_iterations in NUM_OF_ITERATIONS:
                        configs.append({
                            'initial_solution': initial_solution,
                            'cooling_rate': cooling_rate,
                            'oor_penalty': oor_penalty,
                            'selection_bias': selection_bias,
                            'num_iterations': num_iterations
                        })
    return configs

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_single_run(config: Dict[str, Any], 
                       rulebook: Dict[str, Any], 
                       run_index: int) -> Dict[str, Any]:
    """
    Run a single evaluation with the given configuration and rulebook.
    """
    # Extract configuration parameters
    initial_solution = config['initial_solution']
    cooling_rate = config['cooling_rate']
    num_iterations = config['num_iterations']
    oor_penalty = config['oor_penalty']
    selection_bias = config['selection_bias']
    
    # Get size ranges and target proportions from rulebook
    size_ranges = []
    target_proportions = []
    for rng in rulebook["collection_ranges"]:
        size_ranges.append(rng["range"])
        target_proportions.append(rng["target_fraction"])
    
    # Set seed for reproducibility
    random_seed = 1000 + run_index
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Start timing
    start_time = time.time()
    
    # Get chunks from rulebook
    try:
        chunks = get_chunks(rulebook)
        if not chunks:
            return {
                'config': config,
                'rulebook_title': rulebook['content_title'],
                'run_index': run_index,
                'success': False,
                'error': "Failed to generate chunks from rulebook"
            }
    except Exception as e:
        return {
            'config': config,
            'rulebook_title': rulebook['content_title'],
            'run_index': run_index,
            'success': False,
            'error': f"Exception in chunk generation: {str(e)}"
        }
    
    # Convert chunks from list of dicts to list of tuples
    chunk_tuples = []
    for chunk in chunks:
        topic = chunk['topic']
        sentiment = chunk['sentiment']
        word_count = chunk['wc']
        chunk_tuples.append((topic, sentiment, word_count))
    
    # Create initial solution based on the selected method    
    try:
        if initial_solution == 'simple':
            # Simple initial solution - create a collection for each chunk
            solution = SolutionStructure(
                size_ranges=size_ranges,
                target_proportions=target_proportions,
                mode=rulebook['collection_mode']
            )
            for chunk in chunk_tuples:
                collection_idx = solution.create_new_collection()
                solution.add_chunks_to_collection(collection_idx, [chunk])
        
        elif initial_solution == 'greedy':
            # Greedy initial solution - use the greedy algorithm to create collections
            solution = create_greedy_initial_solution(
                chunks=chunk_tuples,
                size_ranges=size_ranges, 
                target_proportions=target_proportions, 
                mode=rulebook['collection_mode'], 
                fill_factor=0.75
            )
        else:
            raise ValueError(f"Unknown initial solution method: {initial_solution}")
    except Exception as e:
        return {
            'config': config,
            'rulebook_title': rulebook['content_title'],
            'run_index': run_index,
            'success': False,
            'error': f"Exception in chunk generation: {str(e)}"
        }
    
    # Set hooks to collect intermediate data
    accepted_moves = 0
    rejected_moves = 0
    iterations_data = []
    
    # Custom callback to track progress
    def sa_callback(iteration, T, current_cost, best_cost, num_collections, accepted):
        nonlocal accepted_moves, rejected_moves
        
        if accepted:
            accepted_moves += 1
        else:
            rejected_moves += 1
        
        # Record every n iterations
        if iteration % RECORD_ITERATION_INTERVAL == 0:
            iterations_data.append({
                'iteration': iteration,
                'temperature': T,
                'current_cost': current_cost,
                'best_cost': best_cost,
                'num_collections': num_collections
            })
    
    # Run the aggregation algorithm
    try:
        # Optimize collections using simulated annealing
        solution = optimize_collections_with_simulated_annealing(
            solution,
            max_iterations=num_iterations,
            cooling_rate=cooling_rate,
            oor_penalty_factor=oor_penalty,
            selection_bias=selection_bias,
            callback=sa_callback
        )
        
        # Measure execution time
        execution_time = time.time() - start_time
        
        if solution is None:
            return {
                'config': config,
                'rulebook_title': rulebook['content_title'],
                'run_index': run_index,
                'success': False,
                'error': "Aggregation algorithm returned None",
                'execution_time': execution_time,
                'iterations_data': iterations_data
            }
        
        # Calculate distribution match quality
        distribution_match = solution.get_total_absolute_deviation()
        
        # Calculate out-of-range fraction
        oor_faction = solution.get_out_of_range_collections_fraction()
        
        # Calculate additional metrics
        num_collections = len(solution.get_active_collection_indices())
                
        # Calculate average collection size
        if num_collections > 0:
            sizes = [solution.get_collection_size(idx) for idx in solution.get_active_collection_indices()]                
            avg_collection_size = sum(sizes) / len(sizes)
            size_std_dev = np.std(sizes) if len(sizes) > 1 else 0
        else:
            avg_collection_size = 0
            size_std_dev = 0
        
        return {
            'config': config,
            'rulebook_title': rulebook['content_title'],
            'run_index': run_index,
            'success': True,
            'num_collections': num_collections,
            'distribution_match': distribution_match,
            'oor_fraction': oor_faction,
            'execution_time': execution_time,
            'accepted_moves': accepted_moves,
            'rejected_moves': rejected_moves,
            'total_moves': accepted_moves + rejected_moves,
            'acceptance_ratio': accepted_moves / (accepted_moves + rejected_moves) if (accepted_moves + rejected_moves) > 0 else 0,
            'avg_collection_size': avg_collection_size,
            'size_std_dev': size_std_dev,
            'iterations_data': iterations_data,
            'solution': solution
        }
        
    except Exception as e:
        return {
            'config': config,
            'rulebook_title': rulebook['content_title'],
            'run_index': run_index,
            'success': False,
            'error': f"Exception in aggregation: {str(e)}",
            'traceback': traceback.format_exc(),
            'execution_time': time.time() - start_time,
            'iterations_data': iterations_data
        }

def worker_function(args):
    """Worker function for parallel processing."""
    config, rulebook, run_index = args
    return evaluate_single_run(config, rulebook, run_index)

def run_evaluations(configs: List[Dict[str, Any]], 
                   rulebooks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run evaluations for all configurations across all rulebooks and runs.
    """
    all_results = []
    
    # Calculate total number of evaluations
    total_evaluations = len(configs) * len(rulebooks) * NUM_RUNS_PER_CONFIG
    print(f"\nRunning {total_evaluations} total evaluations "
          f"({len(configs)} configs * {len(rulebooks)} rulebooks * {NUM_RUNS_PER_CONFIG} runs each)")
    
    # Create all evaluation tasks
    tasks = []
    for config in configs:
        for rulebook in rulebooks:
            for run in range(NUM_RUNS_PER_CONFIG):
                tasks.append((config, rulebook, run))
    
    # Run evaluations (parallel or sequential)
    if USE_MULTIPROCESSING and MAX_WORKERS > 1:
        print(f"Using multiprocessing with {MAX_WORKERS} workers")
        with mp.Pool(processes=MAX_WORKERS) as pool:
            results_iter = pool.imap_unordered(worker_function, tasks)
            all_results = list(tqdm(results_iter, total=len(tasks), 
                                   desc="Running evaluations"))
    else:
        print("Running evaluations sequentially")
        for task in tqdm(tasks, desc="Running evaluations"):
            result = worker_function(task)
            all_results.append(result)
    
    return all_results

# ============================================================================
# RESULTS ANALYSIS FUNCTIONS
# ============================================================================

def calculate_summary_statistics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate summary statistics for each configuration and rulebook.
    """
    # Filter for successful runs only
    successful_runs = [r for r in results if r['success']]
    
    if not successful_runs:
        print("No successful runs to analyze!")
        return pd.DataFrame()
    
    # Group by configuration and rulebook
    summaries = []
    
    # Get unique configs and rulebooks
    configs = {get_configuration_name(r['config']): r['config'] for r in successful_runs}
    rulebook_titles = sorted(list(set(r['rulebook_title'] for r in successful_runs)))
    
    for config_name, config in configs.items():
        for rulebook_title in rulebook_titles:
            # Filter runs for this config and rulebook
            config_rb_runs = [r for r in successful_runs 
                            if get_configuration_name(r['config']) == config_name 
                            and r['rulebook_title'] == rulebook_title]
            
            if not config_rb_runs:
                continue
                
            # Calculate statistics
            num_runs = len(config_rb_runs)
            avg_time = np.mean([r['execution_time'] for r in config_rb_runs])
            avg_collections = np.mean([r['num_collections'] for r in config_rb_runs])
            avg_dist_match = np.mean([r['distribution_match'] for r in config_rb_runs])
            avg_oor_fraction = np.mean([r['oor_fraction'] for r in config_rb_runs])
            avg_acceptance_ratio = np.mean([r['acceptance_ratio'] for r in config_rb_runs])
            
            # Add summary to list
            summaries.append({
                'configuration': config_name,
                'initial_solution': config['initial_solution'],
                'cooling_rate': config['cooling_rate'],
                'oor_penalty': config['oor_penalty'],
                'selection_bias': config['selection_bias'],
                'num_iterations': config['num_iterations'],
                'rulebook': rulebook_title,
                'num_runs': num_runs,
                'avg_execution_time': avg_time,
                'avg_collections': avg_collections,
                'avg_distribution_match': avg_dist_match,
                'avg_oor_fraction': avg_oor_fraction,
                'avg_acceptance_ratio': avg_acceptance_ratio,
            })
    
    return pd.DataFrame(summaries)

def get_best_run_for_config_rulebook(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get the best solution for each configuration and rulebook combination.
    """
    successful_runs = [r for r in results if r['success']]
    if not successful_runs:
        return {}
        
    best_runs = {}
    for run in successful_runs:
        config_name = get_configuration_name(run['config'])
        key = f"{config_name}_{run['rulebook_title']}"
        
        if key not in best_runs or run['distribution_match'] > best_runs[key]['distribution_match']:
            best_runs[key] = run
    
    return best_runs

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(summary_df: pd.DataFrame, 
                         results: List[Dict[str, Any]],
                         output_path: str):
    """
    Create visualizations to compare results across configurations.
    """
    if not CREATE_VISUALIZATIONS:
        return
        
    if summary_df.empty:
        print("No data available for visualization")
        return
        
    print("\nGenerating visualizations...")
    
    # Create visualizations directory
    viz_path = os.path.join(output_path, "visualizations")
    os.makedirs(viz_path, exist_ok=True)
    
    # Get best run for each config and rulebook combination
    best_runs = get_best_run_for_config_rulebook(results)
    
    if not best_runs:
        print("No successful runs to visualize")
        return
    
    # Create solution visualizations
    create_solution_plots(best_runs, viz_path)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create convergence plots
    create_convergence_plots(best_runs, viz_path)

def save_figure(fig, filename, path, formats):
    """Save figure in multiple formats."""
    for fmt in formats:
        full_path = os.path.join(path, f"{filename}.{fmt}")
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

def create_solution_plots(best_runs: Dict[str, Any], output_path: str):
    
    # Create a separate directory for solution visualizations
    solution_viz_path = os.path.join(output_path, "best_solutions_visualizations")
    os.makedirs(solution_viz_path, exist_ok=True)
    
    # Isolate the best solutions
    best_solutions = {k: v['solution'] for k, v in best_runs.items()}
    
    # Generate visualizations for best solutions
    for config, solution in best_solutions.items():
        fig, _ = solution.visualize_solution(title=f"Solution Visualization: {config}", show=False)
        save_figure(fig, f"solution_plot_{config}", solution_viz_path, VISUALIZATION_FORMATS)

def create_convergence_plots(best_runs: Dict[str, Any], output_path: str):
    """Create line plots showing convergence patterns."""

    # Create a separate directory for solution visualizations
    convergence_viz_path = os.path.join(output_path, "convergence_visualizations")
    os.makedirs(convergence_viz_path, exist_ok=True)
    
    for config, run in best_runs.items():
        iterations_data = run['iterations_data']
        if not iterations_data:
            continue
        
        # Extract data for plotting
        iterations = [d['iteration'] for d in iterations_data]
        costs = [d['current_cost'] for d in iterations_data]
        best_costs = [d['best_cost'] for d in iterations_data]
        temp = [d['temperature'] for d in iterations_data]
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        # First y-axis for costs
        ax1.plot(iterations, costs, label='Current Cost', color='blue', alpha=0.7)
        ax1.plot(iterations, best_costs, label='Best Cost', color='green', alpha=0.7, linestyle='--')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Second y-axis for temperature
        ax2 = ax1.twinx()
        ax2.plot(iterations, temp, label='Temperature', color='red', linestyle=':')
        ax2.set_ylabel("Temperature", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Title and legend
        ax1.set_title(f"Convergence Plot: {config}")
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Save figure
        save_figure(fig, f"convergence_plot_{config}", convergence_viz_path, VISUALIZATION_FORMATS)
    
# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(summary_df: pd.DataFrame, 
                  all_results: List[Dict[str, Any]], 
                  output_path: str):
    """Export results to CSV and JSON for further analysis."""
    
    try:
        if SAVE_RESULTS_TO_CSV and not summary_df.empty:
            summary_csv_path = os.path.join(output_path, "summary_results.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved summary results to: {summary_csv_path}")
    except Exception as e:
        print(f"Error saving summary results to CSV: {str(e)}")
    
    try:
        if SAVE_RESULTS_TO_JSON:
            # Export full results (without large iteration structures)
            compact_results = []
            for result in all_results:
                
                # Convert config dict to string for better readability in JSON
                compact = {"config_str": get_configuration_name(result['config'])}
                    
                # Create a copy without the iteration structure
                compact.update({k: v for k, v in result.items() if k != 'iterations_data' and k != 'solution'})
                compact_results.append(compact)
                
            json_path = os.path.join(output_path, "evaluation_results.json")
            with open(json_path, 'w') as f:
                json.dump(compact_results, f, indent=4)
            print(f"Saved detailed results to: {json_path}")
    except Exception as e:
        print(f"Error saving detailed results to JSON: {str(e)}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("CHUNK AGGREGATOR EVALUATION")
    print("=" * 80)
    
    # Create output directory
    output_path = setup_output_directory()
    print(f"\nResults will be saved to: {output_path}")
    
    # Generate rulebooks
    rulebooks = generate_test_rulebooks()
    if not rulebooks:
        print("No valid rulebooks were generated. Exiting.")
        return
    ALL_RULEBOOKS = {f"Rulebook_{i}": rb for i, rb in enumerate(rulebooks)}
        
    print(f"Successfully generated {len(rulebooks)} rulebooks")
    
    # Save configuration parameters
    config_summary = {
        "initial_solution_methods": INITIAL_SOLUTION_METHODS,
        "cooling_rates": COOLING_RATES,
        "num_of_iterations": NUM_OF_ITERATIONS,
        "oor_penalty_factors": OOR_PENALTY_FACTOR,
        "selection_biases": SELECTION_BIAS,
        "num_runs_per_config": NUM_RUNS_PER_CONFIG,
        "record_iteration_interval": RECORD_ITERATION_INTERVAL,
        "use_multiprocessing": USE_MULTIPROCESSING,
        "rulebook_params": RULEBOOK_PARAMS,
        "all_generated_rulebooks": ALL_RULEBOOKS
    }
    with open(os.path.join(output_path, "evaluation_config.json"), 'w') as f:
        json.dump(config_summary, f, indent=4)
    
    # Generate all configurations
    configurations = get_all_configurations()
    print(f"Testing {len(configurations)} different configurations")
    
    # Run all evaluations
    all_results = run_evaluations(configurations, rulebooks)
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    summary_df = calculate_summary_statistics(all_results)
    
    # Export results
    export_results(summary_df, all_results, output_path)
    
    if summary_df.empty:
        print("No successful evaluations to analyze!")
        return
    
    # Print summary of best configurations
    print("\nSorted configurations by distribution match quality:")
    sorted_configs = summary_df.sort_values('avg_distribution_match', ascending=False)
    print(sorted_configs[['configuration', 'rulebook', 'avg_distribution_match', 'avg_execution_time']].to_string(index=False))
    
    # Create visualizations
    create_visualizations(summary_df, all_results, output_path)
    
    print("\nEvaluation completed!")
    print(f"All results and visualizations saved to: {output_path}")

if __name__ == "__main__":
    main()