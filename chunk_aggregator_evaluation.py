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
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import copy

# Import required modules from the project
from _eval.rulebook_gen import generate_rulebook
from chunk_manager.chunk_aggregator import aggregate_chunks
from chunk_manager.chunk_partitioner import get_chunks
from chunk_manager.rulebook_parser import validate_rulebook_values
from dataset_manager.dataset_structurer import create_dataset_structure, validate_dataset_values
from dataset_manager.dataset_analyser import get_basic_counts, get_collection_distribution

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Experiment configurations to test
INITIAL_SOLUTION_METHODS = ["simple", "greedy"]
COST_FUNCTIONS = ["simple", "enhanced"]
MOVE_SELECTORS = ["static", "adaptive"]
COOLING_RATES = [0.999, 0.990, 0.950]

# Test run parameters
NUM_RUNS_PER_CONFIG = 3  # Number of times to run each configuration
TIME_LIMIT_PER_RUN = 10  # Maximum time (seconds) for each run
MAX_ITERATIONS = None    # Maximum iterations (None for no limit)
RECORD_ITERATION_INTERVAL = 10  # Record data every N iterations

# Multiprocessing settings
USE_MULTIPROCESSING = True  # Set to False to disable multiprocessing
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Use all but one CPU core

# Rulebook generation parameters
RULEBOOK_PARAMS = [
    # Small rulebook - few topics, balanced distribution
    {
        "mode": "word",
        "content_title": "Small Balanced Review",
        "total": 5000,
        "topics": ["Quality", "Price", "Design", "Performance", "Support"],
        "topic_concentration": 5.0,  # Balanced
        "sentiment_concentration": 3.0,
        "chunk_size_avg": 60,
        "collection_ranges_count": 4,
        "random_seed": 42
    },
    # Medium rulebook - more topics, skewed distribution
    {
        "mode": "word",
        "content_title": "Medium Skewed Review",
        "total": 15000,
        "topics": [
            "Quality", "Price", "Design", "Performance", "Support",
            "Reliability", "Innovation", "Ergonomics", "Value", "Features"
        ],
        "topic_concentration": 1.5,  # Skewed
        "sentiment_concentration": 2.0,
        "chunk_size_avg": 75,
        "collection_ranges_count": 5,
        "random_seed": 43
    }
]

# Visualization options
CREATE_VISUALIZATIONS = True
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    return (f"{config['initial_solution']}-{config['cost_function']}-"
            f"{config['move_selector']}-{config['cooling_rate']:.3f}")

def get_all_configurations() -> List[Dict[str, Any]]:
    """Generate all combinations of configuration parameters."""
    configs = []
    
    for initial_solution in INITIAL_SOLUTION_METHODS:
        for cost_function in COST_FUNCTIONS:
            for move_selector in MOVE_SELECTORS:
                for cooling_rate in COOLING_RATES:
                    configs.append({
                        'initial_solution': initial_solution,
                        'cost_function': cost_function,
                        'move_selector': move_selector,
                        'cooling_rate': cooling_rate
                    })
    
    return configs

def calculate_distribution_match(state: List[Dict[str, Any]], 
                               size_ranges: List[Dict[str, Any]]) -> float:
    """
    Calculate how well the solution matches the target distribution.
    Returns a normalized score between 0 and 1, where 1 is perfect match.
    """
    total_collections = len(state)
    if total_collections == 0:
        return 0.0
    
    # Count collections in each category and those out of range
    category_counts = {}
    out_of_range_count = 0
    
    for collection in state:
        category = collection.get('size_category')
        if category is not None:
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            out_of_range_count += 1
    
    # Apply penalty for out-of-range collections
    out_of_range_fraction = out_of_range_count / total_collections
    out_of_range_penalty = out_of_range_fraction  # Linear penalty
    
    # Calculate squared differences from target for in-range collections
    in_range_collections = total_collections - out_of_range_count
    squared_diff_sum = 0.0
    
    if in_range_collections > 0:
        for i, size_range in enumerate(size_ranges):
            target_fraction = size_range['target_fraction']
            # Calculate actual fraction relative to total collections
            actual_fraction = category_counts.get(i, 0) / total_collections
            squared_diff_sum += (actual_fraction - target_fraction) ** 2
    else:
        # If all collections are out of range, maximum distribution error
        squared_diff_sum = 1.0
    
    # Convert to a match percentage (1 - normalized_error)
    # Include both distribution mismatch and out-of-range penalty
    distribution_error = min(1.0, squared_diff_sum / 2.0)
    match_score = 1.0 - (distribution_error * (1 - out_of_range_penalty) + out_of_range_penalty)
    
    return max(0.0, min(1.0, match_score))  # Ensure result is between 0 and 1

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
    cost_function = config['cost_function']
    move_selector = config['move_selector']
    cooling_rate = config['cooling_rate']
    
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
    
    # Set hooks to collect intermediate data
    accepted_moves = 0
    rejected_moves = 0
    iterations_data = []
    
    # Custom callback to track progress
    def sa_callback(iteration, T, current_cost, best_cost, current_state, accepted):
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
                'num_collections': len(current_state)
            })
    
    # Run the aggregation algorithm
    try:
        solution = aggregate_chunks(
            chunks=chunks,
            size_ranges=rulebook['collection_ranges'],
            collection_mode=rulebook['collection_mode'],
            initial_solution_fn=initial_solution,
            cost_function=cost_function,
            move_selector=move_selector,
            cooling_rate=cooling_rate,
            time_limit=TIME_LIMIT_PER_RUN,
            max_iter=MAX_ITERATIONS,
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
            
        # Calculate final cost based on specified cost function
        if cost_function == "simple":
            from chunk_manager.chunk_aggregator import compute_cost_simple
            final_cost = compute_cost_simple(solution, rulebook['collection_ranges'])
        else:
            from chunk_manager.chunk_aggregator import compute_cost_enhanced
            final_cost = compute_cost_enhanced(solution, rulebook['collection_ranges'])
            
        # Calculate distribution match quality
        distribution_match = calculate_distribution_match(solution, rulebook['collection_ranges'])
        
        # Calculate additional metrics
        num_collections = len(solution)
        
        # Check if any collections have topics that violate the hard constraint
        topic_violations = 0
        for coll in solution:
            topics = [chunk['topic'] for chunk in coll['chunks']]
            if len(topics) != len(set(topics)):
                topic_violations += 1
                
        # Calculate average collection size
        if num_collections > 0:
            if rulebook['collection_mode'] == 'word':
                sizes = [sum(c['wc'] for c in coll['chunks']) for coll in solution]
            else:  # chunk mode
                sizes = [len(coll['chunks']) for coll in solution]
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
            'final_cost': final_cost,
            'num_collections': num_collections,
            'distribution_match': distribution_match,
            'execution_time': execution_time,
            'accepted_moves': accepted_moves,
            'rejected_moves': rejected_moves,
            'total_moves': accepted_moves + rejected_moves,
            'acceptance_ratio': accepted_moves / (accepted_moves + rejected_moves) if (accepted_moves + rejected_moves) > 0 else 0,
            'topic_violations': topic_violations,
            'avg_collection_size': avg_collection_size,
            'size_std_dev': size_std_dev,
            'iterations_data': iterations_data,
            'solution': solution  # For detailed analysis if needed
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
          f"({len(configs)} configs × {len(rulebooks)} rulebooks × {NUM_RUNS_PER_CONFIG} runs each)")
    
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
            avg_cost = np.mean([r['final_cost'] for r in config_rb_runs])
            std_cost = np.std([r['final_cost'] for r in config_rb_runs])
            avg_time = np.mean([r['execution_time'] for r in config_rb_runs])
            avg_collections = np.mean([r['num_collections'] for r in config_rb_runs])
            avg_dist_match = np.mean([r['distribution_match'] for r in config_rb_runs])
            avg_acceptance_ratio = np.mean([r['acceptance_ratio'] for r in config_rb_runs])
            topic_violations = sum(r.get('topic_violations', 0) for r in config_rb_runs)
            
            # Add summary to list
            summaries.append({
                'configuration': config_name,
                'initial_solution': config['initial_solution'],
                'cost_function': config['cost_function'],
                'move_selector': config['move_selector'],
                'cooling_rate': config['cooling_rate'],
                'rulebook': rulebook_title,
                'num_runs': num_runs,
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'avg_execution_time': avg_time,
                'avg_collections': avg_collections,
                'avg_distribution_match': avg_dist_match,
                'avg_acceptance_ratio': avg_acceptance_ratio,
                'total_topic_violations': topic_violations
            })
    
    return pd.DataFrame(summaries)

def analyze_convergence_patterns(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Analyze convergence patterns from iteration data.
    """
    successful_runs = [r for r in results if r['success'] and 'iterations_data' in r]
    if not successful_runs:
        return {}
        
    convergence_data = {}
    
    for run in successful_runs:
        if not run['iterations_data']:
            continue
            
        config_name = get_configuration_name(run['config'])
        key = f"{config_name}_{run['rulebook_title']}_{run['run_index']}"
        
        # Convert iteration data to DataFrame
        df = pd.DataFrame(run['iterations_data'])
        df['config'] = config_name
        df['rulebook'] = run['rulebook_title']
        df['run'] = run['run_index']
        
        convergence_data[key] = df
    
    return convergence_data

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(summary_df: pd.DataFrame, 
                         convergence_data: Dict[str, pd.DataFrame],
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
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Bar chart comparing performance metrics across configurations
    create_performance_comparison_plot(summary_df, viz_path)
    
    # 2. Line plots showing convergence patterns for best configurations
    if convergence_data:
        create_convergence_plots(convergence_data, viz_path)
    
    # 3. Box plots for statistical comparisons
    create_statistical_comparison_plots(summary_df, viz_path)
    
    # 4. Correlation heatmap between metrics
    create_correlation_heatmap(summary_df, viz_path)
    
    # 5. Parameter sensitivity analysis
    create_parameter_sensitivity_plots(summary_df, viz_path)
    
    # 6. Distribution match visualizations
    create_distribution_match_plots(results, viz_path)

def save_figure(fig, filename, path, formats):
    """Save figure in multiple formats."""
    for fmt in formats:
        full_path = os.path.join(path, f"{filename}.{fmt}")
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

def create_performance_comparison_plot(df: pd.DataFrame, output_path: str):
    """Create bar charts comparing performance metrics across configurations."""
    # Get top 10 configurations by average distribution match
    top_configs = df.sort_values('avg_distribution_match', ascending=False).head(10)
    
    # Create plot with multiple metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison of Top Configurations', fontsize=16)
    
    # Plot average distribution match
    sns.barplot(data=top_configs, x='configuration', y='avg_distribution_match', ax=axes[0, 0])
    axes[0, 0].set_title('Average Distribution Match (higher is better)')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    axes[0, 0].set_ylim(0, 1)
    
    # Plot average cost
    sns.barplot(data=top_configs, x='configuration', y='avg_cost', ax=axes[0, 1])
    axes[0, 1].set_title('Average Cost (lower is better)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot average execution time
    sns.barplot(data=top_configs, x='configuration', y='avg_execution_time', ax=axes[1, 0])
    axes[1, 0].set_title('Average Execution Time (seconds)')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot average number of collections
    sns.barplot(data=top_configs, x='configuration', y='avg_collections', ax=axes[1, 1])
    axes[1, 1].set_title('Average Number of Collections')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_figure(fig, "performance_comparison", output_path, VISUALIZATION_FORMATS)

def create_convergence_plots(convergence_data: Dict[str, pd.DataFrame], output_path: str):
    """Create line plots showing convergence patterns."""
    # Combine all dataframes
    combined_df = pd.concat(convergence_data.values(), ignore_index=True)
    
    # Get unique configurations and rulebooks
    configs = combined_df['config'].unique()
    rulebooks = combined_df['rulebook'].unique()
    
    for rulebook in rulebooks:
        fig, ax = plt.subplots(figsize=(12, 8))
        rulebook_data = combined_df[combined_df['rulebook'] == rulebook]
        
        for config in configs:
            config_data = rulebook_data[rulebook_data['config'] == config]
            
            if not config_data.empty:
                # Group by iteration and average across runs
                avg_data = config_data.groupby('iteration').agg({
                    'current_cost': 'mean',
                    'best_cost': 'mean'
                }).reset_index()
                
                # Plot best cost over iterations
                ax.plot(avg_data['iteration'], avg_data['best_cost'], 
                       label=config, linewidth=2, alpha=0.8)
        
        ax.set_title(f'Convergence Pattern - {rulebook}', fontsize=14)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Cost', fontsize=12)
        ax.legend(title='Configuration')
        ax.grid(True)
        
        # Use log scale for y-axis if values vary by orders of magnitude
        if ax.get_ylim()[1] / max(1e-10, ax.get_ylim()[0]) > 100:
            ax.set_yscale('log')
            
        plt.tight_layout()
        save_figure(fig, f"convergence_{rulebook.replace(' ', '_')}", 
                   output_path, VISUALIZATION_FORMATS)

def create_statistical_comparison_plots(df: pd.DataFrame, output_path: str):
    """Create box plots for statistical comparisons."""
    # Reorganize data for factor comparison
    factors = {
        'initial_solution': 'Initial Solution Method', 
        'cost_function': 'Cost Function',
        'move_selector': 'Move Selector', 
        'cooling_rate': 'Cooling Rate'
    }
    
    for factor, title in factors.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plot
        sns.boxplot(data=df, x=factor, y='avg_distribution_match', ax=ax)
        ax.set_title(f'Impact of {title} on Distribution Match', fontsize=14)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Distribution Match Quality', fontsize=12)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        save_figure(fig, f"boxplot_{factor}", output_path, VISUALIZATION_FORMATS)
        
        # Also create box plot for execution time
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df, x=factor, y='avg_execution_time', ax=ax)
        ax.set_title(f'Impact of {title} on Execution Time', fontsize=14)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        
        plt.tight_layout()
        save_figure(fig, f"boxplot_time_{factor}", output_path, VISUALIZATION_FORMATS)

def create_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Create a correlation heatmap between metrics."""
    # Select numerical columns only
    numeric_cols = [
        'avg_cost', 'avg_execution_time', 'avg_collections', 
        'avg_distribution_match', 'avg_acceptance_ratio'
    ]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Between Evaluation Metrics', fontsize=14)
    
    plt.tight_layout()
    save_figure(fig, "correlation_heatmap", output_path, VISUALIZATION_FORMATS)

def create_parameter_sensitivity_plots(df: pd.DataFrame, output_path: str):
    """Create heat maps for parameter sensitivity analysis."""
    # For each rulebook, create a parameter sensitivity plot
    for rulebook in df['rulebook'].unique():
        rulebook_df = df[df['rulebook'] == rulebook]
        
        # Create pivot tables for each pair of parameters
        param_pairs = [
            ('initial_solution', 'cost_function'),
            ('move_selector', 'cooling_rate')
        ]
        
        for x_param, y_param in param_pairs:
            # Create pivot table
            pivot = rulebook_df.pivot_table(
                index=y_param, 
                columns=x_param, 
                values='avg_distribution_match', 
                aggfunc='mean'
            )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f', ax=ax)
            ax.set_title(f'Parameter Sensitivity: {rulebook}', fontsize=14)
            
            plt.tight_layout()
            filename = f"sensitivity_{rulebook.replace(' ', '_')}_{x_param}_{y_param}"
            save_figure(fig, filename, output_path, VISUALIZATION_FORMATS)

def create_distribution_match_plots(results: List[Dict[str, Any]], output_path: str):
    """Create visualizations comparing target vs. actual distributions."""
    # Filter for successful runs with solutions
    successful_runs = [r for r in results if r['success'] and 'solution' in r]
    
    if not successful_runs:
        return
    
    # Pick a subset of representative runs to visualize
    for run in successful_runs[:5]:  # Just the first 5 successful runs
        config_name = get_configuration_name(run['config'])
        rulebook_title = run['rulebook_title']
        
        # Calculate actual vs target distribution
        solution = run['solution']
        size_ranges = run.get('rulebook_size_ranges', None)
        
        if not size_ranges:
            # This is a fallback if we don't have the original rulebook size ranges
            continue
            
        # Count distributions
        total_collections = len(solution)
        if total_collections == 0:
            continue
            
        actual_counts = {}
        for collection in solution:
            category = collection.get('size_category')
            if category is not None:
                actual_counts[category] = actual_counts.get(category, 0) + 1
        
        # Create comparison dataframe
        comparison_data = []
        for i, size_range in enumerate(size_ranges):
            target_fraction = size_range['target_fraction']
            actual_fraction = actual_counts.get(i, 0) / total_collections
            range_label = f"{size_range['range'][0]}-{size_range['range'][1]}"
            
            comparison_data.append({
                'Range': range_label,
                'Target': target_fraction,
                'Actual': actual_fraction
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        comparison_df.plot(x='Range', y=['Target', 'Actual'], kind='bar', ax=ax)
        
        ax.set_title(f'Distribution Match: {rulebook_title}\nConfig: {config_name}', fontsize=14)
        ax.set_xlabel('Collection Size Range', fontsize=12)
        ax.set_ylabel('Fraction of Collections', fontsize=12)
        ax.set_ylim(0, max(1.0, comparison_df['Target'].max() * 1.5, comparison_df['Actual'].max() * 1.5))
        ax.legend(title='Distribution')
        
        plt.tight_layout()
        filename = f"distribution_{rulebook_title.replace(' ', '_')}_{config_name}"
        save_figure(fig, filename, output_path, VISUALIZATION_FORMATS)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(summary_df: pd.DataFrame, 
                  all_results: List[Dict[str, Any]], 
                  output_path: str):
    """Export results to CSV and JSON for further analysis."""
    if SAVE_RESULTS_TO_CSV and not summary_df.empty:
        summary_csv_path = os.path.join(output_path, "summary_results.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary results to: {summary_csv_path}")
    
    if SAVE_RESULTS_TO_JSON:
        # Export full results (without large solution structures)
        compact_results = []
        for result in all_results:
            # Create a copy without the solution structure
            compact = {k: v for k, v in result.items() if k != 'solution'}
            
            # Convert config dict to string for better readability in JSON
            if 'config' in compact:
                compact['config_str'] = get_configuration_name(compact['config'])
                
            compact_results.append(compact)
            
        json_path = os.path.join(output_path, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(compact_results, f, indent=2)
        print(f"Saved detailed results to: {json_path}")

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
    
    # Save configuration parameters
    config_summary = {
        "initial_solution_methods": INITIAL_SOLUTION_METHODS,
        "cost_functions": COST_FUNCTIONS,
        "move_selectors": MOVE_SELECTORS,
        "cooling_rates": COOLING_RATES,
        "num_runs_per_config": NUM_RUNS_PER_CONFIG,
        "time_limit_per_run": TIME_LIMIT_PER_RUN,
        "max_iterations": MAX_ITERATIONS,
        "use_multiprocessing": USE_MULTIPROCESSING,
        "rulebook_params": RULEBOOK_PARAMS
    }
    
    with open(os.path.join(output_path, "evaluation_config.json"), 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    # Generate rulebooks
    rulebooks = generate_test_rulebooks()
    if not rulebooks:
        print("No valid rulebooks were generated. Exiting.")
        return
        
    print(f"Successfully generated {len(rulebooks)} rulebooks")
    
    # Generate all configurations
    configurations = get_all_configurations()
    print(f"Testing {len(configurations)} different configurations")
    
    # Run all evaluations
    all_results = run_evaluations(configurations, rulebooks)
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    summary_df = calculate_summary_statistics(all_results)
    
    if summary_df.empty:
        print("No successful evaluations to analyze!")
        return
    
    # Print summary of best configurations
    print("\nTop 5 configurations by distribution match quality:")
    top_configs = summary_df.sort_values('avg_distribution_match', ascending=False).head(5)
    print(top_configs[['configuration', 'rulebook', 'avg_distribution_match', 
                      'avg_cost', 'avg_execution_time']].to_string(index=False))
    
    # Analyze convergence patterns
    convergence_data = analyze_convergence_patterns(all_results)
    
    # Create visualizations
    create_visualizations(summary_df, convergence_data, all_results, output_path)
    
    # Export results
    export_results(summary_df, all_results, output_path)
    
    print("\nEvaluation completed!")
    print(f"All results and visualizations saved to: {output_path}")

if __name__ == "__main__":
    main()