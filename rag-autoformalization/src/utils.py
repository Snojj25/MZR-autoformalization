"""
Utility functions for logging, metrics, and visualization
"""
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from config import config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def save_results(results: List[Dict], filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    filepath = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate metrics from results"""
    total = len(results)
    
    # Compilation metrics
    compilation_success = sum(1 for r in results if r.get("compilation_success"))
    compilation_rate = compilation_success / total if total > 0 else 0
    
    # Proof metrics
    proof_success = sum(1 for r in results if r.get("proof_success"))
    proof_rate = proof_success / total if total > 0 else 0
    
    # Iteration metrics
    iterations_list = [r.get("total_iterations", 0) for r in results if r.get("compilation_success")]
    avg_iterations = sum(iterations_list) / len(iterations_list) if iterations_list else 0
    
    # Time metrics
    times = [r.get("total_time", 0) for r in results]
    avg_time = sum(times) / len(times) if times else 0
    
    # First-attempt success
    first_attempt_success = sum(
        1 for r in results 
        if r.get("compilation_success") and r.get("total_iterations") == 1
    )
    first_attempt_rate = first_attempt_success / total if total > 0 else 0
    
    metrics = {
        "total_problems": total,
        "compilation_success_rate": compilation_rate,
        "proof_success_rate": proof_rate,
        "first_attempt_success_rate": first_attempt_rate,
        "avg_iterations_to_success": avg_iterations,
        "avg_time_per_problem": avg_time,
        "metrics_by_iteration": _calculate_by_iteration(results)
    }
    
    return metrics

def _calculate_by_iteration(results: List[Dict]) -> Dict:
    """Calculate success rate by iteration number"""
    by_iteration = {}
    for i in range(1, config.MAX_ITERATIONS + 1):
        success_at_i = sum(
            1 for r in results 
            if r.get("compilation_success") and r.get("total_iterations") == i
        )
        by_iteration[f"iteration_{i}"] = success_at_i
    return by_iteration

def print_summary(results: List[Dict]):
    """Print a formatted summary of results using rich"""
    console = Console()
    metrics = calculate_metrics(results)
    
    # Main metrics table
    summary_table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="white", justify="right", width=20)
    
    summary_table.add_row("Total Problems", str(metrics['total_problems']))
    summary_table.add_row(
        "Compilation Success Rate",
        f"[green]{metrics['compilation_success_rate']:.1%}[/green]"
    )
    summary_table.add_row(
        "Proof Success Rate",
        f"[green]{metrics['proof_success_rate']:.1%}[/green]"
    )
    summary_table.add_row(
        "First-Attempt Success Rate",
        f"[green]{metrics['first_attempt_success_rate']:.1%}[/green]"
    )
    summary_table.add_row(
        "Avg Iterations (when successful)",
        f"{metrics['avg_iterations_to_success']:.2f}"
    )
    summary_table.add_row(
        "Avg Time per Problem",
        f"{metrics['avg_time_per_problem']:.2f}s"
    )
    
    console.print()
    console.print(summary_table)
    
    # Success by iteration table
    if metrics['metrics_by_iteration']:
        iteration_table = Table(title="Success by Iteration", show_header=True, header_style="bold blue")
        iteration_table.add_column("Iteration", style="cyan", justify="center", width=15)
        iteration_table.add_column("Problems Solved", style="green", justify="right", width=20)
        
        for k, v in metrics['metrics_by_iteration'].items():
            iter_num = k.split('_')[1]
            iteration_table.add_row(f"Iteration {iter_num}", str(v))
        
        console.print()
        console.print(iteration_table)
    
    console.print()

def create_results_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis"""
    rows = []
    for r in results:
        rows.append({
            "problem_id": r.get("problem_id"),
            "compilation_success": r.get("compilation_success"),
            "proof_success": r.get("proof_success"),
            "iterations": r.get("total_iterations"),
            "time": r.get("total_time"),
            "proof_tactic": r.get("proof_tactic")
        })
    return pd.DataFrame(rows)