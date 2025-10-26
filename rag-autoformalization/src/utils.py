"""
Utility functions for logging, metrics, and visualization
"""
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from config import config

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
    """Print a formatted summary of results"""
    metrics = calculate_metrics(results)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Problems: {metrics['total_problems']}")
    print(f"\nCompilation Success Rate: {metrics['compilation_success_rate']:.1%}")
    print(f"Proof Success Rate: {metrics['proof_success_rate']:.1%}")
    print(f"First-Attempt Success Rate: {metrics['first_attempt_success_rate']:.1%}")
    print(f"\nAverage Iterations (when successful): {metrics['avg_iterations_to_success']:.2f}")
    print(f"Average Time per Problem: {metrics['avg_time_per_problem']:.2f}s")
    
    print(f"\nSuccess by Iteration:")
    for k, v in metrics['metrics_by_iteration'].items():
        iter_num = k.split('_')[1]
        print(f"  Iteration {iter_num}: {v} problems")
    
    print("="*60 + "\n")

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