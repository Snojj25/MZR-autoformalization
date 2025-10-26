"""
Main entry point for RAG-Enhanced Iterative Autoformalization
"""
import argparse
import json
from src.pipeline import AutoformalizationPipeline
from src.utils import save_results, print_summary, calculate_metrics
from config import config

def load_test_problems(filepath: str = "tests/test_problems.json"):
    """Load test problems from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="RAG-Enhanced Iterative Autoformalization MVP"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "test"],
        default="test",
        help="Execution mode"
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Single problem in natural language (for single mode)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="tests/test_problems.json",
        help="Path to test problems JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename for results"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("="*60)
    print("RAG-Enhanced Iterative Autoformalization MVP")
    print("="*60)
    pipeline = AutoformalizationPipeline()
    
    # Execute based on mode
    if args.mode == "single":
        if not args.problem:
            print("Error: --problem required for single mode")
            return
        
        result = pipeline.process(args.problem)
        results = [result]
        
    elif args.mode == "batch" or args.mode == "test":
        problems = load_test_problems(args.test_file)
        print(f"\nLoaded {len(problems)} test problems")
        results = pipeline.batch_process(problems)
    
    # Save and display results
    save_results(results, args.output)
    print_summary(results)
    
    # Save metrics
    metrics = calculate_metrics(results)
    metrics_file = args.output.replace('.json', '_metrics.json') if args.output else None
    save_results([metrics], metrics_file)
    
    print("\n✓ Pipeline execution complete!")

if __name__ == "__main__":
    main()