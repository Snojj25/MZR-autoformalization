"""
Main entry point for RAG-Enhanced Iterative Autoformalization
"""
import os
# Set tokenizers parallelism before any imports that use it
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import webbrowser
from src.pipeline import AutoformalizationPipeline
from src.utils import save_results, print_summary, calculate_metrics
from src.report_generator import generate_html_report, generate_batch_html_report
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
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report and open in browser"
    )
    parser.add_argument(
        "--disable-manual",
        action="store_true",
        help="Disable manual proof tactics, use LLM-only proof generation"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("="*60)
    print("RAG-Enhanced Iterative Autoformalization MVP")
    print("="*60)
    if args.disable_manual:
        print("⚠ Manual proof tactics disabled - using LLM-only mode")
    pipeline = AutoformalizationPipeline(disable_manual_proof=args.disable_manual)
    
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
    
    # Generate HTML report if requested
    if args.html_report:
        try:
            import os
            if args.mode == "single" and len(results) == 1:
                report_path = generate_html_report(results[0])
                abs_path = os.path.abspath(report_path)
                print(f"\n[HTML Report] Generated: {abs_path}")
                # webbrowser.open(f"file://{abs_path}")
            else:
                report_path = generate_batch_html_report(results)
                abs_path = os.path.abspath(report_path)
                print(f"\n[HTML Report] Generated: {abs_path}")
                webbrowser.open(f"file://{abs_path}")
        except Exception as e:
            print(f"\n[Warning] Failed to generate HTML report: {e}")
    
    print("\n✓ Pipeline execution complete!")

if __name__ == "__main__":
    main()