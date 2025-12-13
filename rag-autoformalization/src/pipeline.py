"""
Main pipeline for RAG-enhanced iterative autoformalization
"""
from typing import Dict
from src.rag_module import MathLibRAG
from src.llm_client import LLMClient
from src.lean_interface import LeanInterface
from src.proof_tactics import ProofTactics
from config import config
import time

class AutoformalizationPipeline:
    def __init__(self):
        """Initialize the pipeline with all components"""
        self.rag = MathLibRAG()
        self.llm = LLMClient()
        self.lean = LeanInterface()
        self.proof_tactics = ProofTactics(llm_client=self.llm)
    
    def process(self, natural_language: str) -> Dict:
        """
        Process a natural language problem through the full pipeline
        
        Args:
            natural_language: Mathematical problem in natural language
            
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        results = {
            "natural_language": natural_language,
            "iterations": [],
            "final_statement": None,
            "compilation_success": False,
            "proof_success": False,
            "proof_tactic": None,
            "total_iterations": 0,
            "total_time": 0,
            "rag_examples": [],
        }
        
        # Step 1: RAG Retrieval
        print(f"\n{'='*60}")
        print(f"Processing: {natural_language[:80]}...")
        print(f"{'='*60}")
        print("\n[1/4] Retrieving similar examples...")
        
        similar_statements = self.rag.retrieve(natural_language)
        results["rag_examples"] = similar_statements
        few_shot_prompt = self.rag.format_for_prompt(similar_statements)
        proof_examples = self.rag.format_proof_examples(similar_statements)
        
        print(f"Retrieved {len(similar_statements)} similar examples: {similar_statements}")
        
        # Step 2: Iterative Formalization with Compiler Feedback
        print("\n[2/4] Starting iterative formalization...")
        
        previous_attempt = None
        compiler_errors = None
        
        for iteration in range(config.MAX_ITERATIONS):
            print(f"\n  Iteration {iteration + 1}/{config.MAX_ITERATIONS}")
            
            # Generate formal statement
            print("    - Generating Lean 4 statement...")
            formal_statement = self.llm.translate_to_lean(
                natural_language=natural_language,
                few_shot_examples=few_shot_prompt,
                previous_attempt=previous_attempt,
                compiler_errors=compiler_errors
            )
            
            if formal_statement is None:
                print("    - LLM generation failed")
                break
            
            # Compile and get feedback
            print("    - Compiling with Lean 4...")
            compile_result = self.lean.compile(
                formal_statement,
                iteration=iteration
            )
            
            iteration_data = {
                "iteration": iteration + 1,
                "formal_statement": formal_statement,
                "compilation_success": compile_result["success"],
                "errors": compile_result["errors"],
                "error_categories": compile_result["error_categories"],
                "lean_file_path": compile_result.get("file_path"),
                "lean_filename": compile_result.get("filename")
            }
            results["iterations"].append(iteration_data)
            
            if compile_result["success"]:
                print("    ✓ Compilation successful!")
                results["final_statement"] = formal_statement
                results["compilation_success"] = True
                results["total_iterations"] = iteration + 1
                break
            else:
                print(f"    ✗ Compilation failed: {compile_result['errors']}")
                if compile_result["error_categories"]:
                    print(f"      Error types: {list(compile_result['error_categories'].keys())}")
                
                # Prepare for next iteration
                previous_attempt = formal_statement
                compiler_errors = "\n".join(compile_result["errors"][:5])  # Top 5 errors
        
        # Step 3: Proof Attempt (if compilation successful)
        if results["compilation_success"]:
            print("\n[3/4] Attempting automated proof...")
            proof_result = self.proof_tactics.attempt_proof(
                results["final_statement"],
                proof_examples=proof_examples
            )
            
            results["proof_success"] = proof_result["proved"]
            results["proof_tactic"] = proof_result["tactic"]
            results["proof_attempts"] = proof_result["attempts"]
            
            if proof_result["proved"]:
                print(f"    ✓ Proof found using tactic: {proof_result['tactic']}")
                results["final_statement"] = proof_result["proof_code"]
            else:
                print(f"    ✗ No proof found (tried {len(proof_result['attempts'])} tactics)")
        else:
            print("\n[3/4] Skipping proof attempt (compilation failed)")
        
        # Step 4: Finalize results
        results["total_time"] = time.time() - start_time
        print(f"\n[4/4] Complete! Total time: {results['total_time']:.2f}s")
        
        return results
    
    def batch_process(self, problems: list) -> list:
        """Process multiple problems"""
        all_results = []
        for i, problem in enumerate(problems, 1):
            print(f"\n\n{'#'*60}")
            print(f"# Problem {i}/{len(problems)}")
            print(f"{'#'*60}")
            
            result = self.process(
                natural_language=problem.get("natural_language", problem)
            )
            all_results.append(result)
        
        return all_results