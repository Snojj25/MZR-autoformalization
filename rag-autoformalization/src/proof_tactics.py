"""
Module for attempting basic proofs using common Lean tactics
"""
from typing import Dict, Optional
from config import config
from src.lean_interface import LeanInterface
from src.llm_client import LLMClient

class ProofTactics:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize proof tactics module
        
        Args:
            llm_client: Optional LLMClient instance for fallback proof generation
        """
        self.lean = LeanInterface()
        self.tactics = config.BASIC_TACTICS
        self.llm_client = llm_client
    
    def attempt_proof(self, formal_statement: str, proof_examples: str = "", disable_manual: bool = False) -> Dict:
        """
        Attempt to prove a statement using basic tactics
        
        Args:
            formal_statement: Lean 4 theorem statement with 'sorry'
            proof_examples: Few-shot examples of similar theorems with proofs (for LLM fallback)
            disable_manual: If True, skip manual tactics and go straight to LLM fallback
            
        Returns:
            Dictionary with proof results
        """
        results = {
            "proved": False,
            "tactic": None,
            "proof_code": None,
            "attempts": []
        }
        
        # Extract theorem declaration (remove 'sorry')
        theorem_decl = formal_statement.replace(":= by sorry", "").strip()
        
        # Try each tactic (unless disabled)
        failed_tactics = []
        if not disable_manual:
            for i, tactic in enumerate(self.tactics):
                proof_code = f"{theorem_decl} := by {tactic}"
                
                # Compile and check (use proof_tactic suffix for filename)
                compile_result = self.lean.compile(proof_code, iteration=i)
                
                attempt_result = {
                    "tactic": tactic,
                    "success": compile_result["success"],
                    "errors": compile_result["errors"],
                    "proof_code": proof_code
                }
                results["attempts"].append(attempt_result)
                
                if compile_result["success"]:
                    results["proved"] = True
                    results["tactic"] = tactic
                    results["proof_code"] = proof_code
                    break
                else:
                    failed_tactics.append(tactic)
        else:
            # Manual tactics disabled, mark as attempted but not used
            results["manual_tactics_disabled"] = True
        
        # If manual tactics failed (or were disabled) and LLM client is available, try LLM-generated proof with refinements
        if not results["proved"] and self.llm_client is not None:
            results["llm_fallback_attempted"] = True
            if disable_manual:
                print("    - Manual tactics disabled, trying LLM-generated proof...")
            else:
                print("    - Manual tactics failed, trying LLM-generated proof...")
            
            previous_proof_attempt = None
            compiler_errors = None
            
            # Try multiple iterations of LLM proof generation with refinements
            for llm_iteration in range(config.MAX_PROOF_ITERATIONS):
                if llm_iteration > 0:
                    print(f"    - Refining LLM-generated proof (attempt {llm_iteration + 1}/{config.MAX_PROOF_ITERATIONS})...")
                else:
                    print("    - Generating initial LLM proof...")
                
                # Generate proof using LLM (initial or refinement)
                llm_proof = self.llm_client.generate_proof_tactic(
                    formal_statement=formal_statement,
                    failed_attempts=failed_tactics if llm_iteration == 0 else None,
                    proof_examples=proof_examples,
                    previous_attempt=previous_proof_attempt,
                    compiler_errors=compiler_errors
                )
                
                if not llm_proof:
                    print(f"    ✗ LLM proof generation failed (attempt {llm_iteration + 1})")
                    # If generation fails, we can't continue refining
                    break

                # Try to compile the LLM-generated proof
                # Use a special iteration number to distinguish from manual attempts
                # Set is_llm_fallback flag to indicate this is from LLM fallback
                compile_result = self.lean.compile(
                    llm_proof, 
                    iteration=len(self.tactics) + llm_iteration, 
                    is_llm_fallback=True
                )
                
                attempt_result = {
                    "tactic": f"llm_generated_iter_{llm_iteration + 1}",
                    "success": compile_result["success"],
                    "errors": compile_result["errors"],
                    "proof_code": llm_proof,
                    "iteration": llm_iteration + 1
                }
                results["attempts"].append(attempt_result)
                
                if compile_result["success"]:
                    results["proved"] = True
                    results["tactic"] = f"llm_generated_iter_{llm_iteration + 1}"
                    results["proof_code"] = llm_proof
                    print(f"    ✓ LLM-generated proof successful (after {llm_iteration + 1} attempt(s))!")
                    break
                else:
                    print(f"    ✗ LLM-generated proof failed (attempt {llm_iteration + 1}): {len(compile_result['errors'])} errors")
                    # Prepare for next iteration
                    previous_proof_attempt = llm_proof
                    compiler_errors = "\n".join(compile_result["errors"][:5])  # Top 5 errors
            
                    
        else:
            results["llm_fallback_attempted"] = False
        
        return results
    
    def attempt_combined_tactics(self, formal_statement: str) -> Dict:
        """Try combinations of tactics (bonus feature)"""
        # For MVP, we'll skip this but provide the structure
        # Future: try "by simp; ring", "by linarith; norm_num", etc.
        return {"proved": False, "note": "Combined tactics not implemented in MVP"}