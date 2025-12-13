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
    
    def attempt_proof(self, formal_statement: str, proof_examples: str = "") -> Dict:
        """
        Attempt to prove a statement using basic tactics
        
        Args:
            formal_statement: Lean 4 theorem statement with 'sorry'
            proof_examples: Few-shot examples of similar theorems with proofs (for LLM fallback)
            
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
        
        # Try each tactic
        failed_tactics = []
        for i, tactic in enumerate(self.tactics):
            proof_code = f"{theorem_decl} := by {tactic}"
            
            # Compile and check (use proof_tactic suffix for filename)
            compile_result = self.lean.compile(proof_code, iteration=i)
            
            attempt_result = {
                "tactic": tactic,
                "success": compile_result["success"],
                "errors": compile_result["errors"]
            }
            results["attempts"].append(attempt_result)
            
            if compile_result["success"]:
                results["proved"] = True
                results["tactic"] = tactic
                results["proof_code"] = proof_code
                break
            else:
                failed_tactics.append(tactic)
        
        # If manual tactics failed and LLM client is available, try LLM-generated proof
        if not results["proved"] and self.llm_client is not None:
            results["llm_fallback_attempted"] = True
            print("    - Manual tactics failed, trying LLM-generated proof...")
            
            # Generate proof using LLM
            llm_proof = self.llm_client.generate_proof_tactic(
                formal_statement=formal_statement,
                failed_attempts=failed_tactics,
                proof_examples=proof_examples
            )
            
            if llm_proof:
                # Try to compile the LLM-generated proof
                # Use a special iteration number to distinguish from manual attempts
                # Set is_llm_fallback flag to indicate this is from LLM fallback
                compile_result = self.lean.compile(llm_proof, iteration=len(self.tactics), is_llm_fallback=True)
                
                attempt_result = {
                    "tactic": "llm_generated",
                    "success": compile_result["success"],
                    "errors": compile_result["errors"],
                    "proof_code": llm_proof
                }
                results["attempts"].append(attempt_result)
                
                if compile_result["success"]:
                    results["proved"] = True
                    results["tactic"] = "llm_generated"
                    results["proof_code"] = llm_proof
                    print("    ✓ LLM-generated proof successful!")
                else:
                    print(f"    ✗ LLM-generated proof failed: {len(compile_result['errors'])} errors")
            else:
                results["llm_fallback_attempted"] = False
                print("    ✗ LLM proof generation failed")
        else:
            results["llm_fallback_attempted"] = False
        
        return results
    
    def attempt_combined_tactics(self, formal_statement: str) -> Dict:
        """Try combinations of tactics (bonus feature)"""
        # For MVP, we'll skip this but provide the structure
        # Future: try "by simp; ring", "by linarith; norm_num", etc.
        return {"proved": False, "note": "Combined tactics not implemented in MVP"}