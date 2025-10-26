"""
Module for attempting basic proofs using common Lean tactics
"""
from typing import Dict, Optional
from config import config
from src.lean_interface import LeanInterface

class ProofTactics:
    def __init__(self):
        """Initialize proof tactics module"""
        self.lean = LeanInterface()
        self.tactics = config.BASIC_TACTICS
    
    def attempt_proof(self, formal_statement: str, problem_id: Optional[str] = None) -> Dict:
        """
        Attempt to prove a statement using basic tactics
        
        Args:
            formal_statement: Lean 4 theorem statement with 'sorry'
            problem_id: Optional problem identifier for filename
            
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
        for i, tactic in enumerate(self.tactics):
            proof_code = f"{theorem_decl} := by {tactic}"
            
            # Compile and check (use proof_tactic suffix for filename)
            proof_problem_id = f"{problem_id}_proof" if problem_id else None
            compile_result = self.lean.compile(proof_code, problem_id=proof_problem_id, iteration=i)
            
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
        
        return results
    
    def attempt_combined_tactics(self, formal_statement: str) -> Dict:
        """Try combinations of tactics (bonus feature)"""
        # For MVP, we'll skip this but provide the structure
        # Future: try "by simp; ring", "by linarith; norm_num", etc.
        return {"proved": False, "note": "Combined tactics not implemented in MVP"}