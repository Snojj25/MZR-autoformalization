"""
Interface for interacting with Lean 4 compiler
"""
import subprocess
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
from config import config

# Set tokenizers parallelism to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LeanInterface:
    def __init__(self):
        """Initialize Lean interface"""
        self.lean_path = config.LEAN_PATH
        self.timeout = config.LEAN_TIMEOUT
        self.lean_examples_dir = config.LEAN_EXAMPLES_DIR
        self.lean_examples_src_dir = config.LEAN_EXAMPLES_SRC_DIR
        
        # Ensure the source directory exists
        os.makedirs(self.lean_examples_src_dir, exist_ok=True)
    
    def compile(self, lean_code: str, iteration: Optional[int] = None, is_llm_fallback: bool = False) -> Dict:
        """
        Compile Lean 4 code and return results
        
        Args:
            lean_code: Lean 4 code to compile
            problem_id: Optional problem identifier for filename
            iteration: Optional iteration number for filename
            is_llm_fallback: Flag indicating if this is from LLM proof fallback
            
        Returns:
            Dictionary with success status, errors, messages, and file path
        """
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_suffix = "_llm_fallback" if is_llm_fallback else ""
        
        if iteration is not None:
            filename = f"pipeline_iter{iteration}{fallback_suffix}_{timestamp}.lean"
        else:
            filename = f"pipeline{fallback_suffix}_{timestamp}.lean"
        
        # Create file in run-specific folder or default location
        file_path = os.path.join(self.lean_examples_src_dir, filename)
        
        # Use longer timeout for LLM fallback attempts (they may be more complex)
        compile_timeout = self.timeout * 2 if is_llm_fallback else self.timeout
        
        try:
            # Write Lean code to file
            with open(file_path, 'w') as f:
                f.write(lean_code)
            
            relative_path = os.path.join("LeanExamples", filename)
            
            result = subprocess.run(
                ["lake", "env", "lean", relative_path],
                cwd=self.lean_examples_dir,
                capture_output=True,
                text=True,
                timeout=compile_timeout
            )

            # Parse results
            success = result.returncode == 0
            # Parse errors from both stderr and stdout (Lean may output errors to either)
            all_output = result.stderr + "\n" + result.stdout if result.stderr else result.stdout
            errors = self._parse_errors(all_output) if not success else []
            
            return {
                "success": success,
                "errors": errors,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "error_categories": self._categorize_errors(errors),
                "file_path": file_path,
                "filename": filename
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "errors": ["Compilation timeout"],
                "stderr": "Timeout",
                "stdout": "",
                "error_categories": {"timeout": ["Compilation timeout"]},
                "file_path": file_path,
                "filename": filename
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [f"Compilation exception: {str(e)}"],
                "stderr": str(e),
                "stdout": "",
                "error_categories": {"exception": [str(e)]},
                "file_path": file_path,
                "filename": filename
            }
        # Note: We keep the file for debugging/visualization instead of deleting it
    
    def _parse_errors(self, output: str) -> List[str]:
        """Extract error messages from Lean output"""
        if not output:
            return ["No error output available"]
        
        errors = []
        lines = output.split('\n')
        
        # Try to capture error blocks (lines containing "error", "warning", etc.)
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['error', 'warning', 'failed', 'unexpected']):
                # Capture this line and next few lines for context
                error_block = line.strip()
                # Add up to 3 more lines for context
                for j in range(1, min(4, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    if next_line:
                        error_block += "\n" + next_line
                errors.append(error_block)
        
        # If no specific errors found, return the entire output
        return errors if errors else [output.strip()] if output.strip() else ["Unknown error occurred"]
    
    def _categorize_errors(self, errors: List[str]) -> Dict[str, List[str]]:
        """Categorize errors by type"""
        categories = {
            "syntax": [],
            "type": [],
            "semantic": [],
            "unknown": []
        }
        
        syntax_patterns = ["unexpected", "expected", "missing"]
        type_patterns = ["type mismatch", "has type", "expected type"]
        semantic_patterns = ["unknown identifier", "not found", "not in scope"]
        
        for error in errors:
            error_lower = error.lower()
            if any(p in error_lower for p in syntax_patterns):
                categories["syntax"].append(error)
            elif any(p in error_lower for p in type_patterns):
                categories["type"].append(error)
            elif any(p in error_lower for p in semantic_patterns):
                categories["semantic"].append(error)
            else:
                categories["unknown"].append(error)
        
        return {k: v for k, v in categories.items() if v}