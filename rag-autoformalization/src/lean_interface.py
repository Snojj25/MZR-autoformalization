"""
Interface for interacting with Lean 4 compiler
"""
import subprocess
import tempfile
import os
import re
from typing import Dict, List
from config import config

class LeanInterface:
    def __init__(self):
        """Initialize Lean interface"""
        self.lean_path = config.LEAN_PATH
        self.timeout = config.LEAN_TIMEOUT
    
    def compile(self, lean_code: str) -> Dict:
        """
        Compile Lean 4 code and return results
        
        Args:
            lean_code: Lean 4 code to compile
            
        Returns:
            Dictionary with success status, errors, and messages
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.lean', 
            delete=False
        ) as f:
            f.write(lean_code)
            temp_file = f.name
        
        try:
            # Run Lean compiler
            result = subprocess.run(
                [self.lean_path, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse results
            success = result.returncode == 0
            errors = self._parse_errors(result.stderr) if not success else []
            
            return {
                "success": success,
                "errors": errors,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "error_categories": self._categorize_errors(errors)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "errors": ["Compilation timeout"],
                "stderr": "Timeout",
                "stdout": "",
                "error_categories": {"timeout": ["Compilation timeout"]}
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _parse_errors(self, stderr: str) -> List[str]:
        """Extract error messages from stderr"""
        if not stderr:
            return []
        
        # Simple extraction - get lines that look like errors
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower() or 'warning' in line.lower():
                errors.append(line.strip())
        
        return errors if errors else [stderr]
    
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