"""
Basic tests to verify pipeline components work
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_module import MathLibRAG
from src.llm_client import LLMClient
from src.lean_interface import LeanInterface
from config import config

def test_rag_module():
    """Test RAG retrieval"""
    print("\n[TEST] RAG Module")
    rag = MathLibRAG()
    results = rag.retrieve("Prove that x + 0 = x for all real x", top_k=3)
    assert len(results) == 3, "Should retrieve 3 examples"
    assert "natural_language" in results[0], "Should have natural_language field"
    print("✓ RAG module works")

def test_lean_interface():
    """Test Lean compilation"""
    print("\n[TEST] Lean Interface")
    lean = LeanInterface()
    
    # Test valid code
    valid_code = """
import Mathlib

theorem test_thm (x : ℝ) : x + 0 = x := by simp
"""
    result = lean.compile(valid_code)
    print(f"  Valid code compilation: {result['success']}")
    
    # Test invalid code
    invalid_code = """
theorem bad_syntax : invalid
"""
    result = lean.compile(invalid_code)
    assert not result['success'], "Should fail on invalid code"
    assert len(result['errors']) > 0, "Should have errors"
    print("✓ Lean interface works")

def test_config():
    """Test configuration"""
    print("\n[TEST] Configuration")
    assert config.OPENAI_API_KEY is not None, "OPENAI_API_KEY must be set"
    assert os.path.exists(config.DATA_DIR), "Data directory must exist"
    print("✓ Configuration valid")

if __name__ == "__main__":
    print("="*60)
    print("Running Component Tests")
    print("="*60)
    
    try:
        test_config()
        test_rag_module()
        test_lean_interface()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()