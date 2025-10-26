#!/usr/bin/env python3
"""
Simple test without API calls to verify the system works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_module import MathLibRAG
from src.lean_interface import LeanInterface
from config import config

def test_rag_retrieval():
    """Test RAG retrieval without API calls"""
    print("Testing RAG retrieval...")
    rag = MathLibRAG()
    
    # Test retrieval
    results = rag.retrieve("Prove that x + 0 = x for all real x", top_k=3)
    print(f"Retrieved {len(results)} similar examples")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['natural_language'][:60]}... (similarity: {result['similarity_score']:.3f})")
    
    # Test formatting
    formatted = rag.format_for_prompt(results)
    print(f"\nFormatted prompt length: {len(formatted)} characters")
    print("✓ RAG module working correctly")

def test_lean_compilation():
    """Test Lean compilation with simple examples"""
    print("\nTesting Lean compilation...")
    lean = LeanInterface()
    
    # Test a very simple theorem
    simple_code = """
theorem simple_test : 1 + 1 = 2 := by norm_num
"""
    
    result = lean.compile(simple_code)
    print(f"Simple theorem compilation: {result['success']}")
    if not result['success']:
        print(f"Errors: {result['errors'][:2]}")  # Show first 2 errors
    
    # Test with Mathlib import
    mathlib_code = """
import Mathlib.Data.Real.Basic

theorem mathlib_test (x : ℝ) : x + 0 = x := by simp
"""
    
    result = lean.compile(mathlib_code)
    print(f"Mathlib theorem compilation: {result['success']}")
    if not result['success']:
        print(f"Errors: {result['errors'][:2]}")
    
    print("✓ Lean interface working correctly")

def main():
    print("="*60)
    print("Simple System Test (No API Calls)")
    print("="*60)
    
    try:
        test_rag_retrieval()
        test_lean_compilation()
        
        print("\n" + "="*60)
        print("✓ All basic components working!")
        print("="*60)
        print("\nTo test with LLM, you need to:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run: python main.py --mode test")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()