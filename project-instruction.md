# Detailed Instructions for Code Agent: RAG-Enhanced Iterative Autoformalization MVP  

## Project Overview  
Build a **RAG-Enhanced Iterative Autoformalization System** that translates natural language math problems into Lean 4 formal statements, uses compiler feedback for iterative refinement, attempts basic proofs, and leverages RAG (Retrieval-Augmented Generation) for few-shot learning.  

---  

## 🎯 MVP Success Criteria  

The MVP must demonstrate:  
1. ✅ Natural language → Lean 4 translation  
2. ✅ RAG retrieval of similar examples from miniF2F dataset  
3. ✅ Iterative refinement using Lean 4 compiler errors (max 3 iterations)  
4. ✅ Basic proof attempts using common tactics  
5. ✅ Clear metrics and logging  
6. ✅ Simple command-line interface for testing  

---  

## 📁 Project Structure  

Create the following directory structure:  

```  
rag-autoformalization/  
├── README.md  
├── requirements.txt  
├── .env.example  
├── config.py  
├── data/  
│   ├── minif2f_statements.json  
│   └── embeddings/  
│       └── minif2f_embeddings.pkl  
├── src/  
│   ├── __init__.py  
│   ├── rag_module.py  
│   ├── lean_interface.py  
│   ├── llm_client.py  
│   ├── proof_tactics.py  
│   ├── pipeline.py  
│   └── utils.py  
├── prompts/  
│   ├── initial_translation.txt  
│   ├── refinement_with_errors.txt  
│   └── proof_generation.txt  
├── tests/  
│   ├── test_problems.json  
│   └── test_pipeline.py  
├── results/  
│   └── .gitkeep  
└── main.py  
```  

---  

## 🔧 Step 1: Environment Setup  

### Task 1.1: Create `requirements.txt`  
```txt  
openai>=1.0.0  
sentence-transformers>=2.2.0  
faiss-cpu>=1.7.4  
numpy>=1.24.0  
python-dotenv>=1.0.0  
tqdm>=4.65.0  
pandas>=2.0.0  
```  

### Task 1.2: Create `.env.example`  
```bash  
OPENAI_API_KEY=your_api_key_here  
OPENAI_MODEL=gpt-4  
LEAN_PATH=/usr/local/bin/lean  
LEAN_TIMEOUT=10  
MAX_ITERATIONS=3  
TOP_K_RETRIEVAL=3  
```  

### Task 1.3: Create `config.py`  
```python  
import os  
from dotenv import load_dotenv  

load_dotenv()  

class Config:  
    # OpenAI settings  
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  
    OPENAI_TEMPERATURE = 0.6  
    
    # Lean settings  
    LEAN_PATH = os.getenv("LEAN_PATH", "lean")  
    LEAN_TIMEOUT = int(os.getenv("LEAN_TIMEOUT", "10"))  
    
    # Pipeline settings  
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))  
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))  
    
    # Paths  
    DATA_DIR = "data"  
    RESULTS_DIR = "results"  
    PROMPTS_DIR = "prompts"  
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings", "minif2f_embeddings.pkl")  
    MINIF2F_PATH = os.path.join(DATA_DIR, "minif2f_statements.json")  
    
    # Proof tactics  
    BASIC_TACTICS = [  
        "simp",  
        "ring",  
        "linarith",  
        "norm_num",  
        "omega",  
        "aesop"  
    ]  

config = Config()  
```  

---  

## 📊 Step 2: Data Preparation  

### Task 2.1: Create miniF2F sample dataset  

Create `data/minif2f_statements.json` with 20 sample problems:  

```python  
# Agent: Create this file with the following structure  
[  
    {  
        "id": "algebra_001",  
        "natural_language": "Prove that for all positive real numbers x, x + 1/x >= 2",  
        "formal_statement": "theorem algebra_001 (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2",  
        "category": "algebra",  
        "difficulty": "easy"  
    },  
    {  
        "id": "algebra_002",  
        "natural_language": "For any real number a, prove that a^2 >= 0",  
        "formal_statement": "theorem algebra_002 (a : ℝ) : a^2 ≥ 0",  
        "category": "algebra",  
        "difficulty": "easy"  
    },  
    {  
        "id": "algebra_003",  
        "natural_language": "Prove that for positive reals a and b, if a < b then a^2 < b^2",  
        "formal_statement": "theorem algebra_003 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) : a^2 < b^2",  
        "category": "algebra",  
        "difficulty": "medium"  
    },  
    // Add 17 more similar examples covering:  
    // - Basic inequalities  
    // - Algebraic identities  
    // - Simple number theory  
    // - Basic geometry  
]  
```  

**Agent Instruction**: Generate 20 diverse, simple mathematical problems in this format. Include problems from algebra, inequalities, and basic number theory. Ensure formal statements are valid Lean 4 syntax.  

---  

## 🔍 Step 3: RAG Module Implementation  

### Task 3.1: Create `src/rag_module.py`  

```python  
"""  
RAG Module for retrieving similar mathematical statements  
"""  
import json  
import pickle  
import numpy as np  
from sentence_transformers import SentenceTransformer  
import faiss  
from typing import List, Dict  
from config import config  

class MathLibRAG:  
    def __init__(self):  
        """Initialize the RAG module with sentence encoder and FAISS index"""  
        print("Initializing RAG module...")  
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  
        self.statements = self._load_statements()  
        self.embeddings = None  
        self.index = None  
        self._build_or_load_index()  
        
    def _load_statements(self) -> List[Dict]:  
        """Load miniF2F statements from JSON file"""  
        with open(config.MINIF2F_PATH, 'r') as f:  
            return json.load(f)  
    
    def _build_or_load_index(self):  
        """Build FAISS index or load from cache"""  
        try:  
            # Try to load cached embeddings  
            with open(config.EMBEDDINGS_PATH, 'rb') as f:  
                cache = pickle.load(f)  
                self.embeddings = cache['embeddings']  
                self.index = cache['index']  
            print("Loaded cached embeddings")  
        except FileNotFoundError:  
            # Build new embeddings  
            print("Building new embeddings...")  
            self._build_embeddings()  
            self._save_embeddings()  
    
    def _build_embeddings(self):  
        """Build embeddings for all statements"""  
        texts = [stmt['natural_language'] for stmt in self.statements]  
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)  
        
        # Create FAISS index  
        dimension = self.embeddings.shape[1]  
        self.index = faiss.IndexFlatL2(dimension)  
        self.index.add(self.embeddings.astype('float32'))  
    
    def _save_embeddings(self):  
        """Save embeddings to cache"""  
        import os  
        os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True)  
        with open(config.EMBEDDINGS_PATH, 'wb') as f:  
            pickle.dump({  
                'embeddings': self.embeddings,  
                'index': self.index  
            }, f)  
        print(f"Saved embeddings to {config.EMBEDDINGS_PATH}")  
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:  
        """  
        Retrieve top-k similar statements for a given query  
        
        Args:  
            query: Natural language mathematical problem  
            top_k: Number of similar examples to retrieve  
            
        Returns:  
            List of similar statement dictionaries  
        """  
        if top_k is None:  
            top_k = config.TOP_K_RETRIEVAL  
            
        # Encode query  
        query_embedding = self.encoder.encode([query])  
        
        # Search in FAISS index  
        distances, indices = self.index.search(  
            query_embedding.astype('float32'),   
            top_k  
        )  
        
        # Return similar statements  
        similar_statements = []  
        for idx, distance in zip(indices[0], distances[0]):  
            stmt = self.statements[idx].copy()  
            stmt['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity  
            similar_statements.append(stmt)  
        
        return similar_statements  
    
    def format_for_prompt(self, similar_statements: List[Dict]) -> str:  
        """Format retrieved statements as few-shot examples"""  
        formatted = "Here are similar problems and their formalizations:\n\n"  
        for i, stmt in enumerate(similar_statements, 1):  
            formatted += f"Example {i}:\n"  
            formatted += f"Natural Language: {stmt['natural_language']}\n"  
            formatted += f"Formal Statement: {stmt['formal_statement']}\n\n"  
        return formatted  
```  

**Agent Task**: Implement this exact code in `src/rag_module.py`. Ensure all imports are correct and add error handling for missing files.  

---  

## 🤖 Step 4: LLM Client Implementation  

### Task 4.1: Create `src/llm_client.py`  

```python  
"""  
LLM Client for interacting with OpenAI API  
"""  
import openai  
from typing import Optional  
from config import config  

class LLMClient:  
    def __init__(self):  
        """Initialize OpenAI client"""  
        openai.api_key = config.OPENAI_API_KEY  
        self.model = config.OPENAI_MODEL  
        self.temperature = config.OPENAI_TEMPERATURE  
    
    def generate(self, prompt: str, system_message: str = None) -> str:  
        """  
        Generate completion from OpenAI  
        
        Args:  
            prompt: User prompt  
            system_message: Optional system message  
            
        Returns:  
            Generated text  
        """  
        messages = []  
        if system_message:  
            messages.append({"role": "system", "content": system_message})  
        messages.append({"role": "user", "content": prompt})  
        
        try:  
            response = openai.ChatCompletion.create(  
                model=self.model,  
                messages=messages,  
                temperature=self.temperature,  
                max_tokens=1000  
            )  
            return response.choices[0].message.content.strip()  
        except Exception as e:  
            print(f"Error calling OpenAI API: {e}")  
            return None  
    
    def translate_to_lean(  
        self,   
        natural_language: str,   
        few_shot_examples: str = "",  
        previous_attempt: str = None,  
        compiler_errors: str = None  
    ) -> str:  
        """  
        Translate natural language to Lean 4 with optional refinement  
        
        Args:  
            natural_language: Problem statement  
            few_shot_examples: RAG-retrieved examples  
            previous_attempt: Previous formalization attempt  
            compiler_errors: Errors from previous attempt  
            
        Returns:  
            Lean 4 formal statement  
        """  
        # Load appropriate prompt  
        if previous_attempt is None:  
            prompt = self._load_prompt("initial_translation.txt")  
            prompt = prompt.format(  
                few_shot_examples=few_shot_examples,  
                natural_language=natural_language  
            )  
        else:  
            prompt = self._load_prompt("refinement_with_errors.txt")  
            prompt = prompt.format(  
                few_shot_examples=few_shot_examples,  
                natural_language=natural_language,  
                previous_attempt=previous_attempt,  
                compiler_errors=compiler_errors  
            )  
        
        system_message = "You are an expert in Lean 4 theorem proving and mathematical formalization."  
        
        return self.generate(prompt, system_message)  
    
    def _load_prompt(self, filename: str) -> str:  
        """Load prompt template from file"""  
        import os  
        path = os.path.join(config.PROMPTS_DIR, filename)  
        with open(path, 'r') as f:  
            return f.read()  
```  

**Agent Task**: Implement this code. Note that you'll need to create the prompt templates next.  

---  

## 📝 Step 5: Prompt Templates  

### Task 5.1: Create `prompts/initial_translation.txt`  

```text  
You are translating mathematical problems into Lean 4 formal statements.  

{few_shot_examples}  

Now translate this problem:  
Natural Language: {natural_language}  

Requirements:  
1. Output ONLY the Lean 4 theorem statement (no proof)  
2. Use correct Lean 4 syntax  
3. Include necessary imports (import Mathlib)  
4. Use appropriate type annotations  
5. Follow the pattern from the examples above  

Output format:  
import Mathlib  

theorem [name] [parameters] : [statement] := by sorry  
```  

### Task 5.2: Create `prompts/refinement_with_errors.txt`  

```text  
You previously attempted to formalize this problem, but the Lean 4 compiler reported errors.  

{few_shot_examples}  

Original Problem: {natural_language}  

Previous Attempt:  
{previous_attempt}  

Compiler Errors:  
{compiler_errors}  

Please fix the errors and generate a corrected Lean 4 statement. Focus on:  
1. Syntax errors (missing tokens, incorrect structure)  
2. Type mismatches  
3. Unknown identifiers (use correct Mathlib imports)  

Output ONLY the corrected Lean 4 code:  
```  

### Task 5.3: Create `prompts/proof_generation.txt`  

```text  
Generate a Lean 4 proof for this theorem:  

{formal_statement}  

You can use these tactics: {available_tactics}  

Output the complete theorem with proof:  
```  

**Agent Task**: Create these three files exactly as shown in the `prompts/` directory.  

---  

## 🔌 Step 6: Lean Interface  

### Task 6.1: Create `src/lean_interface.py`  

```python  
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
```  

**Agent Task**: Implement this code exactly. This handles all compilation and error parsing.  

---  

## 🎯 Step 7: Proof Tactics Module  

### Task 7.1: Create `src/proof_tactics.py`  

```python  
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
    
    def attempt_proof(self, formal_statement: str) -> Dict:  
        """  
        Attempt to prove a statement using basic tactics  
        
        Args:  
            formal_statement: Lean 4 theorem statement with 'sorry'  
            
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
        for tactic in self.tactics:  
            proof_code = f"{theorem_decl} := by {tactic}"  
            
            # Compile and check  
            compile_result = self.lean.compile(proof_code)  
            
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
```  

**Agent Task**: Implement this code. This will try basic tactics to automatically prove theorems.  

---  

## 🔄 Step 8: Main Pipeline Implementation  

### Task 8.1: Create `src/pipeline.py`  

```python  
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
        print("Initializing pipeline components...")  
        self.rag = MathLibRAG()  
        self.llm = LLMClient()  
        self.lean = LeanInterface()  
        self.proof_tactics = ProofTactics()  
        print("Pipeline ready!")  
    
    def process(self, natural_language: str, problem_id: str = None) -> Dict:  
        """  
        Process a natural language problem through the full pipeline  
        
        Args:  
            natural_language: Mathematical problem in natural language  
            problem_id: Optional identifier for logging  
            
        Returns:  
            Complete results dictionary  
        """  
        start_time = time.time()  
        
        results = {  
            "problem_id": problem_id,  
            "natural_language": natural_language,  
            "iterations": [],  
            "final_statement": None,  
            "compilation_success": False,  
            "proof_success": False,  
            "proof_tactic": None,  
            "total_iterations": 0,  
            "total_time": 0,  
            "rag_examples": []  
        }  
        
        # Step 1: RAG Retrieval  
        print(f"\n{'='*60}")  
        print(f"Processing: {natural_language[:80]}...")  
        print(f"{'='*60}")  
        print("\n[1/4] Retrieving similar examples...")  
        
        similar_statements = self.rag.retrieve(natural_language)  
        results["rag_examples"] = similar_statements  
        few_shot_prompt = self.rag.format_for_prompt(similar_statements)  
        
        print(f"Retrieved {len(similar_statements)} similar examples")  
        
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
            compile_result = self.lean.compile(formal_statement)  
            
            iteration_data = {  
                "iteration": iteration + 1,  
                "formal_statement": formal_statement,  
                "compilation_success": compile_result["success"],  
                "errors": compile_result["errors"],  
                "error_categories": compile_result["error_categories"]  
            }  
            results["iterations"].append(iteration_data)  
            
            if compile_result["success"]:  
                print("    ✓ Compilation successful!")  
                results["final_statement"] = formal_statement  
                results["compilation_success"] = True  
                results["total_iterations"] = iteration + 1  
                break  
            else:  
                print(f"    ✗ Compilation failed: {len(compile_result['errors'])} errors")  
                if compile_result["error_categories"]:  
                    print(f"      Error types: {list(compile_result['error_categories'].keys())}")  
                
                # Prepare for next iteration  
                previous_attempt = formal_statement  
                compiler_errors = "\n".join(compile_result["errors"][:5])  # Top 5 errors  
        
        # Step 3: Proof Attempt (if compilation successful)  
        if results["compilation_success"]:  
            print("\n[3/4] Attempting automated proof...")  
            proof_result = self.proof_tactics.attempt_proof(results["final_statement"])  
            
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
                natural_language=problem.get("natural_language", problem),  
                problem_id=problem.get("id", f"problem_{i}")  
            )  
            all_results.append(result)  
        
        return all_results  
```  

**Agent Task**: Implement this exact pipeline code. This is the core of the system that orchestrates all components.  

---  

## 🛠️ Step 9: Utility Functions  

### Task 9.1: Create `src/utils.py`  

```python  
"""  
Utility functions for logging, metrics, and visualization  
"""  
import json  
import os  
from datetime import datetime  
from typing import List, Dict  
import pandas as pd  
from config import config  

def save_results(results: List[Dict], filename: str = None):  
    """Save results to JSON file"""  
    if filename is None:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        filename = f"results_{timestamp}.json"  
    
    filepath = os.path.join(config.RESULTS_DIR, filename)  
    os.makedirs(config.RESULTS_DIR, exist_ok=True)  
    
    with open(filepath, 'w') as f:  
        json.dump(results, f, indent=2)  
    
    print(f"\nResults saved to: {filepath}")  
    return filepath  

def calculate_metrics(results: List[Dict]) -> Dict:  
    """Calculate aggregate metrics from results"""  
    total = len(results)  
    
    # Compilation metrics  
    compilation_success = sum(1 for r in results if r.get("compilation_success"))  
    compilation_rate = compilation_success / total if total > 0 else 0  
    
    # Proof metrics  
    proof_success = sum(1 for r in results if r.get("proof_success"))  
    proof_rate = proof_success / total if total > 0 else 0  
    
    # Iteration metrics  
    iterations_list = [r.get("total_iterations", 0) for r in results if r.get("compilation_success")]  
    avg_iterations = sum(iterations_list) / len(iterations_list) if iterations_list else 0  
    
    # Time metrics  
    times = [r.get("total_time", 0) for r in results]  
    avg_time = sum(times) / len(times) if times else 0  
    
    # First-attempt success  
    first_attempt_success = sum(  
        1 for r in results   
        if r.get("compilation_success") and r.get("total_iterations") == 1  
    )  
    first_attempt_rate = first_attempt_success / total if total > 0 else 0  
    
    metrics = {  
        "total_problems": total,  
        "compilation_success_rate": compilation_rate,  
        "proof_success_rate": proof_rate,  
        "first_attempt_success_rate": first_attempt_rate,  
        "avg_iterations_to_success": avg_iterations,  
        "avg_time_per_problem": avg_time,  
        "metrics_by_iteration": _calculate_by_iteration(results)  
    }  
    
    return metrics  

def _calculate_by_iteration(results: List[Dict]) -> Dict:  
    """Calculate success rate by iteration number"""  
    by_iteration = {}  
    for i in range(1, config.MAX_ITERATIONS + 1):  
        success_at_i = sum(  
            1 for r in results   
            if r.get("compilation_success") and r.get("total_iterations") == i  
        )  
        by_iteration[f"iteration_{i}"] = success_at_i  
    return by_iteration  

def print_summary(results: List[Dict]):  
    """Print a formatted summary of results"""  
    metrics = calculate_metrics(results)  
    
    print("\n" + "="*60)  
    print("EVALUATION SUMMARY")  
    print("="*60)  
    print(f"Total Problems: {metrics['total_problems']}")  
    print(f"\nCompilation Success Rate: {metrics['compilation_success_rate']:.1%}")  
    print(f"Proof Success Rate: {metrics['proof_success_rate']:.1%}")  
    print(f"First-Attempt Success Rate: {metrics['first_attempt_success_rate']:.1%}")  
    print(f"\nAverage Iterations (when successful): {metrics['avg_iterations_to_success']:.2f}")  
    print(f"Average Time per Problem: {metrics['avg_time_per_problem']:.2f}s")  
    
    print(f"\nSuccess by Iteration:")  
    for k, v in metrics['metrics_by_iteration'].items():  
        iter_num = k.split('_')[1]  
        print(f"  Iteration {iter_num}: {v} problems")  
    
    print("="*60 + "\n")  

def create_results_dataframe(results: List[Dict]) -> pd.DataFrame:  
    """Convert results to pandas DataFrame for analysis"""  
    rows = []  
    for r in results:  
        rows.append({  
            "problem_id": r.get("problem_id"),  
            "compilation_success": r.get("compilation_success"),  
            "proof_success": r.get("proof_success"),  
            "iterations": r.get("total_iterations"),  
            "time": r.get("total_time"),  
            "proof_tactic": r.get("proof_tactic")  
        })  
    return pd.DataFrame(rows)  
```  

**Agent Task**: Implement these utility functions for metrics and result processing.  

---  

## 🚀 Step 10: Main Entry Point  

### Task 10.1: Create `main.py`  

```python  
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
```  

**Agent Task**: Implement this main entry point exactly as shown.  

---  

## 🧪 Step 11: Test Problems  

### Task 11.1: Create `tests/test_problems.json`  

```json  
[  
  {  
    "id": "test_001",  
    "natural_language": "Prove that for all real numbers x, x + 0 = x",  
    "category": "algebra",  
    "expected_difficulty": "trivial"  
  },  
  {  
    "id": "test_002",  
    "natural_language": "Show that 1 + 1 equals 2",  
    "category": "arithmetic",  
    "expected_difficulty": "trivial"  
  },  
  {  
    "id": "test_003",  
    "natural_language": "For any real number a, prove that a * 1 = a",  
    "category": "algebra",  
    "expected_difficulty": "trivial"  
  },  
  {  
    "id": "test_004",  
    "natural_language": "Prove that for positive real x, x + 1/x >= 2",  
    "category": "inequality",  
    "expected_difficulty": "easy"  
  },  
  {  
    "id": "test_005",  
    "natural_language": "Show that for any real a, a^2 >= 0",  
    "category": "algebra",  
    "expected_difficulty": "easy"  
  }  
]  
```  

**Agent Task**: Create this test file with 5 simple problems. These will be used to verify the MVP works.  

---  

## 📚 Step 12: README Documentation  

### Task 12.1: Create `README.md`  

```markdown  
# RAG-Enhanced Iterative Autoformalization MVP  

An inference-time system for translating natural language mathematics into Lean 4 formal statements using:  
- **RAG (Retrieval-Augmented Generation)** for few-shot learning  
- **Iterative refinement** with compiler feedback  
- **Automated proof attempts** using basic tactics  

## Features  

✅ Natural language → Lean 4 translation  
✅ Semantic similarity-based retrieval (miniF2F)  
✅ Up to 3 iterations with compiler error feedback  
✅ Basic proof attempts (6 tactics)  
✅ Comprehensive metrics and logging  

## Setup  

1. **Install Lean 4**:  
```bash  
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh  
```  

2. **Install Python dependencies**:  
```bash  
pip install -r requirements.txt  
```  

3. **Configure environment**:  
```bash  
cp .env.example .env  
# Edit .env and add your OPENAI_API_KEY  
```  

4. **Prepare data**:  
```bash  
# The first run will automatically build embeddings  
python main.py --mode test  
```  

## Usage  

### Test Mode (5 problems)  
```bash  
python main.py --mode test  
```  

### Single Problem  
```bash  
python main.py --mode single --problem "Prove that for all real x, x + 0 = x"  
```  

### Batch Mode (custom problems)  
```bash  
python main.py --mode batch --test-file my_problems.json  
```  

## Output  

Results are saved in `results/` directory with:  
- Full pipeline execution logs  
- Iteration details with errors  
- Compilation and proof success rates  
- Timing information  
- Metrics summary  

## Metrics  

The system tracks:  
- **Compilation Success Rate**: % of problems that compile  
- **Proof Success Rate**: % that are automatically proven  
- **First-Attempt Success Rate**: % successful without iteration  
- **Average Iterations**: Mean iterations for successful problems  
- **Success by Iteration**: Breakdown per iteration  

## Architecture  

```  
Natural Language Problem  
         ↓  
    RAG Retrieval (Top-3 similar from miniF2F)  
         ↓  
    LLM Translation (with few-shot examples)  
         ↓  
    Lean 4 Compilation  
         ↓  
   [Success?] → Yes → Proof Attempt → Results  
         ↓ No  
    Extract Errors  
         ↓  
    Refine (max 3 iterations)  
```  

## Future Enhancements  

- [ ] Combined tactic attempts  
- [ ] LLM-based whole-proof generation  
- [ ] Adaptive iteration stopping  
- [ ] Web interface  
- [ ] Larger dataset support  
```  

**Agent Task**: Create this README with clear usage instructions.  

---  

## ✅ Step 13: Verification & Testing  

### Task 13.1: Create `tests/test_pipeline.py`  

```python  
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
```  

**Agent Task**: Create this test file to verify components work before running the full pipeline.  

---  

## 🎬 Step 14: Execution Instructions  

### For the Code Agent: Execute in this order  

1. **Create directory structure**:  
```bash  
mkdir -p rag-autoformalization/{src,data/embeddings,prompts,tests,results}  
cd rag-autoformalization  
```  

2. **Create all configuration files**:  
   - `requirements.txt`  
   - `config.py`  
   - `.env.example`  

3. **Create all source files** in order:  
   - `src/utils.py`  
   - `src/rag_module.py`  
   - `src/llm_client.py`  
   - `src/lean_interface.py`  
   - `src/proof_tactics.py`  
   - `src/pipeline.py`  

4. **Create prompt templates**:  
   - `prompts/initial_translation.txt`  
   - `prompts/refinement_with_errors.txt`  
   - `prompts/proof_generation.txt`  

5. **Create data files**:  
   - `data/minif2f_statements.json` (20 sample problems)  
   - `tests/test_problems.json` (5 test problems)  

6. **Create entry points**:  
   - `main.py`  
   - `tests/test_pipeline.py`  
   - `README.md`  

7. **Install and setup**:  
```bash  
pip install -r requirements.txt  
cp .env.example .env  
# IMPORTANT: Edit .env and add OPENAI_API_KEY  
```  

8. **Run verification tests**:  
```bash  
python tests/test_pipeline.py  
```  

9. **Run MVP**:  
```bash  
python main.py --mode test  
```  

10. **Verify output**:  
   - Check `results/` directory for output files  
   - Verify metrics are calculated  
   - Ensure at least some problems compile successfully  

---  

## 🎯 Success Criteria Checklist  

The MVP is complete when:  

- [ ] All files are created without errors  
- [ ] Dependencies install successfully  
- [ ] Component tests pass  
- [ ] RAG retrieval returns similar examples  
- [ ] LLM generates Lean 4 statements  
- [ ] Lean compiler runs and returns feedback  
- [ ] Iteration loop executes (max 3 times)  
- [ ] Proof tactics are attempted  
- [ ] Results are saved to JSON  
- [ ] Metrics are calculated and displayed  
- [ ] At least 1 out of 5 test problems succeeds  

---  

## 📊 Expected MVP Performance  

Based on research, you should see approximately:  

| Metric | Expected Range |  
|--------|----------------|  
| Compilation Success Rate | 40-60% |  
| Proof Success Rate | 10-20% |  
| First-Attempt Success | 20-35% |  
| Avg Iterations | 1.5-2.5 |  
| Time per Problem | 15-45 seconds |  

---  

## 🔧 Troubleshooting Guide  

**If RAG fails**:  
- Check `data/minif2f_statements.json` exists  
- Verify sentence-transformers installation  
- Delete embeddings cache and rebuild  

**If Lean compilation fails**:  
- Verify Lean 4 is installed: `lean --version`  
- Check LEAN_PATH in config  
- Test with simple code manually  

**If LLM fails**:  
- Verify OPENAI_API_KEY in `.env`  
- Check API quota/billing  
- Try with temperature=0.3 for more consistent output  

**If nothing works**:  
- Run `python tests/test_pipeline.py` to isolate issue  
- Check each component individually  
- Verify all imports are correct  

---  

## 📝 Final Notes for Agent  

This MVP demonstrates:  
1. ✅ **RAG integration** with semantic retrieval  
2. ✅ **Iterative refinement** using compiler feedback (PDA-style)  
3. ✅ **Proof attempts** with basic tactics  
4. ✅ **Metrics and evaluation** infrastructure  
5. ✅ **Modular architecture** for easy extension  

After MVP works, you can enhance with:  
- More sophisticated RAG (hybrid retrieval)  
- LLM-based proof generation  
- Web interface (Gradio/Streamlit)  
- Larger test datasets  
- Caching and optimization  
- Better error parsing  
- Proof-step generation (instead of whole-proof)  

**The MVP gives you a working foundation to build upon!** 🚀