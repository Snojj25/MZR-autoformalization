# RAG-Enhanced Autoformalization Architecture

## Overview

This system automatically translates natural language mathematical problems into formal Lean 4 statements using RAG (Retrieval-Augmented Generation) and iterative refinement with compiler feedback. It then attempts to generate proofs using basic Lean tactics.

## High-Level Flow

```
Natural Language Problem
    ↓
1. RAG Retrieval (Find similar examples)
    ↓
2. LLM Translation (Generate Lean code with few-shot examples)
    ↓
3. Lean Compilation (Validate syntax and types)
    ↓
4a. If errors → Refine with compiler feedback → Back to step 2
4b. If success → Attempt proof
    ↓
5. Success Metric Tracking
```

## Architecture Components

### 1. **AutoformalizationPipeline** (`src/pipeline.py`)
The orchestrator that coordinates all components.

**Key Responsibilities:**
- Initialize all sub-systems (RAG, LLM, Lean, Proof)
- Execute the 4-step process for each problem
- Track iterations and results
- Batch process multiple problems

**Main Flow:**
1. **RAG Retrieval** - Get similar problems from miniF2F dataset
2. **Iterative Formalization** - Generate and refine Lean code (max 3 iterations)
3. **Proof Attempt** - Try basic tactics if compilation succeeds
4. **Result Tracking** - Save complete execution history

### 2. **MathLibRAG** (`src/rag_module.py`)
Retrieval-Augmented Generation module for finding similar problems.

**How it works:**
- Uses `SentenceTransformer` (all-MiniLM-L6-v2) to encode problems as vectors
- Builds FAISS index for fast similarity search
- Caches embeddings to disk for faster startup
- Retrieves top-K most similar problems for few-shot learning

**Key Methods:**
- `retrieve(query, base) → List[Dict]` - Find similar problems
- `format_for_prompt(statements) → str` - Create few-shot examples for LLM

**Data:** Loads from `data/minif2f_statements.json` (pre-processed miniF2F theorems)

### 3. **LLMClient** (`src/llm_client.py`)
Interface to OpenAI's GPT models for code generation.

**How it works:**
- Two prompt templates:
  - `initial_translation.txt` - For first attempt
  - `refinement_with_errors.txt` - For error correction
- Includes few-shot examples from RAG
- Uses previous attempt + compiler errors for refinement

**Key Method:**
- `translate_to_lean(nl_problem, examples, prev_attempt, errors) → str`

### 4. **LeanInterface** (`src/lean_interface.py`)
Wraps the Lean 4 compiler for validation and error extraction.

**How it works:**
- Creates temporary `.lean` files
- Executes `lean` compiler via subprocess
- Parses and categorizes errors (syntax, type, semantic, unknown)
- Returns structured compilation results

**Error Categories:**
- **Syntax**: unexpected, expected, missing tokens
- **Type**: type mismatches
- **Semantic**: unknown identifiers, not in scope

### 5. **ProofTactics** (`src/proof_tactics.py`)
Attempts automatic proofs using basic Lean tactics.

**How it works:**
- Tries tactics sequentially: `simp`, `ring`, `linarith`, `norm_num`, `omega`, `aesop`
- Replaces `:= by sorry` with `:= by [tactic]`
- Compiles each attempt to verify success
- Returns first successful proof

**Tactics Used:**
- `simp` - Simplification
- `ring` - Ring/field proofs
- `linarith` - Linear arithmetic
- `norm_num` - Number normalization
- `omega` - Integer linear arithmetic
- `aesop` - Auto-tactic

### 6. **Configuration** (`config.py`)
Centralized settings from environment variables.

**Key Settings:**
- `MAX_ITERATIONS = 3` - Max refinement attempts
- `TOP_K_RETRIEVAL = 3` - Number of similar examples
- `LEAN_TIMEOUT = 10s` - Compiler timeout
- `OPENAI_MODEL = gpt-4` - LLM model

### 7. **Utilities** (`src/utils.py`)
Metrics calculation and result formatting.

**Key Functions:**
- `calculate_metrics(results)` - Aggregate statistics
- `print_summary(results)` - Console output
- `save_results(results, filename)` - JSON export

**Metrics Tracked:**
- Compilation success rate
- Proof success rate
- First-attempt success rate
- Average iterations to success
- Average time per problem

## Data Flow

### Initial Translation
```
Input: "If x and y are positive, then x + y > 0"
    ↓
RAG: Finds 3 similar miniF2F problems
    ↓
LLM Prompt: System message + Few-shot examples + Problem
    ↓
Output: import Mathlib\n\ntheorem pos_sum_pos (x y : ℝ) 
        (hx : x > 0) (hy : y > 0) : x + y > 0 := by sorry
    ↓
Lean Compile: ✓ Success
```

### Iterative Refinement (if errors)
```
LLM Output: [contains errors]
    ↓
Lean Compile: ✗ [errors: "unknown identifier 'ℝ'"]
    ↓
RAG: [Same examples]
    ↓
LLM Prompt: Original problem + Examples + Previous attempt + Errors
    ↓
LLM Output: [fixed code]
    ↓
Lean Compile: ✓ Success (or retry)
```

### Proof Generation (if compilation succeeds)
```
Compiled Statement: theorem pos_sum_pos ... := by sorry
    ↓
Tactic 1: := by simp → ✗
    ↓
Tactic 2: := by ring → ✗
    ↓
Tactic 3: := by linarith → ✓ Success!
    ↓
Final: theorem pos_sum_pos ... := by linarith
```

## Key Design Decisions

1. **Iterative Refinement**: Max 3 attempts to fix errors, preventing infinite loops
2. **RAG Enhancement**: Few-shot examples improve translation quality
3. **Error Categorization**: Helps LLM understand error types
4. **Compilation Validation**: Ensures generated code is syntactically and type-correct
5. **Simple Proof Tactics**: MVP uses basic tactics; can be extended
6. **Result Tracking**: Complete history for analysis and debugging

## Usage

```bash
# Run on test problems
python main.py --mode test

# Single problem
python main.py --mode single --problem "If x > 0 then x^2 > 0"

# Custom test file
python main.py --mode test --test-file tests/custom_problems.json
```

## Results Format

Each result contains:
- `problem_id`: Identifier
- `natural_language`: Original problem
- `rag_examples`: Retrieved similar problems (with similarity scores)
- `iterations`: List of refinement attempts with errors
- `final_statement`: Successful Lean code
- `compilation_success`: Boolean
- `proof_success`: Boolean
- `proof_tactic`: Successful tactic name
- `total_iterations`: Number of attempts needed
- `total_time`: Execution time in seconds

