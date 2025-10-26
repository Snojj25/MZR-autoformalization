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