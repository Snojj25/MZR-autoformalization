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
    LEAN_TIMEOUT = int(os.getenv("LEAN_TIMEOUT", "30"))  # Increased from 10 to 20 seconds for complex proof
    
    # Pipeline settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    MAX_PROOF_ITERATIONS = int(os.getenv("MAX_PROOF_ITERATIONS", "3"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    
    # Paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    PROMPTS_DIR = "prompts"
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings", "f2f_embeddings.pkl")
    MINIF2F_PATH = os.path.join(DATA_DIR, "f2f_statements.json")
    
    # Lean examples directory (for compilation with Mathlib)
    # Get the project root (assumes config.py is in rag-autoformalization/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LEAN_EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "lean-examples")
    LEAN_EXAMPLES_SRC_DIR = os.path.join(LEAN_EXAMPLES_DIR, "LeanExamples")
    
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