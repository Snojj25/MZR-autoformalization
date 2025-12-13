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
        """Load F2F statements from JSON file"""
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
            formatted += f"Lean Statement: {stmt['lean_statement']}\n"
            # Include proof if it exists
            if stmt.get('has_proof', False) and stmt.get('proof') and stmt['proof'] != 'sorry':
                formatted += f"Proof: {stmt['proof']}\n"
            formatted += "\n"
        return formatted
    
    def format_proof_examples(self, similar_statements: List[Dict]) -> str:
        """Format retrieved statements with proofs as examples for proof generation"""
        # Filter to only statements that have proofs
        statements_with_proofs = [
            stmt for stmt in similar_statements 
            if stmt.get('has_proof', False) and stmt.get('proof') and stmt['proof'] != 'sorry'
        ]
        
        if not statements_with_proofs:
            return ""
        
        formatted = "Here are similar theorems with their proofs as examples:\n\n"
        for i, stmt in enumerate(statements_with_proofs, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Theorem: {stmt['lean_statement']}\n"
            formatted += f"Proof: {stmt['proof']}\n\n"
        return formatted