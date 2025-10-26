"""
LLM Client for interacting with OpenAI API
"""
from openai import OpenAI

from typing import Optional
from config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

class LLMClient:
    def __init__(self):
        """Initialize OpenAI client"""
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
            response = client.chat.completions.create(model=self.model,
            messages=messages,
            # temperature=self.temperature,
            max_completion_tokens=1000)
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