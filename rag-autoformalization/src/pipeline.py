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
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

class AutoformalizationPipeline:
    def __init__(self, disable_manual_proof: bool = False):
        """
        Initialize the pipeline with all components
        
        Args:
            disable_manual_proof: If True, skip manual tactics in proof generation
        """
        self.rag = MathLibRAG()
        self.llm = LLMClient()
        self.lean = LeanInterface()
        self.proof_tactics = ProofTactics(llm_client=self.llm)
        self.console = Console()
        self.disable_manual_proof = disable_manual_proof
    
    def process(self, natural_language: str) -> Dict:
        """
        Process a natural language problem through the full pipeline
        
        Args:
            natural_language: Mathematical problem in natural language
            
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        results = {
            "natural_language": natural_language,
            "iterations": [],
            "final_statement": None,
            "compilation_success": False,
            "proof_success": False,
            "proof_tactic": None,
            "total_iterations": 0,
            "total_time": 0,
            "rag_examples": [],
        }
        
        # Step 1: RAG Retrieval
        self.console.print()
        self.console.print(Panel(
            f"[bold blue]RAG-Enhanced Autoformalization Pipeline[/bold blue]\n"
            f"[dim]{natural_language[:80]}{'...' if len(natural_language) > 80 else ''}[/dim]",
            border_style="blue"
        ))
        self.console.print()
        self.console.print(Panel("[bold cyan][1/4] RAG Retrieval[/bold cyan]", border_style="cyan"))
        
        with self.console.status("[bold green]Loading embeddings and searching...", spinner="dots"):
            similar_statements = self.rag.retrieve(natural_language)
        
        results["rag_examples"] = similar_statements
        few_shot_prompt = self.rag.format_for_prompt(similar_statements)
        proof_examples = self.rag.format_proof_examples(similar_statements)
        
        # Display RAG examples in a table
        if similar_statements:
            rag_table = Table(title=f"Retrieved {len(similar_statements)} Similar Examples", show_header=True, header_style="bold magenta")
            rag_table.add_column("Rank", style="cyan", width=6)
            rag_table.add_column("Natural Language", style="white", width=50)
            rag_table.add_column("Similarity", style="green", justify="right", width=10)
            
            for idx, example in enumerate(similar_statements, 1):
                nl_text = example.get('natural_language', '')[:60] + ('...' if len(example.get('natural_language', '')) > 60 else '')
                similarity = example.get('similarity_score', 0)
                rag_table.add_row(str(idx), nl_text, f"{similarity:.3f}")
            
            self.console.print(rag_table)
        else:
            self.console.print("[yellow]⚠ No similar examples found[/yellow]")
        
        # Step 2: Iterative Formalization with Compiler Feedback
        self.console.print()
        self.console.print(Panel("[bold yellow][2/4] Iterative Formalization[/bold yellow]", border_style="yellow"))
        
        previous_attempt = None
        compiler_errors = None
        
        for iteration in range(config.MAX_ITERATIONS):
            self.console.print(f"\n[bold]Iteration {iteration + 1}/{config.MAX_ITERATIONS}[/bold]")
            
            # Generate formal statement
            with self.console.status("[bold green]Generating Lean 4 statement...", spinner="dots"):
                formal_statement = self.llm.translate_to_lean(
                    natural_language=natural_language,
                    few_shot_examples=few_shot_prompt,
                    previous_attempt=previous_attempt,
                    compiler_errors=compiler_errors
                )
            
            if formal_statement is None:
                self.console.print("[red]✗ LLM generation failed[/red]")
                break
            
            # Display generated code
            code_syntax = Syntax(formal_statement, "lean", theme="monokai", line_numbers=True)
            self.console.print(Panel(code_syntax, title=f"[bold]Generated Code (Iteration {iteration + 1})[/bold]", border_style="blue"))
            
            # Compile and get feedback
            with self.console.status("[bold yellow]Compiling with Lean 4...", spinner="dots"):
                compile_result = self.lean.compile(
                    formal_statement,
                    iteration=iteration
                )
            
            iteration_data = {
                "iteration": iteration + 1,
                "formal_statement": formal_statement,
                "compilation_success": compile_result["success"],
                "errors": compile_result["errors"],
                "error_categories": compile_result["error_categories"],
                "lean_file_path": compile_result.get("file_path"),
                "lean_filename": compile_result.get("filename")
            }
            results["iterations"].append(iteration_data)
            
            if compile_result["success"]:
                self.console.print("[bold green]✓ Compilation successful![/bold green]")
                results["final_statement"] = formal_statement
                results["compilation_success"] = True
                results["total_iterations"] = iteration + 1
                break
            else:
                self.console.print("[bold red]✗ Compilation failed[/bold red]")
                
                # Show errors
                if compile_result["errors"]:
                    error_text = "\n".join(compile_result["errors"][:5])
                    self.console.print(Panel(
                        error_text,
                        title="[bold red]Compilation Errors[/bold red]",
                        border_style="red"
                    ))
                
                if compile_result["error_categories"]:
                    error_types = ", ".join(compile_result["error_categories"].keys())
                    self.console.print(f"[dim]Error types: {error_types}[/dim]")
                
                # Prepare for next iteration
                previous_attempt = formal_statement
                compiler_errors = "\n".join(compile_result["errors"][:5])  # Top 5 errors
                
                if iteration < config.MAX_ITERATIONS - 1:
                    self.console.print("[yellow]→ Refining with compiler feedback...[/yellow]")
        
        # Step 3: Proof Attempt (if compilation successful)
        if results["compilation_success"]:
            self.console.print()
            self.console.print(Panel("[bold magenta][3/4] Automated Proof Generation[/bold magenta]", border_style="magenta"))
            
            if self.disable_manual_proof:
                self.console.print("[yellow]⚠ Manual tactics disabled, using LLM only[/yellow]")
            
            proof_result = self.proof_tactics.attempt_proof(
                results["final_statement"],
                proof_examples=proof_examples,
                disable_manual=self.disable_manual_proof
            )
            
            results["proof_success"] = proof_result["proved"]
            results["proof_tactic"] = proof_result["tactic"]
            results["proof_attempts"] = proof_result["attempts"]
            
            if proof_result["proved"]:
                self.console.print(f"[bold green]✓ Proof found using tactic: [cyan]{proof_result['tactic']}[/cyan][/bold green]")
                
                # Display final proof code
                final_code = Syntax(proof_result["proof_code"], "lean", theme="monokai", line_numbers=True)
                self.console.print(Panel(
                    final_code,
                    title="[bold green]Final Proof[/bold green]",
                    border_style="green"
                ))
                results["final_statement"] = proof_result["proof_code"]
            else:
                self.console.print(f"[bold red]✗ No proof found (tried {len(proof_result['attempts'])} tactics)[/bold red]")
        else:
            self.console.print()
            self.console.print(Panel("[dim][3/4] Skipping proof attempt (compilation failed)[/dim]", border_style="dim"))
        
        # Step 4: Finalize results
        results["total_time"] = time.time() - start_time
        self.console.print()
        self.console.print(Panel(
            f"[bold green][4/4] Complete![/bold green]\n"
            f"Total time: [cyan]{results['total_time']:.2f}s[/cyan]\n"
            f"Compilation: {'[green]✓ Success[/green]' if results['compilation_success'] else '[red]✗ Failed[/red]'}\n"
            f"Proof: {'[green]✓ Success[/green]' if results['proof_success'] else '[red]✗ Failed[/red]'}",
            border_style="green"
        ))
        
        return results
    
    def batch_process(self, problems: list) -> list:
        """Process multiple problems"""
        all_results = []
        for i, problem in enumerate(problems, 1):
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]Problem {i}/{len(problems)}[/bold cyan]",
                border_style="cyan"
            ))
            
            result = self.process(
                natural_language=problem.get("natural_language", problem)
            )
            all_results.append(result)
        
        return all_results