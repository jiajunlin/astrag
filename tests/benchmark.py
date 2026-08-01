import time
import ollama
from astrag.pipeline import CodebaseMemory
from astrag.parsing import approx_tokens

def run_benchmark():
    # Configure your test parameters
    MODEL_NAME = "phi3.5"  # Replace with any Ollama model you have pulled
    QUERY = """You are an expert frontend engineer specializing in React, TypeScript, and accessible UI design.

Build a production-ready component for [Insert Feature/Component Name, e.g., a searchable data table with pagination and sorting].

Core Requirements:
Tech Stack: React (functional components and hooks), strict TypeScript (no any types), and Tailwind CSS for utility-first styling.

Accessibility (WCAG 2.1 AA): Ensure full keyboard navigation (Tab, Arrow keys, Enter/Space), proper ARIA attributes, semantic HTML elements, and clear focus states.

State Management & Edge Cases: Explicitly handle loading, empty, and error states. Optimize re-renders using useMemo or useCallback where appropriate.

Maintainability: Keep the component modular, well-commented, and separated into logical sub-components if it exceeds 150 lines.

Deliverables:
Complete TypeScript interfaces and types for all props and data models.

The full, self-contained component implementation.

A brief usage example showing how to mount the component and handle callback events."""
    TARGET_BUDGET = 500              # Set your target astrag token budget

    print("==================================================")
    print(" 1. INDEXING REPOSITORY                           ")
    print("==================================================")
    mem = CodebaseMemory().index_repo(".")
    num_files = len({c.file for c in mem.chunks})
    print(f"Indexed {len(mem.chunks)} chunks across {num_files} files.")

    # -------------------------------------------------------------------------
    # A. Baseline: Raw Context (Simulating standard LLM file dumping)
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print(" 2. PREPARING BASELINE CONTEXT (RAW FILES)        ")
    print("==================================================")
    # Grab the full source text of the top 5 files to simulate a full file dump
    raw_blocks = []
    for chunk in mem.chunks[:5]:
        raw_blocks.append(f"// File: {chunk.file}\n{chunk.source}")
    baseline_context = "\n\n".join(raw_blocks)
    baseline_prompt = f"Codebase Context:\n{baseline_context}\n\nTask: {QUERY}"
    
    # -------------------------------------------------------------------------
    # B. astrag: 0/1 Knapsack Compressed Context
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print(f" 3. PREPARING ASTRAG CONTEXT (BUDGET: {TARGET_BUDGET})")
    print("==================================================")
    ctx = mem.build_context(QUERY, token_budget=TARGET_BUDGET, fetch_bodies=2)
    astrag_prompt = f"{ctx.text}\n\nTask: {QUERY}"

    # -------------------------------------------------------------------------
    # C. Run Ollama Benchmark: Baseline
    # -------------------------------------------------------------------------
    print("\nRunning Ollama evaluation on BASELINE context...")
    res_base = ollama.generate(
        model=MODEL_NAME,
        prompt=baseline_prompt,
        options={"num_predict": 1000}  # Cap output tokens to focus on prompt processing time
    )

    # -------------------------------------------------------------------------
    # D. Run Ollama Benchmark: astrag
    # -------------------------------------------------------------------------
    print("Running Ollama evaluation on ASTRAG context...")
    res_astrag = ollama.generate(
        model=MODEL_NAME,
        prompt=astrag_prompt,
        options={"num_predict": 1000}
    )

    # -------------------------------------------------------------------------
    # E. Extract and Calculate Metrics
    # -------------------------------------------------------------------------
    base_p_tokens = res_base.get("prompt_eval_count", approx_tokens(baseline_prompt))
    base_p_time_ms = res_base.get("prompt_eval_duration", 0) / 1e6  # Convert ns to ms

    astrag_p_tokens = res_astrag.get("prompt_eval_count", approx_tokens(astrag_prompt))
    astrag_p_time_ms = res_astrag.get("prompt_eval_duration", 0) / 1e6  # Convert ns to ms

    # Mathematical calculations
    savings_pct = (1.0 - (astrag_p_tokens / max(1, base_p_tokens))) * 100
    compression_ratio = base_p_tokens / max(1, astrag_p_tokens)
    speedup_factor = base_p_time_ms / max(1e-3, astrag_p_time_ms)

    # -------------------------------------------------------------------------
    # F. Display Results Scorecard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("           ASTRAG VS BASELINE BENCHMARK RESULTS     ")
    print("=" * 55)
    print(f"Model Tested:             {MODEL_NAME}")
    print(f"Target Token Budget:      {TARGET_BUDGET} tokens\n")
    
    print(f"Baseline Prompt Tokens:   {base_p_tokens:,} tokens")
    print(f"Baseline Ingestion Time:  {base_p_time_ms:.2f} ms\n")

    print(f"astrag Prompt Tokens:     {astrag_p_tokens:,} tokens")
    print(f"astrag Ingestion Time:    {astrag_p_time_ms:.2f} ms\n")

    print("-" * 55)
    print(f"Token Reduction:          {savings_pct:.1f}% FEWER TOKENS")
    print(f"Compression Ratio:        {compression_ratio:.2f}x COMPRESSION")
    print(f"Prompt Processing Speed:  {speedup_factor:.2f}x FASTER TIME-TO-FIRST-TOKEN")
    print("=" * 55)
    # Print what Baseline produced
    print("\n--- BASELINE LLM OUTPUT ---")
    print(res_base.get("response", ""))

    # Print what astrag produced
    print("\n--- ASTRAG LLM OUTPUT ---")
    print(res_astrag.get("response", ""))

if __name__ == "__main__":
    run_benchmark()