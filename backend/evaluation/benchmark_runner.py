import os
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from backend.agents.orchestrator.agent import orchestrator
from backend.evaluation.ragas_eval import evaluator

logger = logging.getLogger("rip.evaluation.runner")
console = Console()

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "evals", "benchmark.json"
)

async def run_eval_benchmark(sample_size: int = 5) -> Dict[str, Any]:
    """Load benchmark cases, run through RAG pipeline, evaluate scores, and aggregate."""
    if not os.path.exists(BENCHMARK_PATH):
        raise FileNotFoundError(f"Benchmark file not found at: {BENCHMARK_PATH}")
        
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
        
    # Limit sample size to prevent excessive API costs/times
    cases = all_cases[:min(len(all_cases), sample_size)]
    
    console.print(f"[bold indigo]Starting RAG Evaluation Benchmark. Sample Size: {len(cases)}[/bold indigo]")
    
    results = []
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_recall = 0.0
    total_latency = 0.0
    
    # Initialize rich table
    table = Table(title="RAG Evaluation Results", show_lines=True)
    table.add_column("Query ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Latency (s)", style="green")
    table.add_column("Faithfulness", style="blue")
    table.add_column("Relevancy", style="blue")
    table.add_column("Recall", style="blue")
    
    for idx, case in enumerate(cases):
        query = case["query"]
        ground_truth = case["ground_truth_answer"]
        query_type = case["query_type"]
        
        t_start = time.time()
        
        # 1. Run through full multi-agent pipeline
        session_id = f"benchmark-session-{idx}"
        try:
            pipeline_res = await orchestrator.run_query(query, session_id=session_id)
            answer = pipeline_res.get("answer", "")
            citations = pipeline_res.get("citations", [])
            context_texts = [c.get("snippet", "") for c in citations]
            
            # 2. Evaluate scores
            faithfulness = await evaluator.score_faithfulness(answer, context_texts)
            relevancy = await evaluator.score_answer_relevancy(query, answer)
            recall = await evaluator.score_context_recall(ground_truth, context_texts)
        except Exception as e:
            logger.error(f"Benchmark error for query '{query}': {e}")
            faithfulness, relevancy, recall = 0.0, 0.0, 0.0
            answer = ""
            
        latency = time.time() - t_start
        
        total_faithfulness += faithfulness
        total_relevancy += relevancy
        total_recall += recall
        total_latency += latency
        
        results.append({
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "latency_seconds": latency,
            "scores": {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "recall": recall,
                "overall": (faithfulness + relevancy + recall) / 3.0
            }
        })
        
        table.add_row(
            f"Case #{idx + 1}",
            query_type,
            f"{latency:.2f}s",
            f"{faithfulness:.2f}",
            f"{relevancy:.2f}",
            f"{recall:.2f}"
        )
        
        # Tiny sleep to avoid API rate limiting
        await asyncio.sleep(0.5)
        
    # Aggregate stats
    count = len(cases)
    avg_f = total_faithfulness / count if count > 0 else 0
    avg_rel = total_relevancy / count if count > 0 else 0
    avg_rec = total_recall / count if count > 0 else 0
    avg_lat = total_latency / count if count > 0 else 0
    
    # Save outcomes to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(BENCHMARK_PATH)
    out_file = os.path.join(out_dir, f"results_{timestamp}.json")
    
    summary = {
        "timestamp": timestamp,
        "sample_size": count,
        "averages": {
            "faithfulness": avg_f,
            "relevancy": avg_rel,
            "recall": avg_rec,
            "latency_seconds": avg_lat
        },
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    console.print(table)
    console.print(f"\n[bold green]Averages:[/bold green]")
    console.print(f"Latency: {avg_lat:.2f}s")
    console.print(f"Faithfulness: {avg_f:.2f}")
    console.print(f"Relevancy: {avg_rel:.2f}")
    console.print(f"Recall: {avg_rec:.2f}")
    console.print(f"Results logged to: [cyan]{out_file}[/cyan]\n")
    
    return summary

if __name__ == "__main__":
    import sys
    async def main():
        # Retrieve count argument
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        res = await run_eval_benchmark(sample_size=count)
        
        # Performance Threshold Gates
        # Fail CI if score averages are low
        if res["averages"]["faithfulness"] < 0.75 or res["averages"]["relevancy"] < 0.70:
            console.print("[bold red]CI GATE FAILED: Score averages below target thresholds (Faithfulness < 0.75 or Relevancy < 0.70)[/bold red]")
            sys.exit(1)
        else:
            console.print("[bold green]CI GATE PASSED![/bold green]")
            sys.exit(0)
            
    asyncio.run(main())
