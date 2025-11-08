# scripts/run_full_benchmark.py
from __future__ import annotations

# --- ensure project root is on sys.path ---
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path

from src.core.llm.llm_orchestrator import LLMOrchestrator
from src.core.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.core.evaluation.evaluation_visualizer import EvaluationVisualizer, VizConfig
from src.core.evaluation.evaluation_table_exporter import EvaluationTableExporter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full automated benchmark: LLM → Evaluation → Visualization → Table export."
    )
    parser.add_argument("--prompt_file", type=str, default="data/prompts/benchmark_prompts.txt",
                        help="Path to file containing one prompt per line.")
    parser.add_argument("--num_prompts", type=int, default=100,
                        help="Number of prompts to process.")
    parser.add_argument("--logs_dir", type=str, default="data/logs",
                        help="Directory for raw LLM logs.")
    parser.add_argument("--eval_dir", type=str, default="data/eval_logs",
                        help="Directory for evaluation results.")
    parser.add_argument("--charts_dir", type=str, default="data/eval_charts",
                        help="Directory for plots, tables, and reports.")
    parser.add_argument("--k", type=int, default=10,
                        help="NDCG cutoff parameter k.")
    parser.add_argument("--bootstrap_iters", type=int, default=2000,
                        help="Bootstrap iterations for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic reproducibility.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Deterministic randomness for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    # Ensure directories exist
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.charts_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("=== Phase 1: LLM Generation ===")
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts = [ln.strip() for ln in prompt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prompts = prompts[: args.num_prompts]
    print(f"Loaded {len(prompts)} prompts for benchmark.\n")

    llm_orch = LLMOrchestrator()
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] {prompt}")
        try:
            query_obj = {"processed_query": prompt, "intent": "conceptual"}
            _ = llm_orch.process_query(query_obj)
        except Exception as e:
            (Path(args.logs_dir) / f"error_{idx}.json").write_text(
                json.dumps({"prompt": prompt, "error": str(e)}, indent=2),
                encoding="utf-8",
            )
    llm_orch.close()
    print("LLM benchmark phase completed.\n")

    # ------------------------------------------------------------------
    print("=== Phase 2: Evaluation Metrics ===")
    evaluator = EvaluationOrchestrator(output_dir=args.eval_dir, k=args.k)
    summary_eval = evaluator.evaluate_batch_from_logs(logs_dir=args.logs_dir)
    print(json.dumps(summary_eval, indent=2))

    # ------------------------------------------------------------------
    print("=== Phase 3: Visualization & Statistical Analysis ===")
    cfg = VizConfig(
        logs_dir=args.eval_dir,
        out_dir=args.charts_dir,
        bootstrap_iters=args.bootstrap_iters
    )
    viz = EvaluationVisualizer(cfg)
    summary_viz = viz.run_all()
    print(json.dumps(summary_viz, indent=2))

    # ------------------------------------------------------------------
    print("=== Phase 4: Table Export ===")
    exporter = EvaluationTableExporter(charts_dir=args.charts_dir)
    export_paths = exporter.export()
    print("Generated tables:")
    for k, v in export_paths.items():
        if k != "rows":
            print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 🔧 Phase 5: PDF Report Generation (unique per n and seed)
    print("=== Phase 5: PDF Report Generation ===")
    from src.core.evaluation.report_builder import ReportBuilder
    rb = ReportBuilder(charts_dir=args.charts_dir)
    pdf_path = rb.build(custom_name=f"benchmark_report_n{args.num_prompts}_seed{args.seed}.pdf")
    print(f"Generated PDF report: {pdf_path}")

    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n=== BENCHMARK COMPLETED ===")
    print(f"Processed {len(prompts)} prompts in {elapsed/60:.1f} minutes total.")
    print(f"All outputs available in:\n  {args.eval_dir}\n  {args.charts_dir}")


if __name__ == "__main__":
    main()
