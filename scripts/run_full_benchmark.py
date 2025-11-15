# scripts/run_full_benchmark.py
from __future__ import annotations

# Ensure project root on sys.path
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path
import shutil

from src.core.llm.llm_orchestrator import LLMOrchestrator
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.core.evaluation.evaluation_visualizer import EvaluationVisualizer, VizConfig
from src.core.evaluation.evaluation_table_exporter import EvaluationTableExporter


def _count_files(dir_path: Path, pattern: str) -> int:
    # Count files using a glob pattern
    return sum(1 for _ in dir_path.glob(pattern))


def _load_json(path: Path) -> dict | None:
    # Safe JSON loading
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_directory(path: Path) -> None:
    # Delete all contents of a directory and recreate it
    if path.exists():
        for f in path.glob("*"):
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full automated benchmark: LLM → Evaluation → Visualization → Table export."
    )
    parser.add_argument("--prompt_file", type=str, default="data/prompts/benchmark_prompts.txt")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--logs_dir", type=str, default="data/logs")
    parser.add_argument("--eval_dir", type=str, default="data/eval_logs")
    parser.add_argument("--charts_dir", type=str, default="data/eval_charts")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--intent",
        type=str,
        default=None,
        choices=["conceptual", "chronological", "analytical", "comparative"],
        help="Optional fixed intent; if omitted, automatic classification is used."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    logs_dir = Path(args.logs_dir)
    eval_dir = Path(args.eval_dir)
    charts_dir = Path(args.charts_dir)

    # Clean deterministic directories for this run
    _clean_directory(logs_dir)
    _clean_directory(eval_dir)
    _clean_directory(charts_dir)

    print("=== Benchmark configuration ===")
    print(f"prompt_file  : {Path(args.prompt_file).resolve()}")
    print(f"num_prompts  : {args.num_prompts}")
    print(f"logs_dir     : {logs_dir.resolve()}")
    print(f"eval_dir     : {eval_dir.resolve()}")
    print(f"charts_dir   : {charts_dir.resolve()}")
    print(f"k            : {args.k}")
    print(f"boot iters   : {args.bootstrap_iters}")
    print(f"seed         : {args.seed}")
    print(f"intent       : {args.intent or 'auto'}")
    print()

    # ------------------------------------------------------------------
    print("=== Phase 1: LLM Generation ===")
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts = [
        ln.strip()
        for ln in prompt_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    prompts = prompts[: args.num_prompts]
    print(f"Loaded {len(prompts)} prompts.\n")

    before_logs = _count_files(logs_dir, "llm_*.json")

    llm_orch = LLMOrchestrator(logs_dir=str(logs_dir))
    prompt_orch = PromptOrchestrator()

    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] {prompt}")

        try:
            if args.intent:
                query_obj = {"processed_query": prompt.strip(), "intent": args.intent}
            else:
                pre = prompt_orch.preprocessor.process(prompt)
                intent = pre["intent"]
                refined = prompt_orch.prompt_builder.reformulate_query(
                    pre["processed_query"], intent
                )
                query_obj = {"processed_query": refined, "intent": intent}
                prompt_orch.logger.info(
                    f"Auto intent='{intent}' → refined query='{refined}'"
                )

            _ = llm_orch.process_query(query_obj)

        except Exception as e:
            (logs_dir / f"error_{idx}.json").write_text(
                json.dumps({"prompt": prompt, "error": str(e)}, indent=2),
                encoding="utf-8"
            )

    if hasattr(llm_orch, "close"):
        try:
            llm_orch.close()
        except Exception:
            pass

    after_logs = _count_files(logs_dir, "llm_*.json")
    produced = max(0, after_logs - before_logs)
    print(f"LLM phase completed. Produced {produced} valid llm_*.json logs.\n")

    if produced == 0:
        print("No LLM outputs. Stopping.")
        return

    # ------------------------------------------------------------------
    print("=== Phase 2: Evaluation ===")

    # Derive model_name from logs_dir, e.g. data/logs_phi3_4b → phi3_4b
    model_name = Path(args.logs_dir).name.replace("logs_", "") or "benchmark"

    evaluator = EvaluationOrchestrator(
        base_output_dir=str(eval_dir),
        model_name=model_name,
        k=args.k
    )

    summary_eval = evaluator.evaluate_batch_from_logs(logs_dir=str(logs_dir))
    print(json.dumps(summary_eval, indent=2))

    if not summary_eval or int(summary_eval.get("evaluated_files", 0)) == 0:
        print("No evaluatable data. Stopping before visualization.")
        return

    # ------------------------------------------------------------------
    print("=== Phase 3: Visualization ===")
    cfg = VizConfig(
        logs_dir=str(eval_dir),
        out_dir=str(charts_dir),
        bootstrap_iters=args.bootstrap_iters
    )
    viz = EvaluationVisualizer(cfg)
    summary_viz = viz.run_all()
    print(json.dumps(summary_viz, indent=2))

    summary_json = charts_dir / "summary.json"
    summary = _load_json(summary_json)
    has_summary = bool(summary and "ndcg@k_mean" in summary and "faith_mean" in summary)

    # ------------------------------------------------------------------
    print("=== Phase 4: Table Export ===")
    if has_summary:
        exp = EvaluationTableExporter(charts_dir=str(charts_dir))
        try:
            paths = exp.export()
            for k, v in paths.items():
                if k != "rows":
                    print(f"{k}: {v}")
        except Exception as e:
            print(f"Table export failed: {e}")
    else:
        print("Skipping table export (no summary.json with means).")

    # ------------------------------------------------------------------
    print("=== Phase 5: PDF Report ===")
    if has_summary:
        try:
            from src.core.evaluation.report_builder import ReportBuilder
            rb = ReportBuilder(charts_dir=str(charts_dir), eval_dir=str(eval_dir))
            pdf_path = rb.build(custom_name=f"benchmark_report_n{args.num_prompts}_seed{args.seed}.pdf")
            print(f"Generated PDF report: {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed: {e}")
    else:
        print("Skipping PDF generation (no summary.json with means).")

    elapsed = (time.time() - start_time) / 60
    print(f"\n=== BENCHMARK COMPLETE ({elapsed:.1f} min) ===")
    print(f"Results in: {eval_dir} | {charts_dir}")


if __name__ == "__main__":
    main()
