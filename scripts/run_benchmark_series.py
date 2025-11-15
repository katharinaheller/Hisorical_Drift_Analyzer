# scripts/run_benchmark_series.py
from __future__ import annotations

# --- ensure project root is on sys.path ---
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
import time
import argparse
from pathlib import Path

def run_cmd(cmd: list[str]) -> None:
    """Execute subprocess and stream output."""
    print(">>>", " ".join(cmd))
    start = time.time()
    p = subprocess.Popen(cmd)
    p.wait()
    print(f"⏱Finished in {time.time() - start:.1f}s\n")

def main() -> None:
    p = argparse.ArgumentParser(description="Automated multi-stage benchmark runner.")
    p.add_argument("--prompt_file", default="data/prompts/benchmark_prompts.txt")
    p.add_argument("--logs_dir", default="data/logs")
    p.add_argument("--eval_dir", default="data/eval_logs")
    p.add_argument("--charts_dir", default="data/eval_charts")
    p.add_argument("--bootstrap_iters", type=int, default=2000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ns", nargs="+", type=int, default=[10, 30, 50, 100])
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 101])
    args = p.parse_args()

    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.charts_dir).mkdir(parents=True, exist_ok=True)

    for n in args.ns:
        print(f"\n=== Benchmark stage n={n} ===")
        for seed in args.seeds:
            cmd = [
                "poetry", "run", "python", "scripts/run_full_benchmark.py",
                "--prompt_file", args.prompt_file,
                "--num_prompts", str(n),
                "--logs_dir", args.logs_dir,
                "--eval_dir", args.eval_dir,
                "--charts_dir", args.charts_dir,
                "--k", str(args.k),
                "--bootstrap_iters", str(args.bootstrap_iters),
                "--seed", str(seed)
            ]
            run_cmd(cmd)

        print(f"\n=== Post-analysis for n={n} ===")
        eval_cmd = [
            "poetry", "run", "python", "scripts/run_eval_analytics.py",
            "--logs_dir", args.eval_dir,
            "--out_dir", args.charts_dir,
            "--iters", str(args.bootstrap_iters)
        ]
        run_cmd(eval_cmd)

if __name__ == "__main__":
    main()
