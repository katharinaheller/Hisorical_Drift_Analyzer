# scripts/run_eval_analytics.py
from __future__ import annotations

# --- ensure project root is on sys.path ---
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from src.core.evaluation.evaluation_visualizer import EvaluationVisualizer, VizConfig

def main() -> None:
    p = argparse.ArgumentParser(description="Run evaluation analytics and plots.")
    p.add_argument("--logs_dir", type=str, default="data/eval_logs")
    p.add_argument("--out_dir", type=str, default="data/eval_charts")
    p.add_argument("--iters", type=int, default=2000)
    args = p.parse_args()

    cfg = VizConfig(logs_dir=args.logs_dir, out_dir=args.out_dir, bootstrap_iters=args.iters)
    viz = EvaluationVisualizer(cfg)
    summary = viz.run_all()
    print(summary)

if __name__ == "__main__":
    main()
