# scripts/run_convergence_plot.py
from __future__ import annotations
import argparse
from src.core.evaluation.convergence_plotter import ConvergencePlotter

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence of evaluation metrics (mean ±95% CI vs sample size)."
    )
    parser.add_argument(
        "--charts_dir",
        type=str,
        default="data/eval_charts",
        help="Directory containing summary_n*.json files from benchmark series."
    )
    args = parser.parse_args()

    print(f"=== Convergence Plot Generation ===")
    plotter = ConvergencePlotter(charts_dir=args.charts_dir)
    plotter.plot()
    print(f"Convergence plot created in {args.charts_dir}")

if __name__ == "__main__":
    main()
