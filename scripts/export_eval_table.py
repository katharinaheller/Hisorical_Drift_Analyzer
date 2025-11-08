from __future__ import annotations
import argparse
from src.core.evaluation.evaluation_table_exporter import EvaluationTableExporter

def main() -> None:
    parser = argparse.ArgumentParser(description="Export evaluation summary as LaTeX/CSV/Markdown tables.")
    parser.add_argument("--charts_dir", type=str, default="data/eval_charts", help="Directory containing summary.json")
    args = parser.parse_args()
    exp = EvaluationTableExporter(args.charts_dir)
    res = exp.export()
    print("Export completed:")
    for k, v in res.items():
        if k != "rows":
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
