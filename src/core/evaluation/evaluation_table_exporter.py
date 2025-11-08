from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class EvaluationTableExporter:
    """Exports evaluation summary statistics as LaTeX, CSV, and Markdown tables."""

    def __init__(self, charts_dir: str = "data/eval_charts"):
        self.charts_dir = Path(charts_dir)
        self.summary_path = self.charts_dir / "summary.json"
        self.out_tex = self.charts_dir / "evaluation_table.tex"
        self.out_csv = self.charts_dir / "evaluation_table.csv"
        self.out_md  = self.charts_dir / "evaluation_table.md"

    # ------------------------------------------------------------------
    def _load_summary(self) -> Dict[str, Any]:
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in {self.charts_dir}")
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    def _format_value(self, mean: float, lo: float, hi: float, digits: int = 3) -> str:
        """Format mean ± CI string."""
        return f"{mean:.{digits}f} ± {((hi - lo) / 2):.{digits}f}"

    # ------------------------------------------------------------------
    def export(self) -> Dict[str, Any]:
        """Generate LaTeX/CSV/MD tables from summary.json."""
        data = self._load_summary()
        if not data or "files" not in data:
            raise ValueError("Invalid or incomplete summary.json")

        rows: List[Dict[str, Any]] = [
            {
                "Metric": "NDCG@k",
                "Mean ± CI": self._format_value(data["ndcg@k_mean"],
                                                data["ndcg@k_ci95_lo"],
                                                data["ndcg@k_ci95_hi"]),
                "Median": f"{data['ndcg@k_median']:.3f}",
                "Std": f"{data['ndcg@k_std']:.3f}"
            },
            {
                "Metric": "Faithfulness",
                "Mean ± CI": self._format_value(data["faith_mean"],
                                                data["faith_ci95_lo"],
                                                data["faith_ci95_hi"]),
                "Median": f"{data['faith_median']:.3f}",
                "Std": f"{data['faith_std']:.3f}"
            }
        ]

        df = pd.DataFrame(rows)
        # CSV
        df.to_csv(self.out_csv, index=False, encoding="utf-8")

        # Markdown
        md_lines = ["| Metric | Mean ± CI | Median | Std |",
                    "|:--|--:|--:|--:|"]
        for r in rows:
            md_lines.append(f"| {r['Metric']} | {r['Mean ± CI']} | {r['Median']} | {r['Std']} |")
        self.out_md.write_text("\n".join(md_lines), encoding="utf-8")

        # LaTeX
        tex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Evaluation metrics with 95\\% confidence intervals (bootstrap).}",
            "\\label{tab:evaluation_results}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Metric & Mean $\\pm$ CI & Median & Std \\\\",
            "\\midrule",
        ]
        for r in rows:
            tex.append(f"{r['Metric']} & {r['Mean ± CI']} & {r['Median']} & {r['Std']} \\\\")
        tex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        self.out_tex.write_text("\n".join(tex), encoding="utf-8")

        return {
            "csv": str(self.out_csv),
            "md": str(self.out_md),
            "tex": str(self.out_tex),
            "rows": len(rows)
        }
