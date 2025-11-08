from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from src.core.evaluation.plot_style import apply_scientific_style, annotate_sample_info

class ConvergencePlotter:
    """Aggregates summary.json files across n-stages and visualizes mean ±95% CI convergence."""

    def __init__(self, charts_dir: str = "data/eval_charts"):
        self.charts_dir = Path(charts_dir)
        apply_scientific_style()

    def _load_stage_summaries(self) -> List[dict]:
        """Collect all summary_n*.json files (one per n-stage)."""
        summaries = []
        for fp in sorted(self.charts_dir.glob("summary_n*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                data["n"] = int("".join([c for c in fp.stem if c.isdigit()]))
                summaries.append(data)
            except Exception:
                continue
        return sorted(summaries, key=lambda x: x["n"])

    def plot(self) -> None:
        """Plot mean ±95% CI vs n for NDCG@k and Faithfulness."""
        data = self._load_stage_summaries()
        if not data:
            print("No stage summaries found in charts directory.")
            return

        ns = np.array([d["n"] for d in data])
        nd_mean = np.array([d["ndcg@k_mean"] for d in data])
        nd_lo = np.array([d["ndcg@k_ci95_lo"] for d in data])
        nd_hi = np.array([d["ndcg@k_ci95_hi"] for d in data])

        fa_mean = np.array([d["faith_mean"] for d in data])
        fa_lo = np.array([d["faith_ci95_lo"] for d in data])
        fa_hi = np.array([d["faith_ci95_hi"] for d in data])

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        # NDCG@k line
        ax.plot(ns, nd_mean, "-o", color="#1b9e77", label="NDCG@k mean")
        ax.fill_between(ns, nd_lo, nd_hi, color="#1b9e77", alpha=0.2)

        # Faithfulness line
        ax.plot(ns, fa_mean, "-o", color="#d95f02", label="Faithfulness mean")
        ax.fill_between(ns, fa_lo, fa_hi, color="#d95f02", alpha=0.2)

        ax.set_xlabel("Sample size n")
        ax.set_ylabel("Metric value")
        ax.set_title("Convergence of Evaluation Metrics (mean ±95% CI)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        annotate_sample_info(ax, n=len(ns))

        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(self.charts_dir / f"convergence_plot.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Convergence plot saved to {self.charts_dir}/convergence_plot.*")
