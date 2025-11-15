# src/core/evaluation/multi_model_report_builder.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


FAITH_COLORS = {
    "high": "#1a9850",
    "medium": "#fee08b",
    "low": "#d73027",
}

FAITH_LABELS = {
    "high": "High (≥0.80)",
    "medium": "Medium (0.50–0.79)",
    "low": "Low (<0.50)",
}


class MultiModelReportBuilder:
    """Aggregated PDF report comparing multiple LLMs on NDCG and Faithfulness."""

    def __init__(self, base_dir: str = "data"):
        self.base = Path(base_dir)
        self.eval_dirs = sorted(self.base.glob("eval_logs_*"))
        self.out_dir = self.base / "model_comparison"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        self.styleN = styles["Normal"]
        self.styleH = styles["Heading1"]
        self.styleH2 = styles["Heading2"]

    # ---------------------------------------------------------
    def _load_all_results(self) -> pd.DataFrame:
        rows = []
        for d in self.eval_dirs:
            model = d.name.replace("eval_logs_", "")
            for fp in d.glob("*_evaluation.json"):
                try:
                    x = json.loads(fp.read_text(encoding="utf-8"))
                    rows.append({
                        "query_id": x.get("query_id"),
                        "ndcg": float(x.get("ndcg@k", np.nan)),
                        "faith": float(x.get("faithfulness", np.nan)),
                        "model": model,
                    })
                except Exception:
                    continue
        return pd.DataFrame(rows)

    # ---------------------------------------------------------
    def _faith_band(self, v: float) -> str:
        if np.isnan(v):
            return "missing"
        if v >= 0.80:
            return "high"
        if v >= 0.50:
            return "medium"
        return "low"

    # ---------------------------------------------------------
    def _plot_faithfulness_band_comparison(self, df: pd.DataFrame) -> Path:
        df["band"] = df["faith"].apply(self._faith_band)
        bands = ["high", "medium", "low"]
        models = sorted(df["model"].unique())

        counts = (
            df.groupby(["model", "band"])["query_id"]
            .count()
            .unstack(fill_value=0)
            .reindex(columns=bands, fill_value=0)
        )

        x = np.arange(len(models))
        width = 0.22

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for i, b in enumerate(bands):
            offsets = x + (i - 1) * width
            ax.bar(
                offsets,
                counts[b].values,
                width=width,
                color=FAITH_COLORS[b],
                edgecolor="black",
                alpha=0.9,
                label=FAITH_LABELS[b]
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10)
        ax.set_ylabel("Number of queries")
        ax.set_title("Faithfulness band comparison across models")
        ax.legend(frameon=False)

        out = self.out_dir / "faithfulness_model_comparison.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # ---------------------------------------------------------
    def _aggregate_stats(self, df: pd.DataFrame) -> list[list[str]]:
        rows = [["Model", "Mean NDCG", "Mean Faith", "Median NDCG", "Median Faith", "Std NDCG", "Std Faith"]]
        for m in sorted(df["model"].unique()):
            d = df[df["model"] == m]
            rows.append([
                m,
                f"{d['ndcg'].mean():.3f}",
                f"{d['faith'].mean():.3f}",
                f"{d['ndcg'].median():.3f}",
                f"{d['faith'].median():.3f}",
                f"{d['ndcg'].std():.3f}",
                f"{d['faith'].std():.3f}",
            ])
        return rows

    # ---------------------------------------------------------
    def build(self, name: str = "multi_model_benchmark_report.pdf") -> Path:
        pdf_path = self.out_dir / name
        df = self._load_all_results()

        if df.empty:
            raise ValueError("No evaluation files found for any model.")

        # create plots
        plot_path = self._plot_faithfulness_band_comparison(df)

        # compute stats
        stats = self._aggregate_stats(df)

        # PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        story = []

        # Title
        story.append(Paragraph("<b>Multi-Model Benchmark Report</b>", self.styleH))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styleN))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph(
            "This report compares multiple LLM profiles in terms of retrieval relevance "
            "(NDCG@k) and factual grounding (Faithfulness). All evaluations were performed "
            "using an identical prompt set, identical retrieval stack, and identical parameters.",
            self.styleN
        ))
        story.append(PageBreak())

        # Stats Table
        story.append(Paragraph("1. Summary Statistics per Model", self.styleH))
        table = Table(stats, colWidths=[3.2 * cm] * 7)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(table)
        story.append(PageBreak())

        # Plots
        story.append(Paragraph("2. Faithfulness Band Comparison", self.styleH))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(plot_path), width=15 * cm, height=9 * cm))
        story.append(PageBreak())

        # Done
        story.append(Paragraph("End of Report", self.styleN))

        doc.build(story)
        return pdf_path
