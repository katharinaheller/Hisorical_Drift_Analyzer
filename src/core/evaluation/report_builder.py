# src/core/evaluation/report_builder.py
from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("ReportBuilder")


class ReportBuilder:
    """Generates a publication-ready PDF report summarizing benchmark results with correlations and outlier analysis."""

    def __init__(self, charts_dir: str = "data/eval_charts"):
        self.charts_dir = Path(charts_dir)
        self.styles = getSampleStyleSheet()
        self.styleN = self.styles["Normal"]
        self.styleH = self.styles["Heading1"]
        self.styleH2 = self.styles["Heading2"]

    # ------------------------------------------------------------------
    def _load_summary(self) -> dict:
        summary_path = self.charts_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in {self.charts_dir}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _load_detailed_results(self) -> pd.DataFrame:
        """Load all evaluation JSON files to compute correlations."""
        eval_logs = sorted(self.charts_dir.parent.glob("eval_logs/*_evaluation.json"))
        records = []
        for fp in eval_logs:
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                records.append({
                    "query_id": d.get("query_id", fp.stem),
                    "ndcg@k": float(d.get("ndcg@k", np.nan)),
                    "faithfulness": float(d.get("faithfulness", np.nan))
                })
            except Exception:
                continue
        return pd.DataFrame(records).dropna(subset=["ndcg@k", "faithfulness"])

    def _load_table(self) -> list[list[str]]:
        table_path = self.charts_dir / "evaluation_table.csv"
        if not table_path.exists():
            return []
        rows = [ln.strip().split(",") for ln in table_path.read_text(encoding="utf-8").splitlines()]
        return rows

    def _find_images(self) -> list[Path]:
        """Collect all visualization images (.png) for inclusion."""
        return sorted(self.charts_dir.glob("*.png"))

    # ------------------------------------------------------------------
    def _compute_correlations(self, df: pd.DataFrame) -> dict:
        """Compute Pearson and Spearman correlations between NDCG@k and Faithfulness."""
        if df.empty:
            return {}
        pearson_corr, pearson_p = pearsonr(df["ndcg@k"], df["faithfulness"])
        spearman_corr, spearman_p = spearmanr(df["ndcg@k"], df["faithfulness"])
        return {
            "pearson_r": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_corr,
            "spearman_p": spearman_p
        }

    def _detect_outliers(self, df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
        """Identify statistical outliers based on z-scores."""
        if df.empty:
            return df
        df = df.copy()
        df["z_faith"] = (df["faithfulness"] - df["faithfulness"].mean()) / (df["faithfulness"].std() + 1e-6)
        df["z_ndcg"] = (df["ndcg@k"] - df["ndcg@k"].mean()) / (df["ndcg@k"].std() + 1e-6)
        df["is_outlier"] = (abs(df["z_faith"]) > z_thresh) | (abs(df["z_ndcg"]) > z_thresh)
        return df[df["is_outlier"]]

    # ------------------------------------------------------------------
    def build(self, custom_name: str | None = None) -> Path:
        """Compose and export the extended PDF benchmark report.

        Parameters
        ----------
        custom_name : str | None
            Optional custom PDF filename, e.g. 'benchmark_report_n50.pdf'.
            If None, report name is derived automatically from the number of evaluated prompts.
        """
        summary = self._load_summary()
        num_prompts = summary.get("files", 0)
        auto_name = f"benchmark_report_n{num_prompts}.pdf" if num_prompts else "benchmark_report.pdf"
        pdf_name = custom_name or auto_name
        pdf_path = self.charts_dir / pdf_name

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        story = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # --- Title page ---
        story.append(Paragraph("<b>Benchmark Evaluation Report</b>", self.styles["Title"]))
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(f"Generated automatically on {now}", self.styleN))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "This report summarizes automated evaluations of retrieval-augmented generation "
            "systems within the <i>Historical Drift Analyzer</i> architecture. "
            "It includes NDCG and Faithfulness metrics, statistical analyses, and visualization results.",
            self.styleN,
        ))
        story.append(PageBreak())

        # --- Summary metrics ---
        story.append(Paragraph("1. Summary Statistics", self.styleH))
        story.append(Spacer(1, 0.3 * cm))
        kv_pairs = [[k.replace('_', ' '), f"{v:.4f}" if isinstance(v, (int, float)) else str(v)]
                    for k, v in summary.items()]
        table = Table(kv_pairs, colWidths=[8 * cm, 6 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(table)
        story.append(PageBreak())

        # --- Correlation & Outlier Analysis ---
        df = self._load_detailed_results()
        if not df.empty:
            story.append(Paragraph("2. Correlation and Outlier Analysis", self.styleH))
            story.append(Spacer(1, 0.3 * cm))

            corr = self._compute_correlations(df)
            if corr:
                corr_rows = [
                    ["Metric Pair", "r / ρ", "p-value"],
                    ["Pearson (linear)", f"{corr['pearson_r']:.3f}", f"{corr['pearson_p']:.3e}"],
                    ["Spearman (rank)", f"{corr['spearman_rho']:.3f}", f"{corr['spearman_p']:.3e}"],
                ]
                corr_table = Table(corr_rows, colWidths=[6 * cm, 4 * cm, 4 * cm])
                corr_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                story.append(corr_table)
                story.append(Spacer(1, 0.5 * cm))
                story.append(Paragraph(
                    "Pearson r reflects linear association; Spearman ρ captures rank correlation. "
                    "Typical weak-to-moderate positive dependency indicates retrieval homogeneity, "
                    "suggesting that Faithfulness could be further contrasted with semantic coherence metrics "
                    "(e.g., BERTScore or FactScore).",
                    self.styleN,
                ))
                story.append(Spacer(1, 0.3 * cm))

            outliers = self._detect_outliers(df)
            if not outliers.empty:
                story.append(Paragraph("Detected Outliers", self.styleH2))
                story.append(Spacer(1, 0.3 * cm))
                data_rows = [["Query ID", "NDCG@k", "Faithfulness", "z_ndcg", "z_faith"]]
                for _, r in outliers.iterrows():
                    data_rows.append([
                        r["query_id"][:40],
                        f"{r['ndcg@k']:.3f}",
                        f"{r['faithfulness']:.3f}",
                        f"{r['z_ndcg']:.2f}",
                        f"{r['z_faith']:.2f}",
                    ])
                tab = Table(data_rows, colWidths=[6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
                tab.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                story.append(tab)
            else:
                story.append(Paragraph("No statistical outliers detected.", self.styleN))
            story.append(PageBreak())

        # --- Figures ---
        images = self._find_images()
        if images:
            story.append(Paragraph("3. Visual Analytics", self.styleH))
            story.append(Spacer(1, 0.3 * cm))
            for img_path in images:
                story.append(Paragraph(img_path.name.replace("_", " "), self.styleH2))
                story.append(Image(str(img_path), width=15 * cm, height=9 * cm))
                story.append(Spacer(1, 0.5 * cm))

        # --- Interpretation ---
        story.append(PageBreak())
        story.append(Paragraph("4. Interpretation and Next Steps", self.styleH))
        story.append(Paragraph(
            "The retrieval architecture is in its target state. "
            "Remaining limitations are primarily in the Faithfulness level rather than in retrieval quality. "
            "Future experiments should contrast temporal weighting (temporal_mode=True vs. False) "
            "and perform decade-based query analyses after year-detection enhancement.",
            self.styleN,
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "A convergence plot (mean ±95% CI vs. n) should be used to demonstrate statistical stabilization "
            "as sample size increases. All results are reproducible by rerunning the benchmark scripts "
            "with identical configurations.",
            self.styleN,
        ))

        # --- Build PDF ---
        doc.build(story)
        logger.info(f"PDF report with correlations & outlier analysis generated → {pdf_path}")
        return pdf_path
