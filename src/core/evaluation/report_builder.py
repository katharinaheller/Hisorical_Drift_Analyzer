# src/core/evaluation/report_builder.py
from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("ReportBuilder")


class ReportBuilder:
    """Minimal, robust statistical PDF report: summary stats, correlations, outliers, and charts."""

    def __init__(self, charts_dir: str = "data/eval_charts", eval_dir: str | None = None):
        # Base folder for figures & summary.json
        self.charts_dir = Path(charts_dir)

        # Evaluation-log directory (robust for eval_logs_*)
        if eval_dir:
            self.eval_dir = Path(eval_dir)
        else:
            # Autodiscover eval_logs directories
            parent = self.charts_dir.parent
            candidates = sorted(parent.glob("eval_logs_*"))
            if candidates:
                self.eval_dir = candidates[-1]
            else:
                self.eval_dir = parent / "eval_logs"

        # ReportLab styles
        styles = getSampleStyleSheet()
        self.styleN = styles["Normal"]
        self.styleH = styles["Heading1"]
        self.styleH2 = styles["Heading2"]

    # ------------------------------------------------------------------
    def _load_summary(self) -> dict:
        # Read summary.json produced by EvaluationVisualizer
        summary_path = self.charts_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json missing in {self.charts_dir}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    def _load_detailed_results(self) -> pd.DataFrame:
        # Load all *_evaluation.json for correlation/outlier analysis
        records = []
        for fp in sorted(self.eval_dir.glob("*_evaluation.json")):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                records.append({
                    "query_id": d.get("query_id", fp.stem),
                    "ndcg@k": float(d.get("ndcg@k", np.nan)),
                    "faithfulness": float(d.get("faithfulness", np.nan))
                })
            except Exception:
                continue
        df = pd.DataFrame(records)
        return df.dropna(subset=["ndcg@k", "faithfulness"])

    # ------------------------------------------------------------------
    def _compute_correlations(self, df: pd.DataFrame) -> dict:
        # Pearson + Spearman between NDCG and Faithfulness
        if df.empty:
            return {}
        x = df["ndcg@k"].astype(float)
        y = df["faithfulness"].astype(float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return {
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp
        }

    # ------------------------------------------------------------------
    def _detect_outliers(self, df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
        # Robust z-score outlier detection
        if df.empty:
            return df
        eps = 1e-6
        df = df.copy()
        df["z_ndcg"] = (df["ndcg@k"] - df["ndcg@k"].mean()) / (df["ndcg@k"].std() + eps)
        df["z_faith"] = (df["faithfulness"] - df["faithfulness"].mean()) / (df["faithfulness"].std() + eps)
        df["is_outlier"] = (df["z_ndcg"].abs() > z_thresh) | (df["z_faith"].abs() > z_thresh)
        return df[df["is_outlier"]]

    # ------------------------------------------------------------------
    def _find_charts(self) -> list[Path]:
        # Include all PNG charts (histograms, correlation plots, etc.)
        return sorted(self.charts_dir.glob("*.png"))

    # ------------------------------------------------------------------
    def build(self, custom_name: str | None = None) -> Path:
        # Load summary data
        summary = self._load_summary()
        n = summary.get("files", 0)

        # PDF name
        pdf_name = custom_name or f"benchmark_report_n{n}.pdf"
        pdf_path = self.charts_dir / pdf_name

        # PDF document setup
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )
        story = []

        # ------------------------------------------------------------------
        # Title Page
        story.append(Paragraph("<b>Statistical Benchmark Report</b>", self.styleH))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styleN))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(
            "This report provides quantitative evaluation results for NDCG@k and Faithfulness, "
            "correlation diagnostics, outlier detection, and statistical visualizations.",
            self.styleN,
        ))
        story.append(PageBreak())

        # ------------------------------------------------------------------
        # Summary Statistics
        story.append(Paragraph("1. Summary Statistics", self.styleH))
        story.append(Spacer(1, 0.2 * cm))

        rows = []
        for k, v in summary.items():
            label = k.replace("_", " ")
            if isinstance(v, (int, float)):
                rows.append([label, f"{v:.4f}"])
            else:
                rows.append([label, str(v)])

        table = Table(rows, colWidths=[8 * cm, 6 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))
        story.append(table)
        story.append(PageBreak())

        # ------------------------------------------------------------------
        # Correlations + Outliers
        df = self._load_detailed_results()
        if not df.empty:
            story.append(Paragraph("2. Correlation and Outlier Analysis", self.styleH))
            story.append(Spacer(1, 0.2 * cm))

            corr = self._compute_correlations(df)
            if corr:
                crows = [
                    ["Metric Pair", "Correlation", "p-value"],
                    ["Pearson r", f"{corr['pearson_r']:.3f}", f"{corr['pearson_p']:.3e}"],
                    ["Spearman ρ", f"{corr['spearman_rho']:.3f}", f"{corr['spearman_p']:.3e}"],
                ]
                ct = Table(crows, colWidths=[6 * cm, 4 * cm, 4 * cm])
                ct.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER")
                ]))
                story.append(ct)
                story.append(Spacer(1, 0.3 * cm))

            out = self._detect_outliers(df)
            if not out.empty:
                story.append(Paragraph("Detected Outliers", self.styleH2))
                story.append(Spacer(1, 0.2 * cm))

                trows = [["Query", "NDCG@k", "Faithfulness", "z_ndcg", "z_faith"]]
                for _, r in out.iterrows():
                    qid = r["query_id"]
                    wrapped = "\n".join([qid[i:i+45] for i in range(0, len(qid), 45)])
                    trows.append([
                        wrapped,
                        f"{r['ndcg@k']:.3f}",
                        f"{r['faithfulness']:.3f}",
                        f"{r['z_ndcg']:.2f}",
                        f"{r['z_faith']:.2f}",
                    ])

                ot = Table(
                    trows,
                    colWidths=[8 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 2.0 * cm]
                )
                ot.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8)
                ]))
                story.append(ot)
            else:
                story.append(Paragraph("No outliers detected.", self.styleN))

            story.append(PageBreak())

        # ------------------------------------------------------------------
        # Charts
        images = self._find_charts()
        if images:
            story.append(Paragraph("3. Visual Analytics", self.styleH))
            story.append(Spacer(1, 0.2 * cm))
            for img in images:
                story.append(Paragraph(img.name.replace("_", " "), self.styleH2))
                story.append(Image(str(img), width=15 * cm, height=9 * cm))
                story.append(Spacer(1, 0.4 * cm))

        # ------------------------------------------------------------------
        doc.build(story)
        logger.info(f"PDF generated → {pdf_path}")
        return pdf_path
