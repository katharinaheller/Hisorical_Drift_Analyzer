# src/core/evaluation/evaluation_visualizer.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr  # for rank-based correlation

from src.core.evaluation.plot_style import (
    apply_scientific_style,
)

# TUM/Harvard palette plus traffic-light bands
PRIMARY_COLOR = "#003359"      # deep blue (NDCG)
SECONDARY_COLOR = "#CC0000"    # crimson (Faithfulness line plots)
ACCENT_COLOR = "#FFB300"       # warm accent for scatter

FAITH_GOOD_COLOR = "#1a9850"   # green for high faithfulness
FAITH_MED_COLOR = "#fee08b"    # yellow for medium faithfulness
FAITH_LOW_COLOR = "#d73027"    # red for low faithfulness


@dataclass
class VizConfig:
    logs_dir: str = "data/eval_logs"
    out_dir: str = "data/eval_charts"
    pattern: str = "*_evaluation.json"
    bootstrap_iters: int = 2000
    random_seed: int = 42
    iqr_k: float = 1.5
    z_thresh: float = 3.0

    # MODERATE BANDS (angepasst)
    faith_high_thr: float = 0.70     # vorher 0.80
    faith_mid_thr: float = 0.40      # vorher 0.50
    # below mid_thr is considered poor


class EvaluationVisualizer:
    """
    Publication-oriented evaluation visualizer.
    """

    def __init__(self, cfg: VizConfig | None = None):
        self.cfg = cfg or VizConfig()
        self.logs_dir = Path(self.cfg.logs_dir)
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.cfg.random_seed)
        apply_scientific_style()
        self._fig_no = 1

    # ------------------------------------------------------------------
    def _load_eval_rows(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for fp in sorted(self.logs_dir.glob(self.cfg.pattern)):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                rows.append({
                    "query_id": data.get("query_id", fp.stem),
                    "ndcg@k": float(data.get("ndcg@k", np.nan)),
                    "faithfulness": float(data.get("faithfulness", np.nan)),
                    "citation_hit_rate": float(data.get("citation_hit_rate", np.nan)),
                    "dominant_decade": data.get("dominant_decade", None),
                    "model": data.get("model")
                             or data.get("llm_profile")
                             or data.get("model_name"),
                })
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=["query_id", "ndcg@k", "faithfulness", "citation_hit_rate", "dominant_decade", "model"]
            )
        return df.dropna(how="all", subset=["ndcg@k", "faithfulness"])

    # ------------------------------------------------------------------
    def _bootstrap_ci(self, arr: np.ndarray, iters: int) -> Tuple[float, float, float]:
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan"), float("nan")
        if len(arr) == 1:
            m = float(arr[0])
            return m, m, m
        n = len(arr)
        idx = np.random.randint(0, n, size=(iters, n))
        boot = np.mean(arr[idx], axis=1)
        return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # ------------------------------------------------------------------
    def _outliers_iqr(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, float)
        clean = arr[~np.isnan(arr)]
        if len(clean) == 0:
            return np.zeros_like(arr, bool)
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros_like(arr, bool)
        lo = q1 - self.cfg.iqr_k * iqr
        hi = q3 + self.cfg.iqr_k * iqr
        return (arr < lo) | (arr > hi)

    def _outliers_z(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, float)
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        if sd == 0 or np.isnan(sd):
            return np.zeros_like(arr, bool)
        z = (arr - mu) / sd
        return np.abs(z) > self.cfg.z_thresh

    # ------------------------------------------------------------------
    def _save_df(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.out_dir / f"{name}.csv", index=False, encoding="utf-8")

    def _save_fig(self, fig: plt.Figure, stem: str) -> None:
        fig.savefig(self.out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(self.out_dir / f"{stem}.svg", bbox_inches="tight")

    def _titled(self, base: str) -> str:
        title = f"Figure {self._fig_no}: {base}"
        self._fig_no += 1
        return title

    # ------------------------------------------------------------------
    def _faithfulness_band(self, val: float) -> str:
        if np.isnan(val):
            return "missing"
        if val >= self.cfg.faith_high_thr:
            return "high"
        if val >= self.cfg.faith_mid_thr:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    def plot_ndcg_histogram(self, df: pd.DataFrame) -> None:
        vals = df["ndcg@k"].astype(float).dropna().values
        if len(vals) == 0:
            return

        min_v, max_v = np.min(vals), np.max(vals)
        span = max_v - min_v

        if span < 0.05 and max_v > 0.8:
            xmin = max(0.8, min_v - 0.01)
            xmax = 1.0
            bins = np.linspace(xmin, xmax, 12)
        elif span < 0.2:
            xmin = max(0.0, min_v - 0.05)
            xmax = min(1.0, max_v + 0.05)
            bins = np.linspace(xmin, xmax, 15)
        else:
            xmin, xmax = 0.0, 1.0
            bins = 15

        fig = plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=bins, color=PRIMARY_COLOR, edgecolor="black", alpha=0.8)
        plt.xlabel("NDCG@k")
        plt.ylabel("Count")
        plt.title(self._titled("Distribution of NDCG@k"))
        plt.xlim(xmin, xmax)
        ymin, ymax = plt.ylim()
        plt.ylim(0, max(ymax, 3))
        plt.tight_layout()
        self._save_fig(fig, "hist_ndcg")
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_faithfulness_bands_global(self, df: pd.DataFrame) -> None:
        vals = df["faithfulness"].astype(float).dropna().values
        if len(vals) == 0:
            return

        bands = [self._faithfulness_band(v) for v in vals]
        series = pd.Series(bands)
        counts = series.value_counts().reindex(["high", "medium", "low"], fill_value=0)

        colors = [
            FAITH_GOOD_COLOR,
            FAITH_MED_COLOR,
            FAITH_LOW_COLOR,
        ]
        labels = [
            "High (≥ 0.70)",
            "Medium (0.40–0.69)",
            "Low (< 0.40)"
        ]

        fig = plt.figure(figsize=(6, 4))
        x = np.arange(len(counts))
        plt.bar(x, counts.values, color=colors, edgecolor="black", alpha=0.85)
        plt.xticks(x, labels, rotation=10)
        plt.ylabel("Number of queries")
        plt.xlabel("Faithfulness band")
        plt.title(self._titled("Faithfulness bands across all queries"))
        for i, v in enumerate(counts.values):
            plt.text(i, v + 0.1, str(int(v)), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        self._save_fig(fig, "faithfulness_bands_global")
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_faithfulness_bands_by_model(self, df: pd.DataFrame) -> None:
        if "model" not in df.columns:
            return
        df_model = df.dropna(subset=["model"])
        if df_model.empty:
            return

        models = sorted(df_model["model"].unique())
        if len(models) <= 1:
            return

        df_model = df_model.copy()
        df_model["faith_band"] = df_model["faithfulness"].astype(float).apply(self._faithfulness_band)

        band_order = ["high", "medium", "low"]
        band_labels = [
            "High (≥ 0.70)",
            "Medium (0.40–0.69)",
            "Low (< 0.40)"
        ]
        band_color_map = {
            "high": FAITH_GOOD_COLOR,
            "medium": FAITH_MED_COLOR,
            "low": FAITH_LOW_COLOR,
        }

        counts = (
            df_model.groupby(["model", "faith_band"])["query_id"]
            .count()
            .unstack(fill_value=0)
            .reindex(columns=band_order, fill_value=0)
        )

        x = np.arange(len(models))
        width = 0.22

        fig = plt.figure(figsize=(7, 4.5))
        for i, band in enumerate(band_order):
            offsets = x + (i - 1) * width
            plt.bar(
                offsets,
                counts[band].values,
                width=width,
                label=band_labels[i],
                color=band_color_map[band],
                edgecolor="black",
                alpha=0.9,
            )

        plt.xticks(x, models, rotation=10)
        plt.ylabel("Number of queries")
        plt.xlabel("LLM profile")
        plt.title(self._titled("Faithfulness band distribution by LLM"))
        plt.legend(frameon=False)
        plt.tight_layout()
        self._save_fig(fig, "faithfulness_bands_by_model")
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_scatter_correlation(self, df: pd.DataFrame) -> Tuple[float, float]:
        x = df["ndcg@k"].astype(float).values
        y = df["faithfulness"].astype(float).values
        mask = (~np.isnan(x)) & (~np.isnan(y))

        if mask.sum() < 3:
            rp = rs = float("nan")
        else:
            rp = float(np.corrcoef(x[mask], y[mask])[0, 1])
            rs, _ = spearmanr(x[mask], y[mask])

        fig = plt.figure(figsize=(6, 4))
        plt.scatter(x[mask], y[mask], s=26, alpha=0.7, color=ACCENT_COLOR)
        plt.xlabel("NDCG@k")
        plt.ylabel("Faithfulness")
        plt.title(self._titled(f"Scatter NDCG@k vs Faithfulness (r={rp:.3f}, ρ={rs:.3f})"))
        plt.tight_layout()
        self._save_fig(fig, "scatter_ndcg_vs_faithfulness")
        plt.close(fig)
        return rp, rs

    # ------------------------------------------------------------------
    def plot_run_order_control(self, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        df["idx"] = np.arange(len(df))

        for col, color in [("ndcg@k", PRIMARY_COLOR), ("faithfulness", SECONDARY_COLOR)]:
            vals = df[col].astype(float).values
            if len(vals) == 0:
                continue

            mu, sd = np.nanmean(vals), np.nanstd(vals)
            ucl, lcl = mu + 3 * sd, mu - 3 * sd

            fig = plt.figure(figsize=(8, 4))
            ax = plt.gca()
            ax.fill_between(df["idx"], mu - sd, mu + sd, color="gray", alpha=0.18)
            ax.plot(df["idx"], vals, marker="o", linestyle="-", color=color, linewidth=1.1)
            ax.axhline(mu, linestyle="--", color=color, linewidth=0.9)
            ax.axhline(ucl, linestyle=":", color="black", linewidth=0.8)
            ax.axhline(lcl, linestyle=":", color="black", linewidth=0.8)
            ax.set_xlabel("Run index")
            ax.set_ylabel(col)
            ax.set_title(self._titled(f"Run-order chart for {col}"))
            plt.tight_layout()
            self._save_fig(fig, f"run_order_{col}")
            plt.close(fig)

    # ------------------------------------------------------------------
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        nd = df["ndcg@k"].astype(float).values
        fa = df["faithfulness"].astype(float).values

        mask = (
            self._outliers_iqr(nd)
            | self._outliers_z(nd)
            | self._outliers_iqr(fa)
            | self._outliers_z(fa)
        )
        return df[mask].copy()

    # ------------------------------------------------------------------
    def summarize(self, df: pd.DataFrame) -> Dict[str, Any]:
        nd = df["ndcg@k"].astype(float).values
        fa = df["faithfulness"].astype(float).values

        nd_m, nd_lo, nd_hi = self._bootstrap_ci(nd, self.cfg.bootstrap_iters)
        fa_m, fa_lo, fa_hi = self._bootstrap_ci(fa, self.cfg.bootstrap_iters)

        summary = {
            "files": int(df.shape[0]),
            "ndcg@k_mean": nd_m,
            "ndcg@k_ci95_lo": nd_lo,
            "ndcg@k_ci95_hi": nd_hi,
            "faith_mean": fa_m,
            "faith_ci95_lo": fa_lo,
            "faith_ci95_hi": fa_hi,
            "ndcg@k_median": float(np.nanmedian(nd)),
            "faith_median": float(np.nanmedian(fa)),
            "ndcg@k_std": float(np.nanstd(nd)),
            "faith_std": float(np.nanstd(fa)),
            "bootstrap_iters": self.cfg.bootstrap_iters,
            "iqr_k": self.cfg.iqr_k,
            "z_thresh": self.cfg.z_thresh,
            "faith_high_thr": self.cfg.faith_high_thr,
            "faith_mid_thr": self.cfg.faith_mid_thr,
        }

        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        df = self._load_eval_rows()
        self._save_df("raw_eval", df)

        if df.empty:
            summary = {"files": 0}
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        self.plot_ndcg_histogram(df)
        self.plot_faithfulness_bands_global(df)
        self.plot_faithfulness_bands_by_model(df)
        self.plot_scatter_correlation(df)
        self.plot_run_order_control(df)

        outliers = self.detect_outliers(df)
        self._save_df("outliers", outliers)

        summary = self.summarize(df)
        return summary
