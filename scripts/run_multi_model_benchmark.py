# scripts/run_multi_model_benchmark.py
from __future__ import annotations

# Add project root
import sys
import os
import subprocess
import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.evaluation.multi_model_report_builder import MultiModelReportBuilder

# Color scale for six-level classification
FAITH_COLORS = {
    "excellent":  "#006d2c",
    "good":       "#31a354",
    "fair":       "#a1d99b",
    "borderline": "#fed976",
    "poor":       "#fd8d3c",
    "critical":   "#e31a1c",
}

FAITH_LABELS = {
    "excellent":  "Excellent (≥0.95)",
    "good":       "Good (0.85–0.94)",
    "fair":       "Fair (0.75–0.84)",
    "borderline": "Borderline (0.60–0.74)",
    "poor":       "Poor (0.40–0.59)",
    "critical":   "Critical (<0.40)",
}


# ---------------------------------------------------------

def run(cmd: list[str]) -> None:
    # Wrapper for subprocess execution
    print(">>>", " ".join(cmd))
    start = time.time()
    p = subprocess.Popen(cmd)
    p.wait()
    print(f"finished in {time.time() - start:.1f}s\n")


# ---------------------------------------------------------

def load_eval_df(eval_dir: Path) -> pd.DataFrame:
    # Load all evaluation json files in the directory
    rows = []
    for fp in eval_dir.glob("*_evaluation.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        nd = d.get("ndcg@k")
        fa = d.get("faithfulness")

        rows.append({
            "query_id": d.get("query_id"),
            "ndcg": float(nd) if nd is not None else np.nan,
            "faith": float(fa) if fa is not None else np.nan,
            "model": d.get("model_name", "unknown"),
        })

    df = pd.DataFrame(rows)
    if "faith" not in df:
        df["faith"] = np.nan
    return df


# ---------------------------------------------------------

def faith_band(f: float) -> str:
    # Six-level categorization
    if np.isnan(f):
        return "missing"
    v = float(f)
    if v >= 0.95:
        return "excellent"
    if v >= 0.85:
        return "good"
    if v >= 0.75:
        return "fair"
    if v >= 0.60:
        return "borderline"
    if v >= 0.40:
        return "poor"
    return "critical"


# ---------------------------------------------------------

def clear_and_prepare(path: Path) -> None:
    # Clear directory and reconstruct it
    if path.exists():
        for f in path.glob("*"):
            if f.is_file():
                f.unlink()
            else:
                import shutil
                shutil.rmtree(f)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------

def plot_global_comparison(df: pd.DataFrame, out_path: Path) -> None:
    # Draw multi-model band distribution
    df = df.copy()
    df["band"] = df["faith"].apply(faith_band)

    models = sorted(df["model"].unique())
    bands = ["excellent", "good", "fair", "borderline", "poor", "critical"]

    counts = (
        df.groupby(["model", "band"])["query_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=bands, fill_value=0)
    )

    x = np.arange(len(models))
    width = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, b in enumerate(bands):
        offsets = x + (i - (len(bands) - 1) / 2) * width
        ax.bar(
            offsets,
            counts[b].values,
            width=width,
            color=FAITH_COLORS[b],
            edgecolor="black",
            alpha=0.9,
            label=FAITH_LABELS[b],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("Number of queries")
    ax.set_title("Faithfulness band comparison across models")
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model benchmark with strict isolation and build comparison report."
    )
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--prompt_file", type=str, default="data/prompts/benchmark_prompts.txt")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = {
        "phi3_4b": "phi3-4b",
        "mistral_7b": "mistral-7b-instruct",
        "llama3_8b": "llama3-8b-instruct",
    }

    all_dfs = []

    for profile, model_id in models.items():
        print(f"=== RUNNING MODEL: {profile} ({model_id}) ===")

        logs_dir = Path(f"data/logs_{profile}")
        eval_dir = Path(f"data/eval_logs_{profile}")
        charts_dir = Path(f"data/eval_charts_{profile}")

        clear_and_prepare(logs_dir)
        clear_and_prepare(eval_dir)
        clear_and_prepare(charts_dir)

        # Run isolated benchmark
        cmd = [
            "poetry", "run", "python", "scripts/run_full_benchmark.py",
            "--prompt_file", args.prompt_file,
            "--num_prompts", str(args.num_prompts),
            "--logs_dir", str(logs_dir),
            "--eval_dir", str(eval_dir),
            "--charts_dir", str(charts_dir),
            "--k", str(args.k),
            "--bootstrap_iters", str(args.bootstrap_iters),
            "--seed", str(args.seed),
        ]
        run(cmd)

        df = load_eval_df(eval_dir)
        df["model"] = profile
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    outdir = Path("data/model_comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(outdir / "all_models_evaluation.csv", index=False, encoding="utf-8")

    plot_global_comparison(df_all, outdir / "faithfulness_model_comparison")

    rb = MultiModelReportBuilder(base_dir="data")
    pdf_path = rb.build(name="multi_model_benchmark_report.pdf")

    print("\nAll done.")
    print("→ CSV :", outdir / "all_models_evaluation.csv")
    print("→ Plots:", outdir / "faithfulness_model_comparison.*")
    print("→ PDF :", pdf_path)


if __name__ == "__main__":
    main()
