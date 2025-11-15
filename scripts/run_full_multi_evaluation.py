# scripts/run_multi_model_benchmark.py
from __future__ import annotations
import sys
import os
import subprocess
import time
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# traffic-light colors consistent with your visualizer
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


def run(cmd: list[str]) -> None:
    # execute subprocess and wait
    print(">>>", " ".join(cmd))
    start = time.time()
    p = subprocess.Popen(cmd)
    p.wait()
    print(f"finished in {time.time() - start:.1f}s\n")


def load_eval_df(eval_dir: Path) -> pd.DataFrame:
    # load *_evaluation.json rows into one dataframe
    rows = []
    for fp in eval_dir.glob("*_evaluation.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            rows.append({
                "query_id": d.get("query_id"),
                "ndcg": float(d.get("ndcg@k", np.nan)),
                "faith": float(d.get("faithfulness", np.nan)),
                "model": d.get("model_name", "unknown"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def faith_band(f: float) -> str:
    # map continuous score to qualitative band
    if np.isnan(f):
        return "missing"
    if f >= 0.80:
        return "high"
    if f >= 0.50:
        return "medium"
    return "low"


def plot_global_comparison(df: pd.DataFrame, out_path: Path) -> None:
    # group counts by model + faithfulness band
    df["band"] = df["faith"].apply(faith_band)

    models = sorted(df["model"].unique())
    bands = ["high", "medium", "low"]

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

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    models = {
        "phi3_4b":    "phi3-4b",
        "mistral_7b": "mistral-7b-instruct",
        "llama3_8b":  "llama3-8b-instruct",
    }

    num_prompts = 100
    prompt_file = "data/prompts/benchmark_prompts.txt"

    all_dfs = []

    for profile, model in models.items():
        print(f"=== RUNNING MODEL: {profile} ===")

        logs   = f"data/logs_{profile}"
        evals  = f"data/eval_logs_{profile}"
        charts = f"data/eval_charts_{profile}"

        cmd = [
            "poetry", "run", "python", "scripts/run_full_benchmark.py",
            "--prompt_file", prompt_file,
            "--num_prompts", str(num_prompts),
            "--logs_dir", logs,
            "--eval_dir", evals,
            "--charts_dir", charts,
            "--k", "10",
            "--bootstrap_iters", "2000",
            "--seed", "42",
        ]
        run(cmd)

        df = load_eval_df(Path(evals))
        df["model"] = profile
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    outdir = Path("data/model_comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(outdir / "all_models_evaluation.csv", index=False, encoding="utf-8")

    plot_global_comparison(df_all, outdir / "faithfulness_model_comparison")

    print("Done. Outputs:")
    print("→ Combined CSV:", outdir / "all_models_evaluation.csv")
    print("→ Comparison plots:", outdir / "faithfulness_model_comparison.*")


if __name__ == "__main__":
    main()
