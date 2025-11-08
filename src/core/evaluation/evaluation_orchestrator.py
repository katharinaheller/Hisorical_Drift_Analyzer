from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from src.core.evaluation.interfaces.i_metric import IMetric
from src.core.evaluation.metrics.ndcg_metric import NDCGMetric
from src.core.evaluation.metrics.faithfulness_metric import FaithfulnessMetric
from src.core.evaluation.ground_truth_builder import GroundTruthBuilder
from src.core.evaluation.utils import make_chunk_id

logger = logging.getLogger("EvaluationOrchestrator")


class EvaluationOrchestrator:
    """Coordinates metric computation and automatic semantic ground-truth generation."""

    def __init__(self, output_dir: str = "data/eval_logs", k: int = 10):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.k = k

        # Core evaluation metrics
        self.metrics: Dict[str, IMetric] = {
            "ndcg@k": NDCGMetric(k=k),
            "faithfulness": FaithfulnessMetric(),
        }

        # Semantic ground truth builder
        self.gt_builder = GroundTruthBuilder()

        logger.info(f"Evaluation orchestrator ready | k={self.k}")

    # ------------------------------------------------------------------
    def _ensure_chunk_ids(self, items: List[Dict[str, Any]]) -> None:
        """Ensure every retrieved chunk has a stable, unique ID."""
        for ch in items:
            if not ch.get("id"):
                ch["id"] = make_chunk_id(ch)

    # ------------------------------------------------------------------
    def _safe_id(self, s: str | None) -> str:
        """Generate a filesystem-safe identifier for query strings."""
        if not s:
            return "query"
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)[:80] or "query"

    # ------------------------------------------------------------------
    def evaluate_single(self, query: str, retrieved_chunks: List[Dict[str, Any]], model_output: str) -> Dict[str, float]:
        """Compute NDCG and Faithfulness for a single query."""
        self._ensure_chunk_ids(retrieved_chunks)

        # Build semantic ground truth dynamically per query
        gt_map = self.gt_builder.build(query, retrieved_chunks)
        relevance_scores = [int(gt_map.get(ch["id"], ch.get("relevance", 0))) for ch in retrieved_chunks]

        ndcg_val = self.metrics["ndcg@k"].compute(relevance_scores=relevance_scores)
        faith_val = self.metrics["faithfulness"].compute(
            context_chunks=[c.get("text", "") for c in retrieved_chunks],
            answer=model_output,
        )

        qid = self._safe_id(query)
        result = {"query_id": qid, "ndcg@k": ndcg_val, "faithfulness": faith_val}

        out_file = self.out / f"{qid}_evaluation.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Evaluation completed → {out_file}")
        return result

    # ------------------------------------------------------------------
    def evaluate_batch_from_logs(self, logs_dir: str = "data/logs", pattern: str = "llm_*.json") -> Dict[str, float]:
        """Run evaluation for all LLM log files and compute aggregate metrics."""
        files = sorted(Path(logs_dir).glob(pattern))
        nd_vals, fa_vals = [], []

        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                query = data.get("query") or data.get("user_query") or data.get("prompt") or ""
                model_output = data.get("model_output") or data.get("answer") or ""
                retrieved = data.get("retrieved_chunks") or data.get("context_snippets") or []

                # Normalize schema
                for rank, ch in enumerate(retrieved, start=1):
                    ch.setdefault("rank", rank)
                    ch.setdefault("final_score", ch.get("score", 0.0))
                    if "text" not in ch and "snippet" in ch:
                        ch["text"] = ch["snippet"]

                if not query or not retrieved:
                    logger.warning(f"Skipped incomplete log: {fp.name}")
                    continue

                res = self.evaluate_single(query, retrieved, model_output)
                nd_vals.append(float(res["ndcg@k"]))
                fa_vals.append(float(res["faithfulness"]))

            except Exception as e:
                err_path = self.out / f"{fp.stem}_eval_error.json"
                err_path.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
                logger.error(f"Evaluation failed for {fp.name}: {e}")

        summary = {
            "files": len(files),
            "mean_ndcg@k": float(sum(nd_vals) / len(nd_vals)) if nd_vals else 0.0,
            "mean_faithfulness": float(sum(fa_vals) / len(fa_vals)) if fa_vals else 0.0,
        }

        (self.out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Batch evaluation completed for {len(files)} logs.")
        return summary
