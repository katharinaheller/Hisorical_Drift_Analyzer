from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Dict, Set
from src.core.evaluation.faithfulness_scorer import FaithfulnessScorer


class Evaluator:
    """Evaluates RAG system performance using NDCG@k and Faithfulness."""

    def __init__(self, output_dir: str = "data/eval_logs", k: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.faithfulness = FaithfulnessScorer()

    # --------------------------------------------------------------
    # Intrinsic metric: Normalized Discounted Cumulative Gain
    # --------------------------------------------------------------
    def ndcg_at_k(self, relevance: List[int]) -> float:
        """Compute NDCG@k given a list of graded relevance scores."""
        def dcg(scores):
            return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:self.k]))
        ideal = sorted(relevance, reverse=True)
        idcg = dcg(ideal)
        return (dcg(relevance) / idcg) if idcg > 0 else 0.0

    # --------------------------------------------------------------
    # Extrinsic metric: Faithfulness between answer and context
    # --------------------------------------------------------------
    def faithfulness_score(self, context: List[str], answer: str) -> float:
        """Compute semantic faithfulness between context and LLM output."""
        return self.faithfulness.compute(context, answer)

    # --------------------------------------------------------------
    # Unified evaluation entry point
    # --------------------------------------------------------------
    def evaluate(self, query_id: str, retrieved_chunks: List[Dict], answer: str) -> Dict:
        """Evaluate retrieval and generation performance for a query."""
        relevance = [c.get("relevance", 0) for c in retrieved_chunks]
        ndcg = self.ndcg_at_k(relevance)
        faith = self.faithfulness_score([c["text"] for c in retrieved_chunks], answer)

        results = {"query_id": query_id, "ndcg@k": ndcg, "faithfulness": faith}
        self._log_results(results)
        return results

    def _log_results(self, metrics: Dict) -> None:
        """Persist evaluation results in JSON for reproducibility."""
        log_file = self.output_dir / f"{metrics['query_id']}_eval.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
