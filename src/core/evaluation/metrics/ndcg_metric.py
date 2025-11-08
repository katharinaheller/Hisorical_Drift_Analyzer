from __future__ import annotations
import math
from typing import List, Dict
from src.core.evaluation.interfaces.i_metric import IMetric

class NDCGMetric(IMetric):
    """Normalized Discounted Cumulative Gain (intrinsic retrieval metric)."""

    def __init__(self, k: int = 5):
        self.k = k

    def compute(self, relevance_scores: List[int]) -> float:
        """Compute NDCG@k from graded relevance list."""
        def dcg(scores: List[int]) -> float:
            return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:self.k]))
        if not relevance_scores:
            return 0.0
        ideal = sorted(relevance_scores, reverse=True)
        idcg = dcg(ideal)
        return (dcg(relevance_scores) / idcg) if idcg > 0 else 0.0

    def describe(self) -> Dict[str, str]:
        return {
            "name": "NDCG@k",
            "type": "intrinsic",
            "description": "Measures retrieval ranking quality using graded relevance with logarithmic discount."
        }
