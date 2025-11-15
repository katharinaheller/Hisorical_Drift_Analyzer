from __future__ import annotations
from typing import List, Dict
import numpy as np

class RelevanceAnnotator:
    """Assigns 0-3 relevance labels based on score distribution."""

    def apply(self, items: List[Dict], population: List[Dict]) -> List[Dict]:
        scores = np.array([float(x.get("final_score", 0.0)) for x in population])
        if scores.size == 0 or np.allclose(scores.std(), 0.0):
            for x in items:
                x["relevance"] = 1
            return items
        q1, q2, q3 = np.quantile(scores, [0.25, 0.5, 0.75])
        for x in items:
            s = float(x.get("final_score", 0.0))
            x["relevance"] = int(0 if s <= q1 else 1 if s <= q2 else 2 if s <= q3 else 3)
        return items
