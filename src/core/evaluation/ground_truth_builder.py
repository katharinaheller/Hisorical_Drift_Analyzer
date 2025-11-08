from __future__ import annotations
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger("GroundTruthBuilder")

class GroundTruthBuilder:
    """Generates automatic, semantic ground-truth relevance labels."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", high_thr: float = 0.35, mid_thr: float = 0.20):
        self.model = SentenceTransformer(model_name)
        self.high_thr = high_thr
        self.mid_thr = mid_thr

    # ------------------------------------------------------------------
    def build(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Return dict of {doc_id: relevance_score (0..3)}."""
        if not query or not retrieved_docs:
            return {}

        q_emb = self.model.encode(query, normalize_embeddings=True)
        truth: Dict[str, int] = {}
        for d in retrieved_docs:
            text = d.get("text", "")
            doc_id = d.get("id") or f"{d.get('metadata', {}).get('source_file')}"
            d_emb = self.model.encode(text, normalize_embeddings=True)
            sim = float(util.cos_sim(q_emb, d_emb))
            if sim >= self.high_thr:
                rel = 3
            elif sim >= self.mid_thr:
                rel = 2
            elif sim >= self.mid_thr / 2:
                rel = 1
            else:
                rel = 0
            truth[doc_id] = rel

        logger.info(f"Generated semantic ground truth for query (avg rel={np.mean(list(truth.values())):.2f})")
        return truth
