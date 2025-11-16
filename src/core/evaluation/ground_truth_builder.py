# src/core/evaluation/ground_truth_builder.py
from __future__ import annotations
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Any, List
import numpy as np
import logging
from src.core.config.config_loader import ConfigLoader

logger = logging.getLogger("GroundTruthBuilder")


class GroundTruthBuilder:
    """
    Generates semantic ground truth labels for intrinsic retrieval evaluation (NDCG).
    The GT is computed by embedding the user query and measuring its similarity
    to each retrieved chunk. Higher similarity implies stronger relevance.
    
    This implementation is fully offline, consistent with your new
    local-embedding faithfulness metric, and uses the same semantic space
    as your retrieval stack.
    """

    def __init__(
        self,
        config_path: str = "configs/embedding.yaml",
        high_thr: float = 0.40,
        mid_thr: float = 0.25,
        low_thr: float = 0.10
    ):
        # Load the same embedding model used in retrieval
        cfg = ConfigLoader(config_path).config
        model_name = cfg.get("options", {}).get(
            "embedding_model", "multi-qa-mpnet-base-dot-v1"
        )
        self.model = SentenceTransformer(model_name)

        # Consistent thresholds with the new offline faithfulness metric
        self.high_thr = high_thr
        self.mid_thr = mid_thr
        self.low_thr = low_thr

    # ------------------------------------------------------------------
    def build(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        if not query or not retrieved_docs:
            return {}

        # Encode query embedding
        q_emb = self.model.encode([query], normalize_embeddings=True)

        truth: Dict[str, int] = {}

        for d in retrieved_docs:
            text = d.get("text", "") or ""
            doc_id = d.get("id") or f"{d.get('metadata', {}).get('source_file')}"

            # Encode chunk embedding
            d_emb = self.model.encode([text], normalize_embeddings=True)
            sim = float(util.cos_sim(q_emb, d_emb)[0][0])

            # Graded relevance assignment
            if sim >= self.high_thr:
                rel = 3
            elif sim >= self.mid_thr:
                rel = 2
            elif sim >= self.low_thr:
                rel = 1
            else:
                rel = 0

            truth[doc_id] = rel

        logger.info(f"Semantic GT created (avg rel={np.mean(list(truth.values())):.2f})")
        return truth