# src/core/evaluation/metrics/faithfulness_metric.py
from __future__ import annotations
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from src.core.evaluation.interfaces.i_metric import IMetric
import numpy as np
import re


class FaithfulnessMetric(IMetric):
    """
    Local embedding-based faithfulness metric.
    Evaluates whether each claim in the answer is semantically grounded
    in at least one retrieved context chunk.

    Scientific basis:
    - Claim-level semantic entailment via cosine similarity
    - Robust max-evidence matching (Zhang et al., 2023)
    - Aggregation over claim support ratios (similar to RAGAS structure)
    """

    def __init__(self,
                 model_name: str = "multi-qa-mpnet-base-dot-v1",
                 high_thr: float = 0.40,
                 mid_thr: float = 0.25):
        # Load local embedding model
        self.model = SentenceTransformer(model_name)
        self.high_thr = high_thr   # strong evidence
        self.mid_thr = mid_thr     # weak evidence

    # ------------------------------------------------------------
    def _split_into_claims(self, answer: str) -> List[str]:
        # Split into short, evidence-checkable units
        raw = re.split(r"[.!?]\s+", answer.strip())
        claims = [c.strip() for c in raw if len(c.strip().split()) >= 3]
        return claims

    # ------------------------------------------------------------
    def compute(self, context_chunks: List[str], answer: str) -> float:
        if not context_chunks or not answer:
            return 0.0

        claims = self._split_into_claims(answer)
        if not claims:
            return 0.0

        # Embeddings (all local, CPU/GPU depending on ST config)
        claim_emb = self.model.encode(claims, normalize_embeddings=True)
        ctx_emb = self.model.encode(context_chunks, normalize_embeddings=True)

        scores = []

        for i, emb in enumerate(claim_emb):
            # Similarity to all retrieved chunks
            sims = util.cos_sim(emb, ctx_emb)[0]  # shape = (num_chunks,)
            max_sim = float(sims.max())

            # Score mapping: high similarity → 1.0, weak → 0.5, no evidence → 0.0
            if max_sim >= self.high_thr:
                s = 1.0
            elif max_sim >= self.mid_thr:
                s = 0.5
            else:
                s = 0.0

            scores.append(s)

        # Faithfulness = mean over claim-level grounding scores
        return float(np.mean(scores))

    # ------------------------------------------------------------
    def describe(self) -> Dict[str, str]:
        return {
            "name": "Local-Embedding-Faithfulness",
            "type": "extrinsic",
            "description": (
                "Checks grounding of each answer claim in retrieved evidence via "
                "embedding-based max similarity across context chunks."
            )
        }
