from __future__ import annotations
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from src.core.evaluation.interfaces.i_metric import IMetric

class FaithfulnessMetric(IMetric):
    """Semantic faithfulness between generated answer and retrieved context (extrinsic)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lightweight sentence transformer
        self.model = SentenceTransformer(model_name)

    def compute(self, context_chunks: List[str], answer: str) -> float:
        """Compute mean cosine similarity between answer and context embeddings."""
        if not context_chunks or not answer:
            return 0.0
        ctx_embeds = self.model.encode(context_chunks, convert_to_tensor=True)
        ans_embed = self.model.encode([answer], convert_to_tensor=True)
        sims = util.cos_sim(ans_embed, ctx_embeds)[0]
        return float(sims.mean().item())

    def describe(self) -> Dict[str, str]:
        return {
            "name": "Faithfulness",
            "type": "extrinsic",
            "description": "Mean cosine similarity between retrieved context embeddings and LLM answer embedding."
        }
