from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer, util


class FaithfulnessScorer:
    """Approximates RAGAS-style faithfulness via cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute(self, context_chunks: List[str], answer: str) -> float:
        """Compute mean cosine similarity between context and answer embeddings."""
        if not context_chunks or not answer:
            return 0.0
        ctx_embeds = self.model.encode(context_chunks, convert_to_tensor=True)
        ans_embed = self.model.encode([answer], convert_to_tensor=True)
        sims = util.cos_sim(ans_embed, ctx_embeds)[0]
        return float(sims.mean().item())
