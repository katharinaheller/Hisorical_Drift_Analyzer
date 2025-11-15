from __future__ import annotations
from typing import List, Dict, Any
from src.core.retrieval.reranker_factory import RerankerFactory

class RerankingPipeline:
    """Unified reranking abstraction."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.cached_type = None
        self.cached_reranker = None

    def _get(self, intent: str):
        rtype = "temporal" if intent == "chronological" else "semantic"
        if rtype != self.cached_type:
            self.cached_reranker = RerankerFactory.from_config({"options": {"reranker": rtype}})
            self.cached_type = rtype
        return self.cached_reranker

    def run(self, docs: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
        reranker = self._get(intent)
        ranked = reranker.rerank(docs, top_k=len(docs))
        for d in ranked:
            d["final_score"] = float(d.get("final_score", d.get("score", 0.0)))
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked
