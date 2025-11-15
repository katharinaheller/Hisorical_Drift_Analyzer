from __future__ import annotations
from typing import List, Dict, Any
from src.core.retrieval.faiss_retriever import FAISSRetriever

class RetrievalPipeline:
    """Handles broad FAISS retrieval only."""

    def __init__(self, retriever: FAISSRetriever, initial_k: int):
        self.retriever = retriever
        self.initial_k = initial_k

    def run(self, query: str, historical: bool) -> List[Dict[str, Any]]:
        # Perform broad retrieval without post-processing
        raw = self.retriever.search(
            query,
            top_k=self.initial_k,
            temporal_mode=historical,
        )
        for r in raw:
            r["query"] = query
        return raw
