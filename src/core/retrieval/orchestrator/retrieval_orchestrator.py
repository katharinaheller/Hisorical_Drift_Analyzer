from __future__ import annotations
import logging
from typing import Dict, Any, List
from src.core.config.config_loader import ConfigLoader
from sentence_transformers import SentenceTransformer
from src.core.retrieval.faiss_retriever import FAISSRetriever

from src.core.retrieval.orchestrator.retrieval_pipeline import RetrievalPipeline
from src.core.retrieval.orchestrator.reranking_pipeline import RerankingPipeline
from src.core.retrieval.orchestrator.diversity_pipeline import DiversityPipeline
from src.core.retrieval.orchestrator.relevance_annotator import RelevanceAnnotator
from src.core.retrieval.orchestrator.final_selector import FinalSelector

from src.core.evaluation.utils import make_chunk_id


class RetrievalOrchestrator:
    """Clean multi-stage retrieval orchestrator for RAG."""

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        self.logger = logging.getLogger("RetrievalOrchestrator")
        cfg_loader = ConfigLoader(config_path)
        self.cfg = cfg_loader.config

        opts = self.cfg["options"]
        paths = self.cfg["paths"]

        # Multi-stage parameters
        self.final_k = int(opts.get("final_k", 10))
        oversample = int(opts.get("oversample_factor", 15))
        self.initial_k = max(self.final_k * oversample, self.final_k * 8)

        # Embedding model
        self.embed_model = SentenceTransformer(opts["embedding_model"])

        # Retriever instance
        self.retriever = FAISSRetriever(
            vector_store_dir=paths["vector_store_dir"],
            model_name=opts["embedding_model"],
            top_k_retrieve=self.initial_k,
            normalize_embeddings=True,
            use_gpu=opts.get("use_gpu", False),
            similarity_metric=opts.get("similarity_metric", "cosine"),
            temporal_awareness=False,
            diversify_sources=opts.get("diversify_sources", True),
        )

        # Pipeline modules
        self.stage_retrieve = RetrievalPipeline(self.retriever, self.initial_k)
        self.stage_rerank = RerankingPipeline(self.cfg)
        self.stage_diversity = DiversityPipeline(self.embed_model)
        self.stage_label = RelevanceAnnotator()
        self.stage_select = FinalSelector()

    # ------------------------------------------------------------------
    def retrieve(self, query: str, intent: str) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        historical = intent == "chronological"

        # Stage 1: Broad retrieval
        raw = self.stage_retrieve.run(query, historical)

        # Stage 2: Full reranking
        ranked = self.stage_rerank.run(raw, intent)

        # Stage 3: Diversity
        diversified = self.stage_diversity.apply(ranked, self.final_k, historical)

        # Stage 4: Relevance annotation
        annotated = self.stage_label.apply(diversified, ranked)

        # Stage 5: Final-k selection
        final = self.stage_select.select(annotated, self.final_k)

        # Assign stable IDs and ranks
        for i, x in enumerate(final, start=1):
            x["rank"] = i
            if not x.get("id"):
                x["id"] = make_chunk_id(x)

        return final

    # ------------------------------------------------------------------
    def close(self):
        self.retriever.close()
