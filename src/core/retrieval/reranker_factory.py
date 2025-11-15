# src/core/retrieval/reranker_factory.py
from __future__ import annotations
import logging
from typing import Any, Dict, Type

from src.core.retrieval.temporal_reranker import TemporalReranker
from src.core.retrieval.semantic_reranker import SemanticReranker
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """Factory for deterministic and clean reranker construction."""

    _registry: Dict[str, Type[IReranker]] = {
        "temporal": TemporalReranker,
        "semantic": SemanticReranker,
    }

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> IReranker:
        """Instantiate reranker from configuration dictionary with clean parameter mapping."""
        opts = cfg.get("options", {})
        rtype = str(opts.get("reranker", "semantic")).lower()

        if rtype not in RerankerFactory._registry:
            raise ValueError(
                f"Unsupported reranker type: {rtype}. "
                f"Available: {list(RerankerFactory._registry.keys())}"
            )

        cls = RerankerFactory._registry[rtype]
        logger.info(f"Initializing reranker of type='{rtype}'")

        # -------------------------------------------------------------
        # Temporal Reranker (Golden Middle Version)
        # -------------------------------------------------------------
        if cls is TemporalReranker:
            return TemporalReranker(
                semantic_threshold=float(opts.get("semantic_threshold", 0.40)),
                min_year=int(opts.get("min_year", 1900)),
                must_include=list(opts.get("must_include", [])),
                blacklist_sources=list(opts.get("blacklist_sources", [])),
            )

        # -------------------------------------------------------------
        # Semantic Reranker
        # -------------------------------------------------------------
        if cls is SemanticReranker:
            return SemanticReranker(
                model_name=str(
                    opts.get(
                        "semantic_model",
                        "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                ),
                top_k=int(opts.get("top_k_rerank", 25)),
                semantic_weight=float(opts.get("semantic_weight", 0.75)),
                use_gpu=bool(opts.get("use_gpu", False)),
            )
