# src/core/retrieval/retrieval_orchestrator.py
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.core.retrieval.faiss_retriever import FAISSRetriever
from src.core.retrieval.reranker_factory import RerankerFactory
from src.core.config.config_loader import ConfigLoader
from src.core.evaluation.utils import make_chunk_id  # # stable id builder for evaluation


class RetrievalOrchestrator:
    """Unified retrieval orchestrator — orchestrates query → retrieval → reranking pipeline."""

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        self.logger = logging.getLogger("RetrievalOrchestrator")
        cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = cfg_loader.config

        opts = self.cfg.get("options", {})
        paths = self.cfg.get("paths", {})

        # Core parameters
        self.top_k = int(opts.get("top_k", 10))
        self.vector_store_dir = str(paths.get("vector_store_dir", "data/vector_store"))
        self.embedding_model = opts.get("embedding_model", "multi-qa-mpnet-base-dot-v1")
        self.diversify_sources_default = bool(opts.get("diversify_sources", True))
        oversample_factor = int(opts.get("oversample_factor", 10))
        self.max_initial = max(self.top_k * oversample_factor, self.top_k * 8)

        # Validate embedding model
        try:
            self.embed_model = SentenceTransformer(self.embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.embedding_model}': {e}")

        # Initialize FAISS retriever
        self.retriever = FAISSRetriever(
            vector_store_dir=self.vector_store_dir,
            model_name=self.embedding_model,
            top_k_retrieve=self.max_initial,
            normalize_embeddings=True,
            use_gpu=False,
            similarity_metric="cosine",
            temporal_awareness=False,
            diversify_sources=self.diversify_sources_default,
        )

        # Cached reranker instance
        self._cached_reranker_type: Optional[str] = None
        self._cached_reranker = None

        self.logger.info(
            f"RetrievalOrchestrator initialized | top_k={self.top_k} | "
            f"default_diversify={self.diversify_sources_default} | model={self.embedding_model}"
        )

    # ------------------------------------------------------------------
    def retrieve(self, query: str, intent: str) -> List[Dict[str, Any]]:
        """Retrieve, rerank, and return top-k chunks based on query and detected intent."""
        if not query or not query.strip():
            self.logger.warning("Empty query ignored.")
            return []

        # Intent-dependent configuration
        is_historical = intent == "chronological"

        # Dynamically control retrieval behaviour
        self.retriever.temporal_awareness = is_historical
        self.retriever.diversify_sources = is_historical  # only enable for chronological mode

        self.logger.info(
            f"Retrieval started | intent={intent} | top_k={self.top_k} | "
            f"temporal_awareness={self.retriever.temporal_awareness} | "
            f"diversify_sources={self.retriever.diversify_sources}"
        )

        # --- initial retrieval ---
        try:
            raw_results = self.retriever.search(
                query, top_k=self.max_initial, temporal_mode=is_historical
            )
        except Exception as e:
            self.logger.exception(f"FAISS retrieval failed: {e}")
            return []

        if not raw_results:
            self.logger.warning("No retrieval results found.")
            return []

        # --- inject query text for reranker ---
        for r in raw_results:
            r["query"] = query.strip()

        # --- dynamic reranker selection ---
        reranker_type = "temporal" if is_historical else "semantic"
        if reranker_type != self._cached_reranker_type or self._cached_reranker is None:
            self._cached_reranker = RerankerFactory.from_config(
                {"options": {"reranker": reranker_type}}
            )
            self._cached_reranker_type = reranker_type

        # --- reranking phase ---
        try:
            reranked = self._cached_reranker.rerank(raw_results, top_k=len(raw_results))
        except Exception as e:
            self.logger.exception(f"Reranking failed ({reranker_type}): {e}")
            reranked = raw_results

        # --- normalization + sorting ---
        for x in reranked:
            x["final_score"] = float(x.get("final_score", x.get("score", 0.0)) or 0.0)
        reranked.sort(key=lambda x: (x["final_score"], x.get("id", "")), reverse=True)

        # --- apply diversity + relevance scoring ---
        diversified = self._enforce_diversity(reranked, self.top_k, is_historical)
        diversified = self._attach_graded_relevance(diversified, reranked)
        final = self._ensure_exact_k(diversified, self.top_k)

        # --- finalize ranks + log stats ---
        for i, x in enumerate(final, start=1):
            if not x.get("id"):
                x["id"] = make_chunk_id(x)
            x["rank"] = i

        self._log_decade_distribution(final)
        self.logger.info(
            f"Retrieval finished | returned={len(final)} | mode={intent} | "
            f"diversity={self.retriever.diversify_sources}"
        )
        return final

    # ------------------------------------------------------------------
    def _enforce_diversity(
        self, results: List[Dict[str, Any]], k: int, historical: bool
    ) -> List[Dict[str, Any]]:
        """Diversify retrieval results by source and decade (for chronological mode)."""
        if not results:
            return []

        if not historical or not self.retriever.diversify_sources:
            # Simple deduplication without diversity
            seen, out = set(), []
            for r in results:
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                h = hash(text)
                if h in seen:
                    continue
                seen.add(h)
                out.append(r)
                if len(out) >= k:
                    break
            return out

        # Historical mode: enforce semantic and temporal spread
        selected, used_sources, used_decades = [], set(), set()
        pool_texts = [self._clean_text(r.get("text", "")) for r in results]
        pool_idxs = [i for i, t in enumerate(pool_texts) if t]
        if not pool_idxs:
            return results[:k]

        embs = self.embed_model.encode(
            [pool_texts[i] for i in pool_idxs], normalize_embeddings=True
        )
        kept_embs = []

        for j, idx in enumerate(pool_idxs):
            r = results[idx]
            if len(selected) >= k:
                break
            meta = r.get("metadata", {}) or {}
            src = (meta.get("source_file") or "unknown").lower()
            year = self._safe_year(r)
            decade = (year // 10) * 10 if year else None

            if src in used_sources and decade in used_decades and len(selected) < int(k * 0.8):
                continue

            cand_emb = embs[j]
            if kept_embs:
                sims = util.cos_sim(cand_emb, kept_embs)[0]
                if float(sims.max()) > 0.95:
                    continue

            selected.append(r)
            kept_embs.append(cand_emb)
            used_sources.add(src)
            if decade:
                used_decades.add(decade)

        if len(selected) < k:
            used_ids = {id(x) for x in selected}
            for r in results:
                if id(r) not in used_ids:
                    selected.append(r)
                    if len(selected) >= k:
                        break
        return selected

    # ------------------------------------------------------------------
    def _attach_graded_relevance(
        self, items: List[Dict[str, Any]], ref_population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assign relevance labels (0–3) based on score quantiles."""
        if not items:
            return items

        scores = np.array(
            [float(x.get("final_score", x.get("score", 0.0)) or 0.0) for x in ref_population],
            dtype=float,
        )
        if scores.size == 0 or np.allclose(scores.std(), 0.0):
            for x in items:
                x["relevance"] = 1
            return items

        try:
            q1, q2, q3 = np.quantile(scores, [0.25, 0.5, 0.75])
        except Exception:
            smin, smax = float(scores.min()), float(scores.max())
            step = (smax - smin) / 4.0 if smax > smin else 1.0
            q1, q2, q3 = smin + step, smin + 2 * step, smin + 3 * step

        for x in items:
            s = float(x.get("final_score", x.get("score", 0.0)) or 0.0)
            x["relevance"] = int(0 if s <= q1 else 1 if s <= q2 else 2 if s <= q3 else 3)
        return items

    # ------------------------------------------------------------------
    def _ensure_exact_k(self, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Guarantee exactly k output results (pad if necessary)."""
        if not results:
            return []
        if len(results) > k:
            return results[:k]
        if len(results) < k:
            pad = results[-1].copy()
            results.extend(pad.copy() for _ in range(k - len(results)))
        return results

    # ------------------------------------------------------------------
    def _safe_year(self, r: Dict[str, Any]) -> Optional[int]:
        """Safely extract year from metadata."""
        meta = r.get("metadata", {}) or {}
        y = meta.get("year", r.get("year"))
        try:
            y = int(y)
            if 1900 <= y <= 2100:
                return y
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _clean_text(self, t: str) -> str:
        """Normalize whitespace for embedding stability."""
        if not t:
            return ""
        return " ".join(t.replace("\n", " ").replace("\r", " ").split())

    # ------------------------------------------------------------------
    def _log_decade_distribution(self, items: List[Dict[str, Any]]) -> None:
        """Log temporal distribution of retrieved documents."""
        hist: Dict[str, int] = defaultdict(int)
        for r in items:
            y = self._safe_year(r)
            decade = f"{(y // 10) * 10}s" if y else "unknown"
            hist[decade] += 1
        msg = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
        self.logger.info(f"Decade distribution: {msg}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close retriever resources."""
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Failed to close retriever: {e}")
        self.logger.info("RetrievalOrchestrator closed cleanly.")
