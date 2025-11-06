# src/core/retrieval/faiss_retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import re
import numpy as np
import logging
from math import exp
from src.core.retrieval.interfaces.i_retriever import IRetriever


class FAISSRetriever(IRetriever):
    """FAISS-based semantic retriever with optional temporal and source diversification."""

    def __init__(
        self,
        vector_store_dir: str,
        model_name: str,
        top_k_retrieve: int = 50,
        normalize_embeddings: bool = True,
        use_gpu: bool = False,
        similarity_metric: str = "cosine",
        temporal_awareness: bool = True,
        temporal_tau: float = 8.0,
        temporal_weight: float = 0.30,
        valid_year_range: Tuple[int, int] = (1900, 2100),
        diversify_sources: bool = True,      # # enable balanced source selection
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "faiss-cpu and sentence-transformers are required. "
                "Install via: poetry add faiss-cpu sentence-transformers"
            ) from e

        self.faiss = faiss
        self.vector_store_dir = Path(vector_store_dir).resolve()
        self.index_path = self.vector_store_dir / "index.faiss"
        self.meta_path = self.vector_store_dir / "metadata.jsonl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Incomplete vector store: {self.vector_store_dir}")

        self.model = SentenceTransformer(model_name)
        self.top_k_retrieve = int(top_k_retrieve)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_gpu = bool(use_gpu)
        self.similarity_metric = similarity_metric.lower().strip()
        self.temporal_awareness = bool(temporal_awareness)
        self.temporal_tau = float(temporal_tau)
        self.temporal_weight = float(temporal_weight)
        self.valid_year_range = valid_year_range
        self.diversify_sources = bool(diversify_sources)

        if self.similarity_metric not in {"cosine", "dot"}:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        # Load FAISS index
        self.logger.info(f"Loading FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS GPU acceleration enabled")
            except Exception as e:
                self.logger.warning(f"GPU mode failed, falling back to CPU: {e}")

        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.logger.info(
            f"FAISSRetriever ready | entries={len(self.metadata)} | metric={self.similarity_metric.upper()} "
            f"| temporal_awareness={self.temporal_awareness} | diversify_sources={self.diversify_sources}"
        )

    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        # Encode query with normalization
        vec = self.model.encode([query], normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vec, dtype="float32")

    def _normalize_scores(self, distances: np.ndarray) -> np.ndarray:
        # Normalize distances depending on metric
        if self.similarity_metric in {"cosine", "dot"}:
            return distances
        return 1 - distances

    def _extract_years_from_query(self, query: str) -> List[int]:
        yrs = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
        lo, hi = self.valid_year_range
        return [y for y in yrs if lo <= y <= hi]

    def _temporal_modulate(self, base_score: float, doc_year: Optional[int], query_years: List[int]) -> float:
        # Exponential weighting for temporal alignment
        if doc_year is None or not query_years:
            return base_score
        nearest = min(abs(doc_year - y) for y in query_years)
        w = exp(-nearest / max(self.temporal_tau, 1e-6))
        return base_score * (1.0 + self.temporal_weight * w)

    def _safe_doc_year(self, entry: Dict[str, Any]) -> Optional[int]:
        meta = entry.get("metadata", {}) or {}
        y = meta.get("year")
        try:
            yi = int(str(y))
            lo, hi = self.valid_year_range
            if lo <= yi <= hi:
                return yi
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _apply_source_diversity(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Ensure chunks from different PDFs dominate early ranks."""
        if not self.diversify_sources or not results:
            return results[:top_k]

        diversified, seen = [], set()
        for r in results:
            src = r["metadata"].get("source_file", "unknown")
            if src not in seen:
                diversified.append(r)
                seen.add(src)
            if len(diversified) >= top_k:
                break

        # Fill up with remaining if fewer than top_k unique
        if len(diversified) < top_k:
            for r in results:
                if r not in diversified:
                    diversified.append(r)
                if len(diversified) >= top_k:
                    break
        return diversified

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None, temporal_mode: bool = True) -> List[Dict[str, Any]]:
        """Similarity search with optional temporal and source diversification."""
        self.temporal_awareness = bool(temporal_mode)
        k = int(top_k or self.top_k_retrieve)
        k = max(1, min(k, self.index.ntotal))

        q_vec = self._encode_query(query)
        D, I = self.index.search(q_vec, k * 5 if self.diversify_sources else k)
        scores = self._normalize_scores(D[0])
        query_years = self._extract_years_from_query(query) if self.temporal_awareness else []

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            doc_year = self._safe_doc_year(entry)
            mod_score = (
                self._temporal_modulate(float(score), doc_year, query_years)
                if self.temporal_awareness else float(score)
            )
            results.append({
                "score": float(mod_score),
                "text": (entry.get("text", "") or "")[:500],
                "metadata": entry.get("metadata", {}) or {},
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        diversified = self._apply_source_diversity(results, top_k=k)

        self.logger.info(
            f"Retrieved {len(diversified)} candidates | temporal_mode={self.temporal_awareness} "
            f"| diversify_sources={self.diversify_sources} | years_in_query={query_years or 'none'}"
        )
        return diversified

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.logger.info("FAISS retriever closed")
