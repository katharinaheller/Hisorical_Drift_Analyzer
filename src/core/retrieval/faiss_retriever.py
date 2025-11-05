# src/core/retrieval/faiss_retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import re
import numpy as np
import logging
from math import exp
from src.core.retrieval.interfaces.i_retriever import IRetriever


class FAISSRetriever(IRetriever):
    """FAISS-basierter semantischer Retriever mit optionaler zeitlicher Gewichtung."""

    def __init__(
        self,
        vector_store_dir: str,
        model_name: str,
        top_k_retrieve: int = 50,             # # breiter Vorabruf
        normalize_embeddings: bool = True,
        use_gpu: bool = False,
        similarity_metric: str = "cosine",    # # 'cosine' oder 'dot'
        temporal_awareness: bool = True,      # # zeitliche Prior-Verstärkung aktivieren
        temporal_tau: float = 8.0,            # # Glättung für |year - query_year|
        temporal_weight: float = 0.30,        # # Stärke der zeitlichen Modulation
        valid_year_range: tuple[int, int] = (1900, 2100),  # # Plausibler Jahrbereich
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "faiss-cpu und sentence-transformers sind erforderlich. "
                "Installiere via: poetry add faiss-cpu sentence-transformers"
            ) from e

        self.faiss = faiss
        self.vector_store_dir = Path(vector_store_dir).resolve()
        self.index_path = self.vector_store_dir / "index.faiss"
        self.meta_path = self.vector_store_dir / "metadata.jsonl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Vector store unvollständig: {self.vector_store_dir}")

        self.model = SentenceTransformer(model_name)
        self.top_k_retrieve = top_k_retrieve
        self.normalize_embeddings = normalize_embeddings
        self.use_gpu = use_gpu
        self.similarity_metric = similarity_metric.lower().strip()
        self.temporal_awareness = temporal_awareness
        self.temporal_tau = float(temporal_tau)
        self.temporal_weight = float(temporal_weight)
        self.valid_year_range = valid_year_range

        if self.similarity_metric not in {"cosine", "dot"}:
            raise ValueError(f"Nicht unterstützte Ähnlichkeitsmetrik: {self.similarity_metric}")

        # # Index laden und optional auf GPU verschieben
        self.logger.info(f"Lade FAISS-Index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS GPU-Beschleunigung aktiv")
            except Exception as e:
                self.logger.warning(f"GPU-Modus fehlgeschlagen, CPU wird genutzt: {e}")

        # # Metadaten einmalig in den Speicher
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        self.logger.info(
            f"FAISSRetriever initialisiert | entries={len(self.metadata)} | metric={self.similarity_metric.upper()} "
            f"| temporal_awareness={self.temporal_awareness}"
        )

    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        # # Einbettung mit optionaler Normalisierung
        vec = self.model.encode([query], normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vec, dtype="float32")

    # ------------------------------------------------------------------
    def _normalize_scores(self, distances: np.ndarray) -> np.ndarray:
        # # FAISS-IP bei Cosine: höher ist besser, daher direkte Verwendung
        if self.similarity_metric in {"cosine", "dot"}:
            return distances
        return 1 - distances

    # ------------------------------------------------------------------
    def _extract_years_from_query(self, query: str) -> List[int]:
        # # Robuste Jahr-Extraktion, nur plausibler Bereich
        yrs = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
        lo, hi = self.valid_year_range
        return [y for y in yrs if lo <= y <= hi]

    # ------------------------------------------------------------------
    def _temporal_modulate(self, base_score: float, doc_year: Optional[int], query_years: List[int]) -> float:
        # # Exponentieller Abstand, nähere Jahre → höheres Gewicht
        if doc_year is None or not query_years:
            return base_score
        nearest = min(abs(doc_year - y) for y in query_years)
        # # Gewicht in [0,1], dann als additive Modulation auf den Basisscore
        w = exp(-nearest / max(self.temporal_tau, 1e-6))
        return base_score * (1.0 + self.temporal_weight * w)

    # ------------------------------------------------------------------
    def _safe_doc_year(self, entry: Dict[str, Any]) -> Optional[int]:
        # # Jahr aus Metadaten, robust geparst
        meta = entry.get("metadata", {}) or {}
        y = meta.get("year", None)
        try:
            yi = int(str(y))
            lo, hi = self.valid_year_range
            if lo <= yi <= hi:
                return yi
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        # # Ähnlichkeitssuche mit optionaler zeitlicher Modulation
        k = top_k or self.top_k_retrieve
        k = max(1, min(k, self.index.ntotal))

        q_vec = self._encode_query(query)
        self.logger.debug(f"FAISS search | k={k} | query='{query[:60]}'")

        D, I = self.index.search(q_vec, k)
        scores = self._normalize_scores(D[0])

        results: List[Dict[str, Any]] = []
        query_years = self._extract_years_from_query(query) if self.temporal_awareness else []

        for score, idx in zip(scores, I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            doc_year = self._safe_doc_year(entry)
            mod_score = self._temporal_modulate(float(score), doc_year, query_years) if self.temporal_awareness else float(score)

            results.append({
                "score": float(mod_score),                       # # ggf. zeitlich moduliert
                "text": (entry.get("text", "") or "")[:500],     # # Vorschau begrenzen
                "metadata": entry.get("metadata", {}) or {},
            })

        # # Sortierung nach moduliertem Score sicherstellen
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        self.logger.info(f"Retrieved {len(results)} candidates for query: '{query[:60]}' "
                         f"| years_in_query={query_years if query_years else 'none'}")
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        # # Ressourcenfreundlicher Abschluss
        self.logger.info("FAISS retriever closed")
