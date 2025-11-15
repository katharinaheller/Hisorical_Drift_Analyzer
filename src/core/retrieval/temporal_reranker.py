from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple, Dict, Optional
from collections import defaultdict
from src.core.retrieval.interfaces.i_reranker import IReranker

logger = logging.getLogger(__name__)

class TemporalReranker(IReranker):
    """
    Semantic-first temporal diversification.
    Ensures that each decade contributes at most one document,
    but only if semantic relevance is sufficiently high.
    No score manipulation. No age penalties or boosts.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.40,  # minimal score required for decade coverage
        min_year: int = 1900,
        must_include: List[str] | None = None,
        blacklist_sources: List[str] | None = None
    ):
        self.semantic_threshold = semantic_threshold
        self.min_year = min_year
        self.must_include = must_include or []
        self.blacklist_sources = blacklist_sources or []

    # ------------------------------------------------------------------
    def rerank(self, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not results:
            return []

        # Extract valid years
        for r in results:
            r["year"] = self._extract_year(r)

        # Apply blacklists
        results = [r for r in results if not self._is_blacklisted(r)]
        if not results:
            return []

        # ---- 1. Pure semantic sort ----
        results_sorted = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)

        # ---- 2. Group by decade ----
        decade_groups = self._group_by_decade(results_sorted)

        # ---- 3. Pick top semantic document per decade (if strong) ----
        selected = []
        seen = set()

        for dec in sorted(decade_groups.keys()):
            best = decade_groups[dec][0]
            if best["score"] >= self.semantic_threshold:
                key = self._src_key(best)
                if key not in seen:
                    selected.append(best)
                    seen.add(key)

        # ---- 4. Fill remaining purely by semantic score ----
        for r in results_sorted:
            key = self._src_key(r)
            if key not in seen:
                selected.append(r)
                seen.add(key)
            if len(selected) >= top_k:
                break

        # ---- 5. Inject must-include sources at front ----
        selected = self._inject_must_include(selected, results, top_k)

        logger.info(f"Temporal diversification complete | decades={len(decade_groups)} | threshold={self.semantic_threshold}")
        return selected[:top_k]

    # ------------------------------------------------------------------
    def _extract_year(self, r):
        meta = r.get("metadata", {})
        y = meta.get("year") or r.get("year")
        try:
            y = int(str(y))
            if y < self.min_year or y > 2100:
                return self.min_year
            return y
        except Exception:
            return self.min_year

    def _group_by_decade(self, results):
        groups = defaultdict(list)
        for r in results:
            decade = (r["year"] // 10) * 10
            groups[decade].append(r)
        return groups

    def _src_key(self, r):
        meta = r.get("metadata", {})
        return (meta.get("source_file") or meta.get("title") or "unknown").lower()

    def _is_blacklisted(self, r):
        key = self._src_key(r)
        return any(b.lower() in key for b in self.blacklist_sources)

    def _inject_must_include(self, ranked, all_results, top_k):
        must = [r for r in all_results if any(m in self._src_key(r) for m in self.must_include)]
        if not must:
            return ranked
        merged, seen = [], set()
        for r in must + ranked:
            key = self._src_key(r)
            if key not in seen:
                merged.append(r)
                seen.add(key)
            if len(merged) >= top_k:
                break
        return merged
