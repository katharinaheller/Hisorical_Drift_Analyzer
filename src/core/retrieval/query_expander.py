# src/core/retrieval/query_expander.py
from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import logging

# Third-party, bewusst etablierte Bibliotheken
from keybert import KeyBERT                            # # transformer-basierte Keyword-Extraktion
from sentence_transformers import SentenceTransformer, util  # # Paraphrasen/Ähnlichkeit

logger = logging.getLogger(__name__)


@dataclass
class HybridQueryExpanderConfig:
    # # Modellnamen sind bewusst konservativ und lokal verfügbar
    kw_model_name: str = "all-MiniLM-L6-v2"            # # für KeyBERT
    paraphrase_model_name: str = "paraphrase-MiniLM-L6-v2"  # # für Paraphrasen
    top_n_keywords: int = 5                            # # Anzahl semantischer Keywords
    num_paraphrases: int = 5                           # # Anzahl Paraphrasen aus Templates
    mmr_diversity: float = 0.7                         # # Diversität für KeyBERT
    max_total: int = 12                                # # Obergrenze der Gesamtvarianten inkl. Original
    dedup_lower: bool = True                           # # Deduplizierung fallunabhängig


class HybridQueryExpander:
    """Adaptive Query-Expansion über KeyBERT (Keywords) + Sentence-Transformers (Paraphrasen)."""

    def __init__(self, cfg: HybridQueryExpanderConfig | None = None):
        self.cfg = cfg or HybridQueryExpanderConfig()
        # # Modelle werden einmalig geladen und wiederverwendet
        self.kw_model = KeyBERT(self.cfg.kw_model_name)                         # # Keyword-Extraktor
        self.pp_model = SentenceTransformer(self.cfg.paraphrase_model_name)     # # Paraphrasen/Ähnlichkeit

    # ------------------------------------------------------------------
    def _expand_keywords(self, query: str) -> List[str]:
        # # Transformer-basierte Schlüsselbegriffe (MMR für Diversität)
        kw: List[Tuple[str, float]] = self.kw_model.extract_keywords(
            query,
            top_n=self.cfg.top_n_keywords,
            use_mmr=True,
            diversity=self.cfg.mmr_diversity
        )
        out: List[str] = []
        for term, _ in kw:
            # # Minimalistische, robuste Variantenbildung ohne Hardcoding von Themenlisten
            out.append(f"{query} {term}")                 # # query + zentrales Semantikwort
            out.append(f"{term}")                         # # reines Semantikwort als eigenständige Fokussierung
        return out

    # ------------------------------------------------------------------
    def _expand_paraphrases(self, query: str) -> List[str]:
        # # Schmale, generische Templates, die typisierte Perspektiven abdecken
        templates = [
            f"evolution of {query}",
            f"historical perspective on {query}",
            f"foundations of {query}",
            f"modern view of {query}",
            f"how {query} changed over time",
            f"definition and background of {query}",
            f"comparative view on {query}",
        ]
        q_emb = self.pp_model.encode(query, convert_to_tensor=True)
        t_emb = self.pp_model.encode(templates, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(q_emb, t_emb)[0]
        idx = sims.argsort(descending=True)
        return [templates[i] for i in idx[: self.cfg.num_paraphrases]]

    # ------------------------------------------------------------------
    def expand(self, query: str) -> List[str]:
        # # Original immer an Position 0
        candidates: List[str] = [query]

        try:
            candidates += self._expand_keywords(query)
        except Exception as e:
            logger.warning(f"KeyBERT expansion failed, continuing without keywords: {e}")

        try:
            candidates += self._expand_paraphrases(query)
        except Exception as e:
            logger.warning(f"Paraphrase expansion failed, continuing without paraphrases: {e}")

        # # Deduplizierung, stabil und deterministisch
        seen = set()
        out: List[str] = []
        for c in candidates:
            key = c.lower() if self.cfg.dedup_lower else c
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= self.cfg.max_total:
                break

        return out
