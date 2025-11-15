from __future__ import annotations
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class DiversityPipeline:
    """Applies semantic and temporal diversity rules."""

    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model

    def _clean(self, t: str) -> str:
        return " ".join(t.replace("\n", " ").replace("\r", " ").split())

    def apply(self, ranked: List[Dict[str, Any]], k: int, historical: bool) -> List[Dict[str, Any]]:
        # Non-historical: simple dedupe
        if not historical:
            seen, out = set(), []
            for r in ranked:
                tx = (r.get("text") or "").strip()
                h = hash(tx)
                if h not in seen:
                    seen.add(h)
                    out.append(r)
                if len(out) >= k:
                    break
            return out

        # Historical: semantic & temporal diversity
        texts = [self._clean(r.get("text", "")) for r in ranked]
        idxs = [i for i, t in enumerate(texts) if t]
        embs = self.embed_model.encode([texts[i] for i in idxs], normalize_embeddings=True)

        selected, kept_embs = [], []
        for j, idx in enumerate(idxs):
            if len(selected) >= k:
                break
            cand = ranked[idx]
            emb = embs[j]
            if kept_embs:
                sims = util.cos_sim(emb, kept_embs)[0]
                if float(sims.max()) > 0.95:
                    continue
            selected.append(cand)
            kept_embs.append(emb)
        return selected
