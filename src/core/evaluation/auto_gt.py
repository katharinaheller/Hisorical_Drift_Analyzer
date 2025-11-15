# src/core/evaluation/auto_gt.py
from __future__ import annotations
import re
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util


# Global citation pattern
_CIT_PATTERN = re.compile(r"\[(\d+)\]")


class AutoGroundTruth:
    """
    Automatic graded relevance labelling for retrieval evaluation.
    Uses same embedding model as your retrieval stack for consistency.
    """

    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1",
                 high_thr: float = 0.30, mid_thr: float = 0.15, low_thr: float = 0.07):
        self.model = SentenceTransformer(model_name)
        self.high_thr = high_thr
        self.mid_thr = mid_thr
        self.low_thr = low_thr

    def _extract_citations(self, output: str) -> List[int]:
        if not output:
            return []
        return [int(m.group(1)) for m in _CIT_PATTERN.finditer(output)]

    def build(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        if not answer or not retrieved_chunks:
            return {}

        # Encode once
        ans_emb = self.model.encode([answer], normalize_embeddings=True)

        labels: Dict[str, int] = {}
        cited = set(self._extract_citations(answer))

        for rank, ch in enumerate(retrieved_chunks, start=1):
            cid = ch.get("id") or f"auto::{rank}"
            text = ch.get("text", "") or ""
            chunk_emb = self.model.encode([text], normalize_embeddings=True)
            sim = float(util.cos_sim(ans_emb, chunk_emb)[0][0])

            # Citations count as strong relevance signals
            cited_here = int(ch.get("rank", rank)) in cited

            # graded relevance
            if cited_here and sim >= self.high_thr:
                rel = 3
            elif sim >= self.high_thr:
                rel = 2
            elif sim >= self.mid_thr:
                rel = 1
            elif sim >= self.low_thr:
                rel = 0
            else:
                rel = 0

            labels[cid] = int(rel)

        return labels
