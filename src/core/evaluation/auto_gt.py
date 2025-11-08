from __future__ import annotations
import re
import math
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# -----------------------------
# Citation parsing
# -----------------------------
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citation_indices(text: str) -> List[int]:
    # Parse numeric citations like [1], [2], [10]
    if not text:
        return []
    return [int(m.group(1)) for m in _CITATION_PATTERN.finditer(text)]

# -----------------------------
# Lightweight TF-IDF utilities
# -----------------------------
def _tokenize(s: str) -> List[str]:
    # Very simple alnum tokenizer, lowercased
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _tf(doc_tokens: List[str]) -> Dict[str, float]:
    # Term frequency as normalized counts
    c = Counter(doc_tokens)
    n = float(sum(c.values())) or 1.0
    return {t: v / n for t, v in c.items()}

def _idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    # Inverse document frequency with log smoothing
    N = float(len(docs_tokens)) or 1.0
    df = Counter()
    for dt in docs_tokens:
        df.update(set(dt))
    return {t: math.log((N + 1.0) / (df[t] + 1.0)) + 1.0 for t in df}

def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Sparse dot product
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())

def _norm(a: Dict[str, float]) -> float:
    # L2 norm
    return math.sqrt(sum(v * v for v in a.values())) or 1e-12

def cosine_tfidf(a_text: str, b_text: str) -> float:
    # Compute cosine similarity over small, local TF-IDF
    a_tok = _tokenize(a_text)
    b_tok = _tokenize(b_text)
    if not a_tok or not b_tok:
        return 0.0
    idf = _idf([a_tok, b_tok])
    a_vec = {t: _tf(a_tok).get(t, 0.0) * idf.get(t, 0.0) for t in set(a_tok)}
    b_vec = {t: _tf(b_tok).get(t, 0.0) * idf.get(t, 0.0) for t in set(b_tok)}
    return _dot(a_vec, b_vec) / (_norm(a_vec) * _norm(b_vec))

# -----------------------------
# Auto-GT grading
# -----------------------------
def auto_grade_relevance(
    query_id: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_output: str,
    high_sim: float = 0.35,
    mid_sim: float = 0.18,
    top_rank_bonus: int = 3
) -> Dict[str, int]:
    """
    Build graded relevance 0..3 without any manual labels.
    Rules:
      3: cited + sim >= high_sim
      2: cited + sim >= mid_sim (or strong)
      1: not cited + sim >= high_sim OR within top 'top_rank_bonus'
      0: else
    """
    # Map numeric citations [i] to chunk indices (1-based)
    cited_numbers = set(extract_citation_indices(model_output))
    # Prepare ranks and ids
    grades: Dict[str, int] = {}
    for rank, ch in enumerate(retrieved_chunks, start=1):
        cid = ch.get("id") or f"auto::{rank}"
        text = ch.get("text", "") or ""
        # Similarity against full answer
        sim = cosine_tfidf(model_output, text)
        # Citation check: try index field or fallback by heuristics
        idx = ch.get("index") or ch.get("rank") or rank
        cited = int(idx) in cited_numbers

        # Score-based grade
        if cited and sim >= high_sim:
            g = 3
        elif cited and sim >= mid_sim:
            g = 2
        elif (not cited) and (sim >= high_sim or rank <= top_rank_bonus):
            g = 1
        else:
            g = 0
        grades[cid] = int(g)

    return grades
