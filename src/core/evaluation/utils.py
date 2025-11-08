from __future__ import annotations
from typing import Dict, Any

def make_chunk_id(chunk: Dict[str, Any]) -> str:
    # Build a stable chunk id from metadata and content hash
    meta = chunk.get("metadata", {}) or {}
    src = meta.get("source_file") or meta.get("title") or "unknown"
    year = meta.get("year", "na")
    text = (chunk.get("text") or "")[:120]
    h = abs(hash(text)) % (10**8)
    return f"{src}::{year}::{h}"
