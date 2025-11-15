from __future__ import annotations
from typing import Dict, Any
import hashlib

def make_chunk_id(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) or {}
    src = meta.get("source_file") or meta.get("title") or "unknown"
    year = meta.get("year", "na")
    text = (chunk.get("text") or "")[:200]
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{src}::{year}::{h}"
