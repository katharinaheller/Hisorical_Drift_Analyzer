from __future__ import annotations
from typing import Any, Dict, List
from src.core.ingestion.chunking.i_chunker import IChunker


class AdaptiveChunker(IChunker):
    """Chunker that adapts to the content structure by splitting at semantic breaks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 200, min_chunk_length: int = 400):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk = ""
        metadata = metadata or {}

        # Split text into paragraphs or sentences (based on full stops, etc.)
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # Add paragraph to current chunk
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += "\n\n" + paragraph
            else:
                # If adding this paragraph exceeds max chunk size, push current chunk
                chunks.append({"text": current_chunk, "metadata": metadata})
                current_chunk = paragraph  # Start new chunk

        # Append any remaining chunk
        if current_chunk:
            chunks.append({"text": current_chunk, "metadata": metadata})

        return chunks
