from __future__ import annotations
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict
from src.core.ingestion.chunking.i_chunker import IChunker
from src.core.ingestion.chunking.static_chunker import StaticChunker
from src.core.ingestion.chunking.adaptive_chunker import AdaptiveChunker
from src.core.ingestion.utils.file_utils import get_file_size, get_pdf_page_count


class ChunkingOrchestrator:
    """Handles the selection and execution of chunking strategies."""

    def __init__(self, config: Dict[str, Any], strategy: str = "adaptive"):
        self.strategy = strategy
        self.config = config
        self.chunker = self.select_chunker()

    def select_chunker(self) -> IChunker:
        """Dynamically select chunking strategy based on config."""

        chunking_mode = self.config["chunking"]["mode"]
        chunk_size = self.config["chunking"]["chunk_size"]
        overlap = self.config["chunking"]["overlap"]

        # Decide if chunk size and overlap should be automatic
        if chunking_mode == "adaptive":
            return AdaptiveChunker(chunk_size=chunk_size, overlap=overlap)
        elif chunking_mode == "static":
            return StaticChunker(chunk_size=chunk_size, overlap=overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_mode}")

    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process the text and return chunks with metadata."""
        return self.chunker.chunk(text, metadata)

    def determine_auto_chunking(self, document: Path) -> Dict[str, Any]:
        """Determine the optimal chunking settings based on document properties (size, page count)."""
        
        # Check file size and page count for larger documents
        file_size = get_file_size(document)
        page_count = get_pdf_page_count(document)

        chunk_size = 1000  # Default chunk size
        overlap = 200  # Default overlap size
        
        # Logic to adjust chunking for large documents
        if file_size > 10 * 1024 * 1024:  # larger than 10 MB
            chunk_size = 5000
            overlap = 500
        elif page_count > 20:  # large PDF with many pages
            chunk_size = 3000
            overlap = 400
        else:  # smaller documents
            chunk_size = 1000
            overlap = 100
        
        return {"chunk_size": chunk_size, "overlap": overlap}
