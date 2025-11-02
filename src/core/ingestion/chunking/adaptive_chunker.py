from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.core.ingestion.chunking.i_chunker import IChunker
from src.core.ingestion.metadata.interfaces.i_page_number_extractor import IPageNumberExtractor
import spacy
import logging

class AdaptiveChunker(IChunker):
    """Chunker that adapts to the content structure by splitting at semantic breaks."""

    def __init__(self,
                 chunk_size: int = 500,
                 overlap: int = 200,
                 min_chunk_length: int = 400,
                 pdf_path: Optional[str] = None,
                 page_number_extractor: Optional[IPageNumberExtractor] = None,
                 text_length: Optional[int] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
        self.pdf_path = pdf_path
        self.page_number_extractor = page_number_extractor
        self.text_length = text_length

        # Initialize spaCy NLP model for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

        # Extract page count if available
        if pdf_path and page_number_extractor:
            try:
                self.page_count = self.page_number_extractor.extract_page_number(pdf_path)
                self.adjust_chunking_based_on_page_count()
            except Exception as e:
                logging.warning(f"Error extracting page count from PDF: {e}")
                self.page_count = None
                self.chunk_size = 1000  # Fallback

        if text_length:
            self.adjust_chunking_based_on_text_length(text_length)

    def adjust_chunking_based_on_page_count(self):
        """Adjust chunk size and overlap based on the page count."""
        if self.page_count:
            if self.page_count > 50:
                self.chunk_size = 1000
                self.overlap = 700
            elif self.page_count > 30:
                self.chunk_size = 1500
                self.overlap = 500
            elif self.page_count > 10:
                self.chunk_size = 2000
                self.overlap = 300
            else:
                self.chunk_size = 2500
                self.overlap = 200
        else:
            self.chunk_size = 1000
            self.overlap = 200

    def adjust_chunking_based_on_text_length(self, text_length: int):
        """Adjust chunk size and overlap based on the length of the text."""
        if text_length:
            if text_length > 5000:
                self.chunk_size = 1000
                self.overlap = 600
            elif text_length > 2000:
                self.chunk_size = 1500
                self.overlap = 400
            else:
                self.chunk_size = 2000
                self.overlap = 200

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk the text into adaptive chunks based on content size and overlap."""
        chunks = []
        current_chunk = ""

        doc = self.nlp(text)  # Use spaCy to process the text
        sentences = [sent.text.strip() for sent in doc.sents]  # Extract sentences from the spaCy doc

        for sentence in sentences:
            # If the current chunk plus sentence exceeds chunk size, store the chunk and start a new one
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:  # Prevent empty chunks
                    chunks.append({
                        "text": current_chunk.strip(),
                        # "chunk_size": len(current_chunk.strip()),  # Removed from output
                        # "overlap": self.overlap  # Removed from output
                    })
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

            # Merge chunks if they are too small
            if len(current_chunk) < self.min_chunk_length and chunks:
                last_chunk = chunks[-1]
                last_chunk["text"] += " " + current_chunk
                current_chunk = ""

        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                # "chunk_size": len(current_chunk.strip()),  # Removed from output
                # "overlap": self.overlap  # Removed from output
            })

        return chunks
