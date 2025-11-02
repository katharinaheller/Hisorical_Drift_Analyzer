from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.core.ingestion.chunking.i_chunker import IChunker
import spacy

class StaticChunker(IChunker):
    """Chunker that uses fixed chunk size and overlap for chunking."""
    
    def __init__(self, 
                 chunk_size: int,  # Configuration-based chunk size
                 overlap: int, 
                 min_chunk_length: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

        # Initialize spaCy NLP model for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk = ""

        doc = self.nlp(text)  # Use spaCy to process the text
        sentences = [sent.text.strip() for sent in doc.sents]  # Extract sentences from the spaCy doc

        for sentence in sentences:
            # Add sentence if it fits into the current chunk
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += ". " + sentence
            else:
                chunks.append({
                    "text": current_chunk.strip(),
                    # "chunk_size": len(current_chunk.strip()),  # Removed from output
                    # "overlap": self.overlap  # Removed from output
                })
                current_chunk = sentence  # Start a new chunk

            # Merge small chunks if necessary
            if len(current_chunk) < self.min_chunk_length and chunks:
                last_chunk = chunks[-1]
                last_chunk["text"] += " " + current_chunk
                current_chunk = ""

        # Append any remaining text as the final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                # "chunk_size": len(current_chunk.strip()),  # Removed from output
                # "overlap": self.overlap  # Removed from output
            })

        return chunks
