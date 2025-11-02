from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Sequence, Optional


class IEmbedder(ABC):
    """Interface for all embedding backends."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        # Return embedding vector for a single text
        pass

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], batch_size: Optional[int] = None) -> List[List[float]]:
        # Return embedding vectors for a batch of texts
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        # Return embedding dimension
        pass

    @abstractmethod
    def close(self) -> None:
        # Release resources if necessary
        pass
