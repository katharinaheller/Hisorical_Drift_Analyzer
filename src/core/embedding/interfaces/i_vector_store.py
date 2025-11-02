from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IVectorStore(ABC):
    """Interface for all vector store backends."""

    @abstractmethod
    def add_vectors(self,
                    vectors: List[List[float]],
                    documents: List[str],
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        # Add vectors and associated payloads to the store
        pass

    @abstractmethod
    def persist(self) -> None:
        # Persist the store to disk
        pass

    @abstractmethod
    def close(self) -> None:
        # Close resources if necessary
        pass
