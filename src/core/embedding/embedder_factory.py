from __future__ import annotations
from typing import Any, Dict
import logging

from src.core.embedding.interfaces.i_embedder import IEmbedder


class SentenceTransformerEmbedder(IEmbedder):
    """Local embedder using sentence-transformers."""

    def __init__(self, model_name: str, dimension: int | None = None, normalize_embeddings: bool = True):
        # Lazy import to avoid hard dependency
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("sentence-transformers is required for SentenceTransformerEmbedder. "
                              "Install via: poetry add sentence-transformers") from e

        self._model = SentenceTransformer(model_name)
        self._normalize = normalize_embeddings
        # If dimension not given, infer from model
        if dimension is None:
            test_vec = self._model.encode("test", normalize_embeddings=self._normalize)
            self._dimension = len(test_vec)
        else:
            self._dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        # Encode single text
        return self._model.encode(text, normalize_embeddings=self._normalize).tolist()

    def embed_batch(self, texts, batch_size=None) -> list[list[float]]:
        # Encode multiple texts
        return self._model.encode(
            list(texts),
            batch_size=batch_size or 32,
            normalize_embeddings=self._normalize
        ).tolist()

    @property
    def dimension(self) -> int:
        # Return embedding dimension
        return self._dimension

    def close(self) -> None:
        # Nothing to close for sentence-transformers
        pass


class EmbedderFactory:
    """Factory for creating IEmbedder instances from config."""

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> IEmbedder:
        opts: Dict[str, Any] = cfg.get("options", {})
        model_name = opts.get("embedding_model", "all-MiniLM-L6-v2")
        normalize = bool(opts.get("normalize_embeddings", True))
        dimension = opts.get("dimension", None)

        backend = opts.get("embedding_backend", "sentence-transformers").lower()

        if backend == "sentence-transformers":
            return SentenceTransformerEmbedder(
                model_name=model_name,
                dimension=dimension,
                normalize_embeddings=normalize,
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")
