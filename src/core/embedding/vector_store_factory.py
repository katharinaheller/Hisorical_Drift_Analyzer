from __future__ import annotations
from typing import Any, Dict, List
import logging
import os
import json
import sqlite3

from src.core.embedding.interfaces.i_vector_store import IVectorStore


class FAISSVectorStore(IVectorStore):
    """FAISS-based local vector store."""

    def __init__(self, persist_dir: str, dimension: int):
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError("faiss-cpu is required for FAISSVectorStore. "
                              "Install via: poetry add faiss-cpu") from e

        self.faiss = faiss
        self.dimension = dimension
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.meta_path = os.path.join(persist_dir, "metadata.jsonl")

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        # Metadata is stored separately as JSONL
        self._meta_fh = open(self.meta_path, "a", encoding="utf-8")

    def add_vectors(self,
                    vectors: List[List[float]],
                    documents: List[str],
                    metadatas: List[Dict[str, Any]] | None = None) -> None:
        import numpy as np  # local import to avoid hard dependency at class load

        arr = np.array(vectors).astype("float32")
        self.index.add(arr)

        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            payload = {
                "text": doc,
                "metadata": meta,
            }
            self._meta_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def persist(self) -> None:
        self.faiss.write_index(self.index, self.index_path)
        self._meta_fh.flush()

    def close(self) -> None:
        try:
            self.persist()
        finally:
            self._meta_fh.close()


class LanceDBVectorStore(IVectorStore):
    """LanceDB-based vector store (local file-based)."""

    def __init__(self, persist_dir: str, dimension: int):
        try:
            import lancedb  # type: ignore
        except ImportError as e:
            raise ImportError("lancedb is required for LanceDBVectorStore. "
                              "Install via: poetry add lancedb") from e

        self.dimension = dimension
        os.makedirs(persist_dir, exist_ok=True)
        self.db = lancedb.connect(persist_dir)
        self.table = self.db.open_table("embeddings") if "embeddings" in self.db.table_names() else \
            self.db.create_table("embeddings", data=[
                {
                    "vector": [0.0] * dimension,
                    "text": "",
                    "metadata": {},
                }
            ])

    def add_vectors(self,
                    vectors: List[List[float]],
                    documents: List[str],
                    metadatas: List[Dict[str, Any]] | None = None) -> None:
        rows = []
        for i, vec in enumerate(vectors):
            rows.append({
                "vector": vec,
                "text": documents[i],
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
            })
        self.table.add(rows)

    def persist(self) -> None:
        # LanceDB persists automatically
        pass

    def close(self) -> None:
        # Nothing to close
        pass


class SQLiteVectorStore(IVectorStore):
    """Very simple SQLite-based vector store (for debugging / small-scale)."""

    def __init__(self, persist_dir: str, dimension: int):
        os.makedirs(persist_dir, exist_ok=True)
        db_path = os.path.join(persist_dir, "vectors.sqlite3")
        self.conn = sqlite3.connect(db_path)
        self.dimension = dimension
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                vector BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_vectors(self,
                    vectors: List[List[float]],
                    documents: List[str],
                    metadatas: List[Dict[str, Any]] | None = None) -> None:
        import numpy as np  # local import
        cur = self.conn.cursor()
        for i, vec in enumerate(vectors):
            meta_str = json.dumps(metadatas[i], ensure_ascii=False) if metadatas and i < len(metadatas) else "{}"
            arr = np.array(vec, dtype="float32").tobytes()
            cur.execute(
                "INSERT INTO embeddings (text, metadata, vector) VALUES (?, ?, ?)",
                (documents[i], meta_str, arr)
            )
        self.conn.commit()

    def persist(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class VectorStoreFactory:
    """Factory for creating IVectorStore instances from config."""

    @staticmethod
    def from_config(cfg: Dict[str, Any], dimension: int) -> IVectorStore:
        opts: Dict[str, Any] = cfg.get("options", {})
        store_name = opts.get("vector_store", "FAISS").upper()

        paths: Dict[str, Any] = cfg.get("paths", {})
        persist_dir = paths.get("vector_store_dir", "data/vector_store")

        if store_name == "FAISS":
            return FAISSVectorStore(persist_dir=persist_dir, dimension=dimension)
        elif store_name == "LANCEDB":
            return LanceDBVectorStore(persist_dir=persist_dir, dimension=dimension)
        elif store_name == "SQLITE":
            return SQLiteVectorStore(persist_dir=persist_dir, dimension=dimension)
        else:
            raise ValueError(f"Unsupported vector store: {store_name}")
