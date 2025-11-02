from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.core.config.config_loader import ConfigLoader
from src.core.embedding.embedder_factory import EmbedderFactory
from src.core.embedding.vector_store_factory import VectorStoreFactory


logger = logging.getLogger("EmbeddingOrchestrator")


def _load_json(path: Path) -> Dict[str, Any]:
    # Load JSON file from disk
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_chunk_files(chunks_dir: Path):
    # Yield all json files in chunks directory
    for p in chunks_dir.glob("*.json"):
        if p.is_file():
            yield p


def _extract_chunks(chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support both {"chunks": [...]} and {"text": "..."} formats
    if "chunks" in chunk_data and isinstance(chunk_data["chunks"], list):
        return chunk_data["chunks"]
    elif "text" in chunk_data:
        return [{"text": chunk_data["text"]}]
    else:
        return []


def main() -> None:
    # Load config
    cfg = ConfigLoader("configs/embedding.yaml").config

    # Logging setup
    opts: Dict[str, Any] = cfg.get("options", {})
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger.info("Starting embedding pipeline")

    # Paths
    paths: Dict[str, Any] = cfg.get("paths", {})
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks")).resolve()
    metadata_dir = Path(paths.get("metadata_dir", "data/processed/metadata")).resolve()

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")
    if not metadata_dir.exists():
        logger.warning(f"Metadata directory does not exist: {metadata_dir} (metadata will be empty)")

    # Build embedder
    embedder = EmbedderFactory.from_config(cfg)
    logger.info(f"Initialized embedder with dimension={embedder.dimension}")

    # Build vector store
    vector_store = VectorStoreFactory.from_config(cfg, dimension=embedder.dimension)
    logger.info("Initialized vector store")

    batch_size = int(opts.get("batch_size", 16))

    texts_batch: List[str] = []
    metas_batch: List[Dict[str, Any]] = []

    try:
        for chunk_file in _iter_chunk_files(chunks_dir):
            chunk_json = _load_json(chunk_file)
            chunks = _extract_chunks(chunk_json)
            if not chunks:
                logger.warning(f"No chunks found in {chunk_file.name}")
                continue

            # Find corresponding metadata
            meta_path = metadata_dir / chunk_file.name
            meta_data: Dict[str, Any] = {}
            if meta_path.exists():
                meta_data = _load_json(meta_path)
            else:
                logger.warning(f"No metadata found for {chunk_file.name}")

            for ch in chunks:
                text = ch.get("text", "").strip()
                if not text:
                    continue

                # Build merged metadata per chunk
                merged_meta: Dict[str, Any] = {
                    "source_file": meta_data.get("source_file", chunk_file.stem),
                    "title": meta_data.get("title"),
                    "authors": meta_data.get("authors"),
                    "year": meta_data.get("year"),
                    "detected_language": meta_data.get("detected_language"),
                    "page_count": meta_data.get("page_count"),
                    # Optionally: add chunk index or file name
                    "origin_chunk_file": str(chunk_file.name),
                }

                texts_batch.append(text)
                metas_batch.append(merged_meta)

                if len(texts_batch) >= batch_size:
                    # Embed current batch
                    try:
                        vectors = embedder.embed_batch(texts_batch, batch_size=batch_size)
                        vector_store.add_vectors(vectors, texts_batch, metas_batch)
                        logger.info(f"Embedded and stored batch of size {len(texts_batch)}")
                    except Exception as e:
                        logger.error(f"Error during embedding or storing batch: {e}")
                    finally:
                        texts_batch = []
                        metas_batch = []

        # Process remaining
        if texts_batch:
            try:
                vectors = embedder.embed_batch(texts_batch, batch_size=batch_size)
                vector_store.add_vectors(vectors, texts_batch, metas_batch)
                logger.info(f"Embedded and stored final batch of size {len(texts_batch)}")
            except Exception as e:
                logger.error(f"Error during final embedding or storing batch: {e}")

        # Persist everything
        vector_store.persist()
        logger.info("Embedding pipeline finished successfully.")

    finally:
        # Ensure resources are closed
        embedder.close()
        vector_store.close()


if __name__ == "__main__":
    main()
