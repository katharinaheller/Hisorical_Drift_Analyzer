from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Any, Dict

from src.core.config.config_loader import ConfigLoader
from src.core.ingestion.utils.file_utils import ensure_dir
from src.core.ingestion.parser.parser_factory import ParserFactory
from src.core.ingestion.metadata.metadata_extractor_factory import MetadataExtractorFactory
from src.core.ingestion.cleaner.rag_text_cleaner import RagTextCleaner
from src.core.ingestion.chunking.chunking_orchestrator import ChunkingOrchestrator


def main():
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = ConfigLoader("configs/ingestion.yaml")  # Load YAML configuration

    opts: Dict[str, Any] = cfg.config.get("options", {})  # Access options
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)

    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("IngestionOrchestrator")
    logger.info("Starting ingestion pipeline")

    # ------------------------------------------------------------------
    # 2. Initialize components
    # ------------------------------------------------------------------
    factory = ParserFactory(cfg.config, logger=logger)  # Pass full configuration to ParserFactory
    metadata_factory = MetadataExtractorFactory.from_config(cfg.config)
    cleaner = RagTextCleaner.default()  # Deterministic multi-stage cleaner

    # Initialize the ChunkingOrchestrator with YAML config
    chunking_orchestrator = ChunkingOrchestrator(config=cfg.config)  # Pass configuration to ChunkingOrchestrator

    active_metadata_fields = opts.get("metadata_fields", [])
    parallelism = opts.get("parallelism", "auto")

    raw_dir = Path(cfg.config["paths"]["raw_pdfs"]).resolve()  # Resolve paths from configuration
    parsed_dir = Path(cfg.config["paths"]["parsed"]).resolve()
    cleaned_dir = Path(cfg.config["paths"].get("cleaned", "data/processed/cleaned")).resolve()
    metadata_dir = Path(cfg.config["paths"]["metadata"]).resolve()
    chunks_dir = Path(cfg.config["paths"].get("chunks", "data/processed/chunks")).resolve()  # Define new chunks directory
    ensure_dir(parsed_dir)
    ensure_dir(cleaned_dir)
    ensure_dir(metadata_dir)
    ensure_dir(chunks_dir)  # Ensure the chunks directory exists

    pdf_files = sorted(raw_dir.glob("*.pdf"))  # Look for PDFs in the directory
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) in {raw_dir}")

    # ------------------------------------------------------------------
    # 3. Parallel or sequential mode
    # ------------------------------------------------------------------
    use_parallel = (
        isinstance(parallelism, int) and parallelism > 1
    ) or (isinstance(parallelism, str) and parallelism.lower() == "auto")

    if use_parallel:
        try:
            parallel_parser = factory.create_parallel_parser()
            logger.info("Running ingestion in parallel mode ...")
            results = parallel_parser.parse_all(raw_dir, parsed_dir)

            for res in results:
                # --- STEP 1: Clean text deterministically ---
                if "text" in res:
                    res["text"] = cleaner.clean(res["text"])

                # --- STEP 2: Chunking ---
                if "text" in res:
                    chunked_data = chunking_orchestrator.process(res["text"], metadata=res["metadata"])
                    res["chunks"] = chunked_data  # Add chunks to the result

                # --- STEP 3: Extract and filter metadata ---
                pdf_name = Path(res["metadata"]["source_file"]).stem
                pdf_path = raw_dir / f"{pdf_name}.pdf"
                all_meta = metadata_factory.extract_all(str(pdf_path))
                filtered_meta = {
                    k: v for k, v in all_meta.items()
                    if not active_metadata_fields or k in active_metadata_fields
                }
                res["metadata"].update(filtered_meta)

                # --- STEP 4: Save outputs ---
                # Save raw parsed output (only text)
                parsed_path = parsed_dir / f"{pdf_name}.parsed.json"
                with open(parsed_path, "w", encoding="utf-8") as f:
                    json.dump({"text": res["text"]}, f, ensure_ascii=False, indent=2)

                # Save cleaned text only as JSON (without hash and length)
                cleaned_path = cleaned_dir / f"{pdf_name}.cleaned.json"
                if "text" in res:
                    cleaned_data = {
                        "cleaned_text": res["text"],
                    }
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

                # Save metadata separately
                meta_path = metadata_dir / f"{pdf_name}.metadata.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(res["metadata"], f, ensure_ascii=False, indent=2)

                # Save chunked data in the 'chunks' directory as JSON
                chunked_path = chunks_dir / f"{pdf_name}.chunks.json"  # Updated to save in the 'chunks' directory
                with open(chunked_path, "w", encoding="utf-8") as f:
                    json.dump({"chunks": res["chunks"]}, f, ensure_ascii=False, indent=2)

            logger.info(f"Parallel ingestion completed ({len(results)} file(s)).")
            return

        except Exception as e:
            logger.error(f"Parallel ingestion failed → falling back to sequential: {e}")

    # ------------------------------------------------------------------
    # 4. Sequential fallback mode
    # ------------------------------------------------------------------
    parser = factory.create_parser()  # Use the appropriate parser based on configuration
    for pdf in pdf_files:
        logger.info(f"Parsing {pdf.name} ...")
        try:
            # ---- STEP 1: Parse text ----
            parsed_result = parser.parse(str(pdf))

            # ---- STEP 2: Clean text ----
            if "text" in parsed_result:
                parsed_result["text"] = cleaner.clean(parsed_result["text"])

            # ---- STEP 3: Chunking ----
            if "text" in parsed_result:
                chunked_data = chunking_orchestrator.process(parsed_result["text"], metadata=parsed_result.get("metadata", {}))
                parsed_result["chunks"] = chunked_data  # Add chunks to the result

            # ---- STEP 4: Extract metadata (metadata module handles it fully) ----
            base_metadata = parsed_result.get("metadata", {})
            all_metadata = metadata_factory.extract_all(str(pdf))
            if active_metadata_fields:
                all_metadata = {
                    k: v for k, v in all_metadata.items() if k in active_metadata_fields
                }
            base_metadata.update(all_metadata)
            parsed_result["metadata"] = base_metadata

            # ---- STEP 5: Save outputs ----
            # Save raw parsed output (only text)
            parsed_path = parsed_dir / f"{pdf.stem}.parsed.json"
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump({"text": parsed_result["text"]}, f, ensure_ascii=False, indent=2)

            # Save cleaned text as JSON (without hash and length)
            cleaned_path = cleaned_dir / f"{pdf.stem}.cleaned.json"
            cleaned_data = {
                "cleaned_text": parsed_result["text"],
            }
            with open(cleaned_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

            # Save metadata separately
            meta_path = metadata_dir / f"{pdf.stem}.metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(base_metadata, f, ensure_ascii=False, indent=2)

            # Save chunked data in the 'chunks' directory as JSON
            chunked_path = chunks_dir / f"{pdf.stem}.chunks.json"  # Updated to save in the 'chunks' directory
            with open(chunked_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": parsed_result["chunks"]}, f, ensure_ascii=False, indent=2)

            logger.info(f"Completed {pdf.name}")

        except Exception as e:
            logger.error(f"Failed to parse {pdf.name}: {e}")

    # ------------------------------------------------------------------
    # 5. Finish
    # ------------------------------------------------------------------
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
