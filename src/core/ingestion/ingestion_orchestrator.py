from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Any, Dict

from src.core.ingestion.config_loader import ConfigLoader
from src.core.ingestion.utils.file_utils import ensure_dir
from src.core.ingestion.parser.parser_factory import ParserFactory
from src.core.ingestion.metadata.metadata_extractor_factory import MetadataExtractorFactory
from src.core.ingestion.cleaner.rag_text_cleaner import RagTextCleaner


def main():
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = ConfigLoader("configs/ingestion.yaml").config
    opts: Dict[str, Any] = cfg.get("options", {})
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)

    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("IngestionOrchestrator")
    logger.info("Starting ingestion pipeline")

    # ------------------------------------------------------------------
    # 2. Initialize components
    # ------------------------------------------------------------------
    factory = ParserFactory(cfg, logger=logger)
    metadata_factory = MetadataExtractorFactory.from_config(cfg)
    cleaner = RagTextCleaner.default()  # deterministic multi-stage cleaner

    active_metadata_fields = opts.get("metadata_fields", [])
    parallelism = opts.get("parallelism", "auto")

    raw_dir = Path(cfg["paths"]["raw_pdfs"]).resolve()
    parsed_dir = Path(cfg["paths"]["parsed"]).resolve()
    cleaned_dir = Path(cfg["paths"].get("cleaned", "data/processed/cleaned")).resolve()
    metadata_dir = Path(cfg["paths"]["metadata"]).resolve()
    ensure_dir(parsed_dir)
    ensure_dir(cleaned_dir)
    ensure_dir(metadata_dir)

    pdf_files = sorted(raw_dir.glob("*.pdf"))
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

                # --- STEP 2: Extract and filter metadata ---
                pdf_name = Path(res["metadata"]["source_file"]).stem
                pdf_path = raw_dir / f"{pdf_name}.pdf"
                all_meta = metadata_factory.extract_all(str(pdf_path))
                filtered_meta = {
                    k: v for k, v in all_meta.items()
                    if not active_metadata_fields or k in active_metadata_fields
                }
                res["metadata"].update(filtered_meta)

                # --- STEP 3: Save outputs ---
                # Save raw parsed output (only text)
                parsed_path = parsed_dir / f"{pdf_name}.parsed.json"
                with open(parsed_path, "w", encoding="utf-8") as f:
                    json.dump({"text": res["text"]}, f, ensure_ascii=False, indent=2)

                # Save cleaned text only
                cleaned_path = cleaned_dir / f"{pdf_name}.cleaned.txt"
                if "text" in res:
                    with open(cleaned_path, "w", encoding="utf-8") as f:
                        f.write(res["text"])

                # Save metadata separately
                meta_path = metadata_dir / f"{pdf_name}.metadata.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(res["metadata"], f, ensure_ascii=False, indent=2)

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

            # ---- STEP 3: Extract metadata (metadata module handles it fully) ----
            base_metadata = parsed_result.get("metadata", {})
            all_metadata = metadata_factory.extract_all(str(pdf))
            if active_metadata_fields:
                all_metadata = {
                    k: v for k, v in all_metadata.items() if k in active_metadata_fields
                }
            base_metadata.update(all_metadata)
            parsed_result["metadata"] = base_metadata

            # ---- STEP 4: Save outputs ----
            # Save raw parsed output (only text)
            parsed_path = parsed_dir / f"{pdf.stem}.parsed.json"
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump({"text": parsed_result["text"]}, f, ensure_ascii=False, indent=2)

            # Save cleaned text only
            cleaned_path = cleaned_dir / f"{pdf.stem}.cleaned.txt"
            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write(parsed_result["text"])

            # Save metadata separately
            meta_path = metadata_dir / f"{pdf.stem}.metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(base_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Completed {pdf.name}")

        except Exception as e:
            logger.error(f"Failed to parse {pdf.name}: {e}")

    # ------------------------------------------------------------------
    # 5. Finish
    # ------------------------------------------------------------------
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
