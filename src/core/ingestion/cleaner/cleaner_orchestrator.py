from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict

from src.core.ingestion.config_loader import ConfigLoader
from src.core.ingestion.utils.file_utils import ensure_dir
from src.core.ingestion.cleaner.rag_text_cleaner import RagTextCleaner


def deterministic_hash(text: str) -> str:
    # Compute deterministic SHA1 fingerprint of cleaned text
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = ConfigLoader("configs/ingestion.yaml").config
    opts: Dict[str, Any] = cfg.get("options", {})
    log_level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)

    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("CleanerOrchestrator")
    logger.info("Starting cleaning phase")

    # ------------------------------------------------------------------
    # 2. Resolve paths
    # ------------------------------------------------------------------
    parsed_dir = Path(cfg["paths"]["parsed"]).resolve()
    cleaned_dir = Path(cfg["paths"].get("cleaned", "data/processed/cleaned")).resolve()
    ensure_dir(cleaned_dir)

    parsed_files = sorted(parsed_dir.glob("*.parsed.json"))
    if not parsed_files:
        logger.warning(f"No parsed files found in {parsed_dir}")
        return

    logger.info(f"Found {len(parsed_files)} parsed file(s) for cleaning")

    # ------------------------------------------------------------------
    # 3. Initialize deterministic multi-stage cleaner
    # ------------------------------------------------------------------
    cleaner = RagTextCleaner.default()

    # ------------------------------------------------------------------
    # 4. Iterate over parsed JSONs
    # ------------------------------------------------------------------
    for idx, parsed_path in enumerate(parsed_files, start=1):
        try:
            with open(parsed_path, "r", encoding="utf-8") as f:
                parsed_data = json.load(f)

            raw_text = parsed_data.get("text", "")
            if not raw_text:
                logger.warning(f"Skipping {parsed_path.name}: no text field")
                continue

            # --- Step 1: Clean text ---
            cleaned_text = cleaner.clean(raw_text)

            # --- Step 2: Compute deterministic hash for auditability ---
            text_hash = deterministic_hash(cleaned_text)

            # --- Step 3: Write cleaned output ---
            cleaned_filename = parsed_path.stem.replace(".parsed", "") + ".cleaned.txt"
            cleaned_path = cleaned_dir / cleaned_filename
            with open(cleaned_path, "w", encoding="utf-8") as cf:
                cf.write(cleaned_text)

            # --- Step 4: Write minimal metadata file for reproducibility ---
            meta_info = {
                "source_parsed": str(parsed_path.name),
                "cleaned_file": str(cleaned_path.name),
                "sha1": text_hash,
                "length_chars": len(cleaned_text),
            }
            meta_path = cleaned_dir / (cleaned_filename + ".meta.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta_info, mf, ensure_ascii=False, indent=2)

            logger.info(f"✓ Cleaned {parsed_path.name} ({idx}/{len(parsed_files)})")

        except Exception as e:
            logger.error(f"✗ Failed to clean {parsed_path.name}: {e}")

    # ------------------------------------------------------------------
    # 5. Finish
    # ------------------------------------------------------------------
    logger.info("Cleaning phase completed successfully.")


if __name__ == "__main__":
    main()
