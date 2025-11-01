from __future__ import annotations
import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from src.core.ingestion.parser.parser_factory import ParserFactory


class ParallelPdfParser:
    """CPU-based parallel parser orchestrator using multiple processes."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.parser_factory = ParserFactory(config, logger=self.logger)

        # Konfiguriere die maximale Anzahl an Workern, basierend auf der Anzahl der CPU-Kerne
        parallel_cfg = config.get("options", {}).get("parallelism", "auto")
        if isinstance(parallel_cfg, int) and parallel_cfg > 0:
            self.num_workers = parallel_cfg
        else:
            # Maximal 4 Worker verwenden, wenn die CPU nicht so viele Kerne hat
            self.num_workers = min(max(1, multiprocessing.cpu_count() - 1), 4)

        self.logger.info(f"Initialized ParallelPdfParser with {self.num_workers} worker(s)")

    # ------------------------------------------------------------------
    def parse_all(self, pdf_dir: str | Path, output_dir: str | Path) -> List[Dict[str, Any]]:
        """
        Parse all PDFs in the given directory and output the parsed results to the output directory.
        
        :param pdf_dir: Directory containing the PDF files.
        :param output_dir: Directory where the parsed results should be saved.
        :return: List of dictionaries containing parsed content from the PDFs.
        """
        pdf_dir, output_dir = Path(pdf_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDFs found in {pdf_dir}")
            return []

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_pdf = {executor.submit(self._parse_single, str(pdf)): pdf for pdf in pdf_files}
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    # Set timeout for each future (e.g., 300 seconds = 5 minutes)
                    result = future.result(timeout=300)  # Timeout set to 5 minutes
                    results.append(result)
                    out_path = output_dir / f"{pdf.stem}.parsed.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Parsed {pdf.name}")
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"Timeout occurred while parsing {pdf.name}")
                except Exception as e:
                    self.logger.error(f"Failed to parse {pdf.name}: {e}")

        return results

    # ------------------------------------------------------------------
    def _parse_single(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a single PDF file using the appropriate parser and return the extracted data.
        
        :param pdf_path: Path to the PDF file to parse.
        :return: A dictionary containing parsed content from the PDF.
        """
        parser = self.parser_factory.create_parser()
        try:
            result = parser.parse(pdf_path)
            return result
        except Exception as e:
            self.logger.error(f"Error parsing {pdf_path}: {e}")
            return {"text": "", "metadata": {"source_file": pdf_path}}

