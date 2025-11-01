from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import multiprocessing

from src.core.ingestion.parser.interfaces.i_pdf_parser import IPdfParser
from src.core.ingestion.parser.pymupdf_parser import PyMuPDFParser
from src.core.ingestion.parser.pdfplumber_parser import PdfPlumberParser  # Import PdfPlumberParser

class ParserFactory:
    """
    Factory for creating local PDF parsers and optional parallel orchestrators.
    Handles YAML parameters like 'parallelism' and parser configuration.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        opts = config.get("options", {})

        self.parser_mode = opts.get("pdf_parser", "auto").lower()  # Get parser mode (fitz, pdfplumber, auto)
        self.parallelism = opts.get("parallelism", "auto")  # Get parallelism setting
        self.language = opts.get("language", "auto")  # Get language setting
        self.exclude_toc = True  # Enforce exclusion of table of contents globally
        self.max_pages = opts.get("max_pages", None)  # Max pages to parse

        # Determine optimal CPU usage based on parallelism
        if isinstance(self.parallelism, int) and self.parallelism > 0:
            self.num_workers = self.parallelism
        else:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)

        self.logger.info(
            f"ParserFactory initialized | mode={self.parser_mode}, "
            f"workers={self.num_workers}, exclude_toc={self.exclude_toc}"
        )

    def create_parser(self) -> IPdfParser:
        """Return configured parser instance based on configuration (fitz, pdfplumber, etc.)."""
        if self.parser_mode == "fitz":
            return PyMuPDFParser(exclude_toc=self.exclude_toc, max_pages=self.max_pages)  # Use PyMuPDFParser
        elif self.parser_mode == "pdfplumber":
            return PdfPlumberParser(exclude_toc=self.exclude_toc, max_pages=self.max_pages)  # Use PdfPlumberParser
        elif self.parser_mode == "auto":
            # Automatically choose the best parser
            try:
                return PdfPlumberParser(exclude_toc=self.exclude_toc, max_pages=self.max_pages)
            except Exception:
                return PyMuPDFParser(exclude_toc=self.exclude_toc, max_pages=self.max_pages)  # Fallback to PyMuPDFParser
        else:
            raise ValueError(f"Unsupported parser mode: {self.parser_mode}")  # Error if parser mode is unsupported

    def create_parallel_parser(self):
        """Return a parallel orchestrator that distributes parsing tasks."""
        from src.core.ingestion.parser.parallel_pdf_parser import ParallelPdfParser
        return ParallelPdfParser(self.config, logger=self.logger)  # Return parallel parser for distribution
