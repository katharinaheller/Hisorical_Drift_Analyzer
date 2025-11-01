"""from __future__ import annotations
from typing import Optional
from pathlib import Path
from lxml import etree
import re
import fitz


class AbstractExtractor:
    """Extracts abstract text directly from GROBID TEI XML or PDF XMP metadata."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> Optional[str]:
        pdf_file = Path(pdf_path)

        # 1. Try GROBID XML
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            abstract = self._extract_from_grobid(xml_path)
            if abstract:
                return abstract

        # 2. Try PDF metadata (some PDFs include "subject"/"keywords"/"description")
        abstract = self._extract_from_pdf_metadata(pdf_file)
        if abstract:
            return abstract

        # 3. Fallback: None (no text heuristics)
        return None

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> Optional[str]:
        """Read abstract-like information from PDF metadata fields."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                for key in ("subject", "Subject", "description", "Description", "keywords", "Keywords"):
                    val = meta.get(key)
                    if isinstance(val, str) and len(val.strip()) > 20:
                        return re.sub(r"\s+", " ", val.strip())
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Path | None:
        xml_candidate = pdf_file.with_suffix(".tei.xml")
        if xml_candidate.exists():
            return xml_candidate
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    # ------------------------------------------------------------------
    def _extract_from_grobid(self, xml_path: Path) -> Optional[str]:
        """Parse TEI XML to extract the abstract section."""
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml = f.read()
            root = etree.fromstring(xml.encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            abs_text = root.xpath("string(//tei:abstract)", namespaces=ns)
            if abs_text and len(abs_text.strip()) > 10:
                return re.sub(r"\s+", " ", abs_text.strip())
        except Exception:
            return None
        return None"""