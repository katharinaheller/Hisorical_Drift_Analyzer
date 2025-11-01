# ==============================================================
# src/core/ingestion/metadata/implementations/title_extractor.py
# ==============================================================

from __future__ import annotations
from pathlib import Path
from typing import Optional
import fitz
import re
from lxml import etree


class TitleExtractor:
    """
    Robust, deterministic extractor for scientific paper titles.
    Filename fallback *is not used* (explizit ausgeschlossen).
    """

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def extract(self, pdf_path: str) -> Optional[str]:
        """
        Extract the most probable title of a given PDF.
        Priority:
          1. PDF metadata
          2. GROBID TEI XML
          3. Layout + heuristic on first page
        Returns None if keine verlässliche Titel gefunden wird.
        """
        pdf_file = Path(pdf_path)

        # 1. PDF metadata
        title = self._extract_from_pdf_metadata(pdf_file)
        if self._is_valid(title):
            return title

        # 2. GROBID TEI XML
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            grobid_title = self._extract_from_grobid(xml_path)
            if self._is_valid(grobid_title):
                return grobid_title

        # 3. Layout + heuristic on first page
        title_from_layout = self._extract_from_first_page_layout(pdf_file)
        if self._is_valid(title_from_layout):
            return title_from_layout

        # Keine Datei-Name Fallback!
        return None

    def _extract_from_pdf_metadata(self, pdf_file: Path) -> Optional[str]:
        """Extract title from embedded PDF metadata fields."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                title_meta = meta.get("title") or meta.get("Title")
                if title_meta:
                    cleaned = self._normalize(title_meta)
                    if self._is_valid(cleaned):
                        return cleaned
        except Exception:
            return None
        return None

    def _find_grobid_xml(self, pdf_file: Path) -> Optional[Path]:
        """Locate associated GROBID TEI XML file (same stem .tei.xml)."""
        candidate1 = pdf_file.with_suffix(".tei.xml")
        if candidate1.exists():
            return candidate1
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    def _extract_from_grobid(self, xml_path: Path) -> Optional[str]:
        """Extract title from structured GROBID TEI XML."""
        try:
            xml_content = xml_path.read_text(encoding="utf-8")
            root = etree.fromstring(xml_content.encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            xpaths = [
                "//tei:analytic/tei:title[@type='main']",
                "//tei:monogr/tei:title[@type='main']",
                "//tei:titleStmt/tei:title[@type='main']",
                "//tei:titleStmt/tei:title"
            ]
            for xp in xpaths:
                t = root.xpath(f"string({xp})", namespaces=ns)
                cleaned = self._normalize(t)
                if self._is_valid(cleaned):
                    return cleaned
        except Exception:
            return None
        return None

    def _extract_from_first_page_layout(self, pdf_file: Path) -> Optional[str]:
        """
        Layout + heuristic extraction: choose candidate block(s) from first page based on
        font size, vertical position, then filter out author/affiliation lines.
        """
        try:
            with fitz.open(pdf_file) as doc:
                if doc.page_count == 0:
                    return None
                page = doc.load_page(0)
                # extract dict to get blocks with font size / positions
                blocks = page.get_text("dict")["blocks"]
        except Exception:
            return None

        # Collect text segments with font size & position
        candidates: list[tuple[float, float, str]] = []  # (fontsize, y0, text)
        for b in blocks:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                for span in l["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    fontsize = span.get("size", 0)
                    y0 = span.get("origin", (0, 0))[1]
                    candidates.append((fontsize, y0, text))

        if not candidates:
            return None

        # Sort by fontsize descending, then y0 ascending (top of page)
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # Choose top-N spans (e.g., top 3) as potential title block
        top_spans = candidates[:3]
        combined = " ".join(span[2] for span in top_spans)
        cleaned = self._normalize(combined)

        # Filter out if looks like author/affiliation
        if self._looks_like_author_block(cleaned):
            # try next spans
            if len(candidates) > 3:
                alt_spans = candidates[3:6]
                combined2 = " ".join(span[2] for span in alt_spans)
                cleaned2 = self._normalize(combined2)
                if self._is_valid(cleaned2) and not self._looks_like_author_block(cleaned2):
                    return cleaned2
            return None

        if self._is_valid(cleaned):
            return cleaned

        return None

    def _looks_like_author_block(self, text: str) -> bool:
        """Heuristic to detect if a line/block is likely authors/affiliations rather than title."""
        if not text:
            return False
        # frequent keywords in affiliations/authors
        if re.search(r"\b(university|dept|department|institute|school|college|laboratory|lab|affiliat|author|authors?)\b", text, re.I):
            return True
        # many names separated by commas/and before a newline
        if re.match(r"[A-Z][a-z]+(\s[A-Z][a-z]+)+,?\s(and|&)\s[A-Z][a-z]+", text):
            return True
        return False

    def _looks_like_title(self, text: str) -> bool:
        """Heuristic: check if a line looks like a plausible title."""
        if not text or len(text) < 5 or len(text) > 250:
            return False
        # filter out common non-title keywords
        forbidden = r"\b(abstract|introduction|keywords?|acknowledgements|references?)\b"
        if re.search(forbidden, text, re.I):
            return False
        word_count = len(text.split())
        if word_count < 3 or word_count > 30:
            return False
        # Must start with uppercase letter (simplified)
        if not text[0].isupper():
            return False
        return True

    def _normalize(self, text: Optional[str]) -> str:
        """Normalize whitespace, remove soft-hyphens and ligatures."""
        if not text:
            return ""
        # merge hyphen-line breaks
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        # Remove arXiv IDs and version tags
        text = re.sub(r"arXiv:\d+\.\d+", "", text)  # Remove arXiv ID
        text = re.sub(r"v\d+", "", text)  # Remove version tags
        return text.strip()

    def _is_valid(self, text: Optional[str]) -> bool:
        """Check if a title candidate is semantically plausible."""
        if not text:
            return False
        t = text.strip()

        # Title should not be too short
        if len(t) < 10:
            return False
        # Title should have a reasonable word count
        if len(t.split()) < 3 or len(t.split()) > 30:
            return False

        # Exclude if it's just numbers or version identifiers
        if re.match(r"^[0-9._-]+$", t):
            return False

        # Exclude arXiv IDs and DOIs
        if re.search(r"arXiv:\d+\.\d+", t) or re.search(r"doi:\S+", t, re.IGNORECASE):
            return False

        # Exclude non-title fragments like "ENTREE, P2 Pl"
        if re.search(r"(P2\sPl|ENTREE|v\d+)", t, re.IGNORECASE):
            return False

        return True
