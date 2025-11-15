from __future__ import annotations
import re
import datetime
import time
from typing import Iterable, Tuple, Optional, Dict, List
from pathlib import Path
import fitz  # PyMuPDF
from lxml import etree

# optional normalizer
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

# optional Crossref client
try:
    from habanero import Crossref
except Exception:
    Crossref = None  # type: ignore

CURRENT_YEAR = datetime.datetime.now().year
FUTURE_GRACE = 1  # allow slight future offset

# strict year patterns
YEAR_STRICT_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
YEAR_CONTEXT_RE = re.compile(
    r"(?:published\s+online[:\s]*|accepted[:\s]*|received[:\s]*|©|copyright)\s*((19|20)\d{2})",
    re.I
)


class YearExtractor:
    """Robust publication year extractor using multi-source strategies."""

    def __init__(self, base_dir: Path | str | None = None,
                 max_text_pages: int = 3,
                 enable_crossref: bool = True):

        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.max_text_pages = max(1, int(max_text_pages))
        self.crossref = None

        if enable_crossref and Crossref is not None:
            try:
                self.crossref = Crossref(mailto="contact@example.com")
            except Exception:
                self.crossref = None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> Optional[str]:
        """Main orchestrator for multi-source year extraction."""
        pdf_file = Path(pdf_path)
        candidates: List[Tuple[int, int, str]] = []  # (priority, score, year_str)

        # 1) GROBID TEI
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            year = self._extract_from_grobid(xml_path)
            if year:
                candidates.append((1, 100, year))

        # 2) PDF metadata (with consistency refinement)
        year_meta, meta_score = self._extract_from_pdf_metadata(pdf_file)
        if year_meta:
            refined = self._refine_with_text_consistency(pdf_file, int(year_meta))
            if refined:
                candidates.append((2, meta_score, refined))

        # 3) Early-page visible text
        year_text, text_score = self._extract_from_page_text(pdf_file, self.max_text_pages)
        if year_text:
            candidates.append((3, text_score, year_text))

        # 4) Filename heuristics
        year_fn, fn_score = self._extract_from_filename(pdf_file.name)
        if year_fn:
            candidates.append((4, fn_score, year_fn))

        # 5) DOI/Title lookup
        if not candidates:
            title, doi, arxiv = self._extract_title_doi_arxiv(pdf_file)
            if arxiv:
                y = self._year_from_arxiv_id(arxiv)
                if y:
                    candidates.append((5, 60, str(y)))

            if (doi or title) and self.crossref:
                y = self._lookup_via_crossref(title, doi)
                if y:
                    candidates.append((6, 70, y))

        if not candidates:
            return None

        # sort by priority, score, then by recency
        candidates.sort(key=lambda t: (t[0], -t[1], -int(t[2])))
        return candidates[0][2]

    # ------------------------------------------------------------------
    def _lookup_via_crossref(self, title: Optional[str], doi: Optional[str]) -> Optional[str]:
        """Crossref lookup as fallback (no hardcoded year rejection)."""
        if not self.crossref:
            return None

        for attempt in range(2):
            try:
                if doi:
                    msg = self.crossref.works(ids=doi).get("message", {})
                elif title:
                    msg = self.crossref.works(query=title, limit=1).get("message", {}).get("items", [{}])[0]
                else:
                    return None

                for key in ("published-print", "published-online", "issued"):
                    info = msg.get(key)
                    if info and "date-parts" in info:
                        y = info["date-parts"][0][0]
                        if self._valid_year(y):
                            return str(y)

            except Exception:
                time.sleep(1)

        return None

    # ------------------------------------------------------------------
    def _extract_title_doi_arxiv(self, pdf_file: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract DOI, arXiv ID, title from first pages."""
        doi_re = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
        arxiv_re = re.compile(
            r"\b(?:arxiv[:/ ]?)?(\d{4}\.\d{4,5}|[a-z\-]+/\d{7}|[0-9]{7,8})(v\d+)?\b",
            re.I
        )

        title = doi = arxiv = None

        try:
            with fitz.open(pdf_file) as doc:
                meta_title = (doc.metadata or {}).get("title")
                if meta_title:
                    title = meta_title.strip()

                txt = ""
                for i in range(min(3, len(doc))):
                    txt += (doc.load_page(i).get_text("text") or "") + "\n"
        except Exception:
            txt = ""

        if unidecode and txt:
            txt = unidecode(txt)

        m = doi_re.search(txt)
        if m:
            doi = m.group(0).rstrip(".,)")

        m = arxiv_re.search(txt)
        if m:
            arxiv = m.group(1)

        if not title and txt:
            for line in txt.splitlines():
                s = line.strip()
                if len(s) > 10 and not re.search(r"(abstract|introduction|contents)", s, re.I):
                    title = s
                    break

        return title, doi, arxiv

    # ------------------------------------------------------------------
    def _year_from_arxiv_id(self, arxiv_id: str) -> Optional[int]:
        """Interpret arXiv ID year component."""
        try:
            m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", arxiv_id)
            if m:
                yy = int(m.group(1))
                y = 2000 + yy if yy < 25 else 1900 + yy
                return y if self._valid_year(y) else None

            m = re.match(r"^[a-z\-]+/(\d{2})(\d{2})\d{3,4}$", arxiv_id, re.I)
            if m:
                yy = int(m.group(1))
                y = 2000 + yy if yy < 25 else 1900 + yy
                return y if self._valid_year(y) else None

            m = re.match(r"^(\d{2})(\d{2})\d{3,4}$", arxiv_id)
            if m:
                yy = int(m.group(1))
                y = 2000 + yy if yy < 25 else 1900 + yy
                return y if self._valid_year(y) else None

        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    def _valid_year(self, y: int) -> bool:
        """Reject only unrealistic values."""
        return 1800 <= y <= CURRENT_YEAR + FUTURE_GRACE

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Optional[Path]:
        """Locate TEI XML file."""
        cand = pdf_file.with_suffix(".tei.xml")
        if cand.exists():
            return cand

        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt

        return None

    # ------------------------------------------------------------------
    def _extract_from_grobid(self, xml_path: Path) -> Optional[str]:
        """Extract year from TEI XML."""
        try:
            xml = etree.fromstring(xml_path.read_bytes())
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}

            xps = [
                "//tei:sourceDesc//tei:imprint/tei:date",
                "//tei:biblStruct//tei:imprint/tei:date",
                "//tei:profileDesc//tei:creation/tei:date"
            ]

            for xp in xps:
                for node in xml.xpath(xp, namespaces=ns):

                    for key in ("when", "when-iso", "notBefore", "notAfter"):
                        val = node.get(key)
                        if val:
                            y = self._year_from_date_string(val)
                            if y:
                                return str(y)

                    txt = (node.text or "").strip()
                    for y in self._years_from_string(txt):
                        return str(y)

        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> Tuple[Optional[str], int]:
        """Parse metadata and extract plausible year."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                kv = {k.lower(): v for k, v in meta.items() if isinstance(v, str)}

                best = (None, -1)

                for key, score in [
                    ("creationdate", 80),
                    ("moddate", 60),
                    ("date", 50)
                ]:
                    val = kv.get(key)
                    if not val:
                        continue

                    y = self._year_from_date_string(val)
                    if y and self._valid_year(y):
                        if score > best[1]:
                            best = (y, score)

                if best[0]:
                    return str(best[0]), best[1]

        except Exception:
            pass

        return None, -1

    # ------------------------------------------------------------------
    def _refine_with_text_consistency(self, pdf_file: Path, year_meta: int) -> Optional[str]:
        """Check consistency of metadata with surrounding text."""
        try:
            with fitz.open(pdf_file) as doc:
                pages = []
                for i in range(min(3, len(doc))):
                    pages.append(doc.load_page(i).get_text("text") or "")
                for i in range(max(0, len(doc) - 2), len(doc)):
                    pages.append(doc.load_page(i).get_text("text") or "")
                txt = "\n".join(pages)

        except Exception:
            return str(year_meta)

        if unidecode:
            txt = unidecode(txt)

        # strong context match
        ctx = YEAR_CONTEXT_RE.findall(txt)
        if ctx:
            yrs = [int(x[0]) for x in ctx if self._valid_year(int(x[0]))]
            if yrs:
                return str(max(yrs))

        # fallback: largest strict year in text
        yrs = [int(m.group()) for m in YEAR_STRICT_RE.finditer(txt)]
        yrs = [y for y in yrs if self._valid_year(y)]

        if not yrs:
            return str(year_meta)

        y_txt = max(yrs)

        # if metadata and text differ too much, trust text
        if abs(y_txt - year_meta) >= 10:
            return str(y_txt)

        return str(year_meta)

    # ------------------------------------------------------------------
    def _extract_from_page_text(self, pdf_file: Path, pages: int) -> Tuple[Optional[str], int]:
        """Extract year from early page content."""
        try:
            with fitz.open(pdf_file) as doc:
                best = (None, -1)

                for i in range(min(len(doc), pages)):
                    txt = doc.load_page(i).get_text("text") or ""

                    if unidecode:
                        txt = unidecode(txt)

                    # direct contextual indicators
                    ctx = YEAR_CONTEXT_RE.findall(txt)
                    if ctx:
                        yrs = [int(x[0]) for x in ctx if self._valid_year(int(x[0]))]
                        if yrs:
                            y = max(yrs)
                            score = 95 - i
                            return str(y), score

                    # strict year pattern
                    yrs = [int(m.group()) for m in YEAR_STRICT_RE.finditer(txt)]
                    yrs = [y for y in yrs if self._valid_year(y)]

                    if yrs:
                        y = max(yrs)
                        score = 45 + (pages - i)
                        if score > best[1]:
                            best = (y, score)

                if best[0]:
                    return str(best[0]), best[1]

        except Exception:
            pass

        return None, -1

    # ------------------------------------------------------------------
    def _extract_from_filename(self, name: str) -> Tuple[Optional[str], int]:
        """Extract year from filename."""
        base = unidecode(name) if unidecode else name
        base = re.sub(r"v\d+\b", "", base, flags=re.I)

        yrs = [y for y in self._years_from_string(base) if self._valid_year(y)]
        if yrs:
            return str(max(yrs)), 30

        return None, -1

    # ------------------------------------------------------------------
    def _year_from_date_string(self, s: str) -> Optional[int]:
        """Interpret year in date strings."""
        if not s:
            return None

        s = s.strip()

        m = re.match(r"^D:(\d{4})", s)
        if m:
            y = int(m.group(1))
            return y if self._valid_year(y) else None

        m = re.match(r"^(\d{4})(?:[-/]\d{2}(?:[-/]\d{2})?)?$", s)
        if m:
            y = int(m.group(1))
            return y if self._valid_year(y) else None

        for y in self._years_from_string(s):
            return y

        return None

    # ------------------------------------------------------------------
    def _years_from_string(self, s: str) -> Iterable[int]:
        """Find all strict isolated 4-digit years."""
        seen = set()
        for m in YEAR_STRICT_RE.finditer(s):
            y = int(m.group(0))
            if y not in seen and self._valid_year(y):
                seen.add(y)
                yield y
