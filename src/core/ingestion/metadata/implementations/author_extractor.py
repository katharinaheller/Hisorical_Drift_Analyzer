# src/core/ingestion/metadata/implementations/author_extractor.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import fitz
import logging
from lxml import etree
import re

# optional imports
try:
    import spacy
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from nameparser import HumanName
    NAMEPARSER_AVAILABLE = True
except ImportError:
    NAMEPARSER_AVAILABLE = False


class LocalNameVerifier:
    """Deterministic local fallback for name validation without ML models."""

    TITLE_PATTERN = re.compile(r"^(dr|prof|mr|ms|mrs|herr|frau)\.?\s", re.IGNORECASE)
    NAME_PATTERN = re.compile(r"^[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+)*$")

    def __init__(self):
        # extended bad keyword set for academic false positives
        self.bad_keywords = {
            "abstract", "introduction", "university", "department", "institute",
            "school", "machine", "learning", "arxiv", "neural", "transformer",
            "appendix", "algorithm", "keywords", "vol", "doi", "intelligence",
            "markov", "chain", "computing", "zentrum", "sciences",
            "centre", "center", "data", "model", "probability", "network"
        }

    def is_person_name(self, text: str) -> bool:
        if not text or len(text) > 80:
            return False
        if any(ch.isdigit() for ch in text):
            return False
        if any(tok.lower() in self.bad_keywords for tok in text.split()):
            return False

        if NAMEPARSER_AVAILABLE:
            try:
                hn = HumanName(text)
                if hn.first or hn.last:
                    return True
            except Exception:
                pass

        if self.TITLE_PATTERN.search(text):
            return True
        return bool(self.NAME_PATTERN.search(text))


class AuthorsExtractor:
    """Deterministic, language-agnostic author extractor with silent mode (no info logs)."""

    def __init__(self, base_dir: Path | str | None = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.name_verifier = LocalNameVerifier()
        self.nlp = None

        if SPACY_AVAILABLE:
            self.nlp = self._init_spacy_pipeline()
        else:
            self.logger.warning("spaCy not installed; PERSON detection disabled.")

    def _init_spacy_pipeline(self):
        """Try to load spaCy model, else build local EntityRuler fallback."""
        try:
            return spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "parser"])
        except Exception:
            try:
                return spacy.load("de_core_news_sm", disable=["tagger", "lemmatizer", "parser"])
            except Exception:
                self.logger.warning("No spaCy model available; using local EntityRuler fallback.")
                return self._build_local_person_ruler()

    def _build_local_person_ruler(self) -> "Language":
        """Constructs a blank English spaCy pipeline with a rule-based PERSON detector."""
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]},
            {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_ALPHA": True}, {"IS_ALPHA": True}]},
            {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"TEXT": {"REGEX": "^[A-Z][a-z]+\\.$"}}]},
        ]
        ruler.add_patterns(patterns)
        return nlp

    def extract(self, pdf_path: str, parsed_document: Dict[str, Any] | None = None) -> List[str]:
        candidates: List[str] = []

        if parsed_document and parsed_document.get("grobid_xml"):
            candidates.extend(self._extract_from_grobid_xml(parsed_document["grobid_xml"]))
        else:
            xml_path = self._find_grobid_sidecar(Path(pdf_path))
            if xml_path:
                candidates.extend(self._extract_from_grobid_file(xml_path))

        if not candidates:
            candidates.extend(self._extract_from_pdf_metadata(Path(pdf_path)))

        if not candidates:
            candidates.extend(self._extract_from_first_page(Path(pdf_path)))

        candidates = [self._normalize(c) for c in candidates if c.strip()]
        verified = [n for n in candidates if self.name_verifier.is_person_name(n)]

        cleaned = [
            re.sub(
                r"\b(Google|Inc\.?|University|Institute|Lab|Labs|Dept\.?|Center|Centre|Zentrum|Sciences?)\b",
                "",
                n,
                flags=re.IGNORECASE
            ).strip()
            for n in verified
        ]

        seen: set[str] = set()
        return [a for a in cleaned if not (a in seen or seen.add(a))]

    def _find_grobid_sidecar(self, pdf_file: Path) -> Optional[Path]:
        local = pdf_file.with_suffix(".tei.xml")
        if local.exists():
            return local
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    def _extract_from_grobid_file(self, xml_path: Path) -> List[str]:
        try:
            return self._extract_from_grobid_xml(xml_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"GROBID XML read error: {e}")
            return []

    def _extract_from_grobid_xml(self, xml_text: str) -> List[str]:
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except Exception as e:
            self.logger.warning(f"GROBID parse error: {e}")
            return []

        authors: List[str] = []
        for xp in [
            "//tei:analytic//tei:author",
            "//tei:monogr//tei:author",
            "//tei:respStmt//tei:author",
            "//tei:editor",
        ]:
            for node in root.xpath(xp, namespaces=ns):
                fn = node.xpath("string(.//tei:forename)", namespaces=ns).strip()
                ln = node.xpath("string(.//tei:surname)", namespaces=ns).strip()
                if fn or ln:
                    authors.append(f"{fn} {ln}".strip())
                else:
                    txt = " ".join(node.xpath(".//text()", namespaces=ns)).strip()
                    if txt:
                        authors.extend(self._extract_persons_ner(txt))
        return authors

    def _extract_from_pdf_metadata(self, pdf_file: Path) -> List[str]:
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
        except Exception as e:
            self.logger.warning(f"PDF metadata read error: {e}")
            return []
        author_field = meta.get("author") or meta.get("Author") or meta.get("authors")
        return self._extract_persons_ner(str(author_field)) if author_field else []

    def _extract_from_first_page(self, pdf_file: Path) -> List[str]:
        try:
            with fitz.open(pdf_file) as doc:
                if doc.page_count == 0:
                    return []
                text = doc.load_page(0).get_text("text")
        except Exception as e:
            self.logger.warning(f"First-page read error: {e}")
            return []
        text = text.split("Abstract")[0] if "Abstract" in text else text
        return self._extract_persons_ner(text)

    def _extract_persons_ner(self, text: str) -> List[str]:
        if not self.nlp or not text.strip():
            return []
        doc = self.nlp(text)
        persons = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if any(e.start <= ent.start < e.end for e in doc.ents if e.label_ in {"ORG", "GPE", "LOC"}):
                    continue
                persons.append(ent.text.strip())
        return [p for p in persons if self._is_name_like(p)]

    def _is_name_like(self, text: str) -> bool:
        bad_keywords = {
            "abstract", "introduction", "university", "department", "institute",
            "arxiv", "model", "learning", "probability", "appendix", "keywords",
            "ai", "neural", "transformer", "algorithm", "vol",
            "intelligence", "computing", "zentrum", "sciences", "centre", "center"
        }
        if any(tok.lower() in bad_keywords for tok in text.split()):
            return False
        if any(ch.isdigit() for ch in text):
            return False
        return 1 <= len(text.split()) <= 5 and text[0].isupper()

    def _normalize(self, s: str) -> str:
        """Normalize spacing and special characters."""
        return " ".join(s.replace("•", "").replace("†", "").split()).strip()
