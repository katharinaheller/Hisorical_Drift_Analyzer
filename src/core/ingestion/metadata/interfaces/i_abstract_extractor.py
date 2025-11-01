"""from __future__ import annotations
from typing import Any, Dict
from lxml import etree
import re
from src.core.ingestion.metadata.interfaces.i_abstract_extractor import IAbstractExtractor

class AbstractExtractor(IAbstractExtractor):
    """Extracts abstract section using GROBID TEI XML."""

    def extract(self, pdf_path: str, parsed_document: Dict[str, Any]) -> str | None:
        xml_data = parsed_document.get("grobid_xml")
        if not xml_data or not xml_data.strip().startswith("<"):
            return None

        try:
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            root = etree.fromstring(xml_data.encode("utf8"))
            # collect all text under <abstract> or <div type='abstract'>
            xpath_candidates = [
                "//tei:abstract",
                "//tei:div[@type='abstract']",
                "//tei:profileDesc/tei:abstract",
            ]
            for path in xpath_candidates:
                text = root.xpath(f"string({path})", namespaces=ns)
                if text and len(text.strip()) > 20:
                    cleaned = re.sub(r"\s+", " ", text.strip())
                    return cleaned
        except Exception:
            return None
        return None
"""