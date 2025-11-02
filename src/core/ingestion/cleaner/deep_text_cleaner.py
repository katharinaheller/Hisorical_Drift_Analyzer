from __future__ import annotations
import re
from cleantext import clean
from src.core.ingestion.cleaner.base_cleaner import BaseTextCleaner

class DeepTextCleaner(BaseTextCleaner):
    """
    Advanced text cleaner using heuristic and statistical patterns
    to remove non-content sections like references, tables, and equations.
    """

    def _clean_impl(self, text: str) -> str:
        # Step 1: Global clean using clean-text
        text = clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_currency_symbols=True,
            no_punct=False,
        )

        # Step 2: Remove reference-like or appendix sections
        text = re.sub(
            r"(?is)(references|bibliography|literaturverzeichnis|acknowledg(e)?ments|appendix).*",
            "",
            text,
        )

        # Step 3: Drop DOI/arXiv/URL lines
        text = re.sub(r"(?im)^\s*(doi|arxiv|http|https)[:\s].*$", "", text)

        # Step 4: Drop lines that are mostly numbers, formulas, or citation lists
        filtered_lines = []
        for line in text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            # Discard lines with >30% digits or symbols
            digit_ratio = sum(ch.isdigit() for ch in clean_line) / max(len(clean_line), 1)
            symbol_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in clean_line) / max(len(clean_line), 1)
            if digit_ratio > 0.3 or symbol_ratio > 0.4:
                continue
            # Drop if line looks like citation or table
            if re.match(r"^\[\d+\]\s*[A-Z][a-z]+", clean_line):
                continue
            if re.match(r"(?i)^table\s+\d+", clean_line):
                continue
            filtered_lines.append(clean_line)

        text = "\n".join(filtered_lines)

        # Step 5: Collapse whitespace and blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
