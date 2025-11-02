from __future__ import annotations
import re
from typing import List
from src.core.ingestion.cleaner.base_cleaner import BaseTextCleaner
from src.core.ingestion.cleaner import rules


class HTMLCleaner(BaseTextCleaner):
    """Removes HTML tags and entities like &nbsp;."""

    def _clean_impl(self, text: str) -> str:
        # Remove all HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove HTML entities like &nbsp;
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        return text


class ScientificNotationCleaner(BaseTextCleaner):
    """Removes scientific notations and references like 'Eq. 1', 'Theorem 3'."""

    def _clean_impl(self, text: str) -> str:
        # Remove scientific format like 'Eq. 1', 'Theorem 2'
        text = re.sub(r'\b(Eq|Theorem|Lemma)\s+\d+\b', '', text)
        return text


class UnicodeNormalizer(BaseTextCleaner):
    """Normalize unicode spaces and zero-width chars."""

    def _clean_impl(self, text: str) -> str:
        # Replace unicode spaces by normal space
        text = rules.UNICODE_SPACES_RE.sub(" ", text)
        return text


class SoftHyphenCleaner(BaseTextCleaner):
    """Remove soft hyphens and join line-broken words."""

    def _clean_impl(self, text: str) -> str:
        # Remove soft hyphen character
        text = text.replace(rules.SOFT_HYPHEN, "")
        # Join hyphenated line breaks
        text = rules.HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
        return text


class LayoutLineJoinCleaner(BaseTextCleaner):
    """
    Join lines that were broken by the PDF layout but belong together.
    We do this conservatively: only if both sides are non-space.
    """

    def _clean_impl(self, text: str) -> str:
        # Join inline linebreaks like "probabilistic\nreasoning" -> "probabilistic reasoning"
        text = rules.INLINE_LINEBREAK_RE.sub(r"\1 \2", text)
        # Collapse too many blank lines
        text = rules.MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()


class HeaderFooterCleaner(BaseTextCleaner):
    """Remove obvious header/footer/single-line noise."""

    def _clean_impl(self, text: str) -> str:
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            raw = line.rstrip()

            # Drop header-like lines
            if rules.match_any(rules.SINGLE_LINE_HEADER_PATTERNS, raw):
                continue

            # Drop footer-like things only if short
            if rules.match_any(rules.FOOTER_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            # Drop funding/preprint if short
            if rules.match_any(rules.FUNDING_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            cleaned_lines.append(raw)

        return "\n".join(cleaned_lines).strip()


class TrailingWhitespaceCleaner(BaseTextCleaner):
    """Normalize whitespace globally."""

    def _clean_impl(self, text: str) -> str:
        # Strip each line
        lines = [ln.rstrip() for ln in text.splitlines()]
        text = "\n".join(lines)
        # Final collapse of multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class ReferencesCleaner(BaseTextCleaner):
    """Removes everything starting from 'References', 'Bibliography', or 'Literaturverzeichnis' sections."""

    def _clean_impl(self, text: str) -> str:
        # Pattern for reference section headers (case-insensitive)
        pattern = re.compile(
            r"(?im)^\s*(references|bibliography|literaturverzeichnis)\s*$"
        )
        match = pattern.search(text)
        if match:
            # Cut everything from the start of the reference section
            cutoff_index = match.start()
            # Only cut if it occurs after the first 20% of the document to avoid false positives
            if cutoff_index > len(text) * 0.2:
                text = text[:cutoff_index]
        return text.strip()
