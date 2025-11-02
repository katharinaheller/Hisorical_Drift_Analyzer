from __future__ import annotations
from typing import List
from src.core.ingestion.cleaner.i_text_cleaner import ITextCleaner

from src.core.ingestion.cleaner.simple_cleaners import (
    UnicodeNormalizer,
    SoftHyphenCleaner,
    HeaderFooterCleaner,
    LayoutLineJoinCleaner,
    TrailingWhitespaceCleaner,
    HTMLCleaner,                 # removes HTML tags and entities
    ScientificNotationCleaner,   # removes scientific notation terms like "Eq. 1"
    ReferencesCleaner,           # removes everything after 'References' or 'Bibliography'
)

from src.core.ingestion.cleaner.deep_text_cleaner import DeepTextCleaner  # advanced cleaning


class RagTextCleaner(ITextCleaner):
    """
    Composite text cleaner for RAG ingestion.
    Executes a deterministic sequence of cleaners to normalize and denoise scientific text.
    """

    def __init__(self, cleaners: List[ITextCleaner]):
        self.cleaners = cleaners

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "RagTextCleaner":
        """
        Build a deterministic chain of text cleaners combining rule-based and deep cleaning.
        """
        cleaners: List[ITextCleaner] = [
            UnicodeNormalizer(),         # normalize spaces and zero-width chars
            SoftHyphenCleaner(),         # remove soft hyphens and join split words
            HeaderFooterCleaner(),       # drop headers, footers, funding info
            LayoutLineJoinCleaner(),     # repair layout-induced line breaks
            TrailingWhitespaceCleaner(), # trim redundant spaces and newlines
            HTMLCleaner(),               # remove HTML tags and entities
            ScientificNotationCleaner(), # remove equations, theorem/lemma markers
            ReferencesCleaner(),         # cut after "References"/"Bibliography"
            DeepTextCleaner(),           # deep filter for noise, refs, tables, math etc.
        ]
        return cls(cleaners)

    # ------------------------------------------------------------------
    def clean(self, text: str) -> str:
        """
        Run all sub-cleaners consecutively in deterministic order.
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text.strip()
