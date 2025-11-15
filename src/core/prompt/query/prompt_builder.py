from __future__ import annotations
import logging
from typing import Literal

logger = logging.getLogger(__name__)
Intent = Literal["chronological", "conceptual", "analytical", "comparative"]
CitationStyle = Literal["ieee"]


class PromptBuilder:
    """
    Builds intent-specific system prompts for RAG/LLM orchestration.

    Fully prevents event-year hallucinations by enforcing that the model
    may use ONLY explicitly stated event years from the snippets.
    """

    def __init__(self, citation_style: CitationStyle = "ieee"):
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        self.citation_style = citation_style

    # ------------------------------------------------------------------
    def _citation_instruction(self) -> str:
        """IEEE-style citation policy with strict anti-hallucination rules."""
        return (
            "Use numeric IEEE-style citations [1], [2], etc., for statements supported by the provided snippets. "
            "Each number corresponds to one unique PDF listed below. Multiple snippets originating from the same PDF share the same number. "
            "Never assign multiple citation numbers to the same source.\n\n"

            "**Your final answer MUST end with a separate section titled 'References'.**\n"
            "This section MUST list all unique PDFs exactly once, in the following strict format:\n"
            "[n] FILENAME.pdf (YEAR)\n\n"

            "Do not fabricate author names, journals, or article titles — only use the given filename and metadata year.\n\n"

            "Temporal Attribution Rules:\n"
            "1. You may ONLY use event years that appear explicitly in the snippet text.\n"
            "2. If the snippet text explicitly contains a year (e.g., 'In the 1950s', 'In 1976'), treat that as the factual historical reference.\n"
            "3. If a snippet DOES NOT contain an explicit event year, you MUST NOT guess, infer, approximate, or estimate any year.\n"
            "   Instead, write exactly: '(event year not stated; described in YEAR PDF [n])'.\n"
            "4. The metadata publication year indicates only when the PDF was published, not when the events occurred.\n"
            "5. Never replace or override an explicit event year with a metadata year.\n"
            "6. Never deduce approximate historical periods from textual content (e.g., never infer '1990s' unless explicitly stated).\n\n"

            "Output Structuring Guidelines:\n"
            "- For every key historical or conceptual point:\n"
            "  • If an explicit event year exists in the snippet → include it.\n"
            "  • If no explicit event year exists → write '(event year not stated; described in YEAR PDF [n])'.\n"
            "- Recommended dual-year structure:\n"
            "  • (1950s; described in 2025 PDF [7]) The Turing Test was proposed as a benchmark.\n"
            "This dual timestamping ensures full temporal grounding without hallucination."
        )

    # ------------------------------------------------------------------
    def _system_prompt_for(self, intent: Intent) -> str:
        """Return intent-specific system instruction with hallucination-proof constraints."""
        cite_rule = self._citation_instruction()

        end_rule = (
            "\n\nIMPORTANT:\n"
            "**Your output MUST end with a final section titled 'References'.**\n"
            "This section must list all unique PDFs exactly once in IEEE numeric format.\n"
        )

        if intent == "chronological":
            return (
                "You are an analytical historian of Artificial Intelligence. "
                "Describe how the concept evolved across time, highlighting paradigm shifts, milestones, and key theoretical transformations. "
                "Present findings in a coherent historical narrative ordered strictly by explicit *event years* found in the snippets. "
                "If a snippet provides no explicit event year, you MUST write '(event year not stated; described in YEAR PDF [n])'. "
                "Never guess or estimate historical periods under any circumstances. "
                "Avoid enumeration; emphasize causal relations and conceptual transitions. "
                f"{cite_rule}"
                f"{end_rule}"
            )

        if intent == "conceptual":
            return (
                "You are a domain expert in Artificial Intelligence. Provide a precise definition, clarify theoretical foundations, "
                "and explain how interpretations evolved across time and publications. "
                "Use event years ONLY if explicitly stated in the snippets. "
                f"{cite_rule}"
                f"{end_rule}"
            )

        if intent == "analytical":
            return (
                "You are a rigorous AI researcher. Analyze mechanisms, methodologies, and implications over time. "
                "Event years may only be used if explicitly present in the snippet text. "
                f"{cite_rule}"
                f"{end_rule}"
            )

        # comparative
        return (
            "You are a comparative analyst. Compare major frameworks or schools of thought, "
            "specifying explicit historical information only when stated in the provided snippets. "
            "Never infer missing event years. "
            f"{cite_rule}"
            f"{end_rule}"
        )

    # ------------------------------------------------------------------
    def reformulate_query(self, query: str, intent: Intent) -> str:
        """Intent-guided canonical reformulation."""
        if not query or not query.strip():
            raise ValueError("Empty query cannot be reformulated")

        q = query.strip()

        if intent == "chronological":
            return (
                f"Trace the historical development and evolution of {q} strictly through the explicit event years present in the snippets. "
                "If no explicit event year is present for a point, note that the event year is not stated."
            )

        if intent == "conceptual":
            return (
                f"Define {q}, describe its theoretical foundations, and explain how definitions evolved historically across publications."
            )

        if intent == "analytical":
            return (
                f"Analyze the mechanisms, strengths, and limitations of {q}, noting origins only when explicitly stated."
            )

        if intent == "comparative":
            return (
                f"Compare and contrast the main theoretical perspectives on {q}, grounding historical claims only in explicit snippet content."
            )

        return q

    # ------------------------------------------------------------------
    def build_prompt(self, query: str, intent: Intent) -> str:
        """Return the complete system prompt including citation + temporal grounding."""
        if not query or not query.strip():
            raise ValueError("Empty query passed to PromptBuilder")
        return self._system_prompt_for(intent)
