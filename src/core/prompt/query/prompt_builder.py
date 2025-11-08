from __future__ import annotations
import html
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)
Intent = Literal["chronological", "conceptual", "analytical", "comparative"]


@dataclass
class PromptBuilderConfig:
    # # max chars per consolidated source snippet
    snippet_char_limit: int = 700
    # # sort sources by year ascending for temporal stability
    sort_chronologically: bool = True
    # # include a compact source overview header
    include_overview: bool = True
    # # enforce numeric-only citations in requirements
    numeric_citations_only: bool = True
    # # drop snippets below a minimal informational length
    min_snippet_len: int = 60


class PromptBuilder:
    """Compose the final LLM prompt using grouped, IEEE-style context references."""

    def __init__(self, cfg: Optional[PromptBuilderConfig] = None):
        self.cfg = cfg or PromptBuilderConfig()
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ------------------------------------------------------------------
    def _safe_year(self, item: Dict[str, Any]) -> int:
        # # parse year defensively
        meta = item.get("metadata", {}) or {}
        year = meta.get("year") or item.get("year") or 0
        try:
            y = int(year)
            return y if 0 < y < 3000 else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        # # normalize whitespace and escape control chars
        t = (text or "").replace("\n", " ").replace("\r", " ")
        t = " ".join(t.split())
        return html.unescape(t)

    # ------------------------------------------------------------------
    def _group_by_source(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # # group chunks per PDF; keep oldest→newest ordering
        grouped: Dict[str, Dict[str, Any]] = {}
        for chunk in items:
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file") or "Unknown.pdf"
            year = meta.get("year") or chunk.get("year", "n/a")
            text = self._clean_text(chunk.get("text", ""))
            if len(text) < self.cfg.min_snippet_len:
                continue
            if src not in grouped:
                grouped[src] = {"year": year, "chunks": []}
            grouped[src]["chunks"].append(text)

        ordered = list(grouped.items())
        if self.cfg.sort_chronologically:
            ordered.sort(key=lambda x: self._safe_year({"metadata": {"year": x[1]["year"]}}))

        result: List[Dict[str, Any]] = []
        for i, (src, meta) in enumerate(ordered, start=1):
            joined = " ".join(meta["chunks"])
            result.append(
                {
                    "index": i,
                    "source_file": src,
                    "year": meta["year"],
                    "text": joined,
                }
            )
        return result

    # ------------------------------------------------------------------
    def _context_overview(self, grouped: List[Dict[str, Any]]) -> str:
        # # compact list for quick visual scan
        lines = [f"[{g['index']}] ({g['year']}) {g['source_file']}" for g in grouped]
        return "Retrieved Context (oldest → newest):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def _system_prompt_for(self, intent: Intent) -> str:
        # # intent-specific instruction; consistent citation rule
        cite_rule = "Use numeric citations [1], [2], etc., and avoid author names or years in parentheses."
        if intent == "chronological":
            return (
                "You are an analytical historian of Artificial Intelligence. "
                "Explain how the concept evolved over time, highlighting paradigm shifts, research trends, and milestones. "
                f"{cite_rule}"
            )
        if intent == "conceptual":
            return (
                "You are an AI expert. Provide a precise definition, core principles, and a clear explanation of the concept. "
                f"{cite_rule}"
            )
        if intent == "analytical":
            return (
                "You are a rigorous AI researcher. Analyze mechanisms, trade-offs, and implications with clear argumentation. "
                f"{cite_rule}"
            )
        # # comparative
        return (
            "You are an analytical researcher. Compare and contrast positions and assess their empirical support. "
            f"{cite_rule}"
        )

    # ------------------------------------------------------------------
    def _snippets_block(self, grouped: List[Dict[str, Any]]) -> str:
        # # build consolidated per-source snippet with char cap
        limit = max(1, self.cfg.snippet_char_limit)
        parts: List[str] = []
        for g in grouped:
            text = self._clean_text(g["text"])[:limit].rstrip()
            parts.append(f"[{g['index']}] ({g['year']}) {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def build_prompt(self, query: str, intent: Intent, retrieved_topk: List[Dict[str, Any]]) -> str:
        # # construct a clean, grouped, IEEE-conform LLM prompt
        if not query or not query.strip():
            raise ValueError("Empty query passed to PromptBuilder")
        if not isinstance(retrieved_topk, list) or len(retrieved_topk) == 0:
            raise ValueError("Empty retrieved list passed to PromptBuilder")

        grouped = self._group_by_source(retrieved_topk)
        if not grouped:
            raise ValueError("No valid snippets available after grouping/filtering")

        sys_prompt = self._system_prompt_for(intent)
        overview = self._context_overview(grouped) if self.cfg.include_overview else ""
        snippets = self._snippets_block(grouped)

        requirements = [
            "Be concise, logically structured, and evidence-based.",
            "Base claims only on the Context Snippets.",
            "Cite using only numeric indices like [1], [2], etc.",
            "If a required fact is missing, state: 'not stated in the context'.",
            "Do not include author names or explicit years in parentheses.",
            "Prefer exact quotations (short) over paraphrases when wording matters.",
        ]

        prompt = f"{sys_prompt}\n\n"
        if overview:
            prompt += f"{overview}\n\n"
        prompt += (
            f"User Question:\n{query.strip()}\n\n"
            f"Context Snippets:\n{snippets}\n\n"
            "Answer requirements:\n- " + "\n- ".join(requirements) + "\n"
        )
        return prompt
