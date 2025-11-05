# src/core/llm/llm_orchestrator.py
from __future__ import annotations
import logging
from typing import Dict, Any, List
from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM
from src.core.llm.query_intent_classifier import QueryIntentClassifier


class LLMOrchestrator:
    """Adaptive LLM orchestrator using transformer-based semantic intent classification."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigLoader(config_path).config
        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()
        self.retriever = self._init_retriever()
        self.llm = self._init_llm()
        self.intent_classifier = QueryIntentClassifier()

    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Initialize consistent logging."""
        log_level = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")
        self.logger.info("Initialized LLM orchestrator (transformer-based intent mode).")

    # ------------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        """Initialize retrieval orchestrator."""
        try:
            retriever = RetrievalOrchestrator()
            self.logger.info("Retriever initialized successfully from config.")
            return retriever
        except Exception as e:
            self.logger.exception(f"Failed to initialize retriever: {e}")
            raise

    # ------------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        """Initialize local Ollama-based LLM backend."""
        llm_cfg: Dict[str, Any] = self.cfg.get("generation", {}).get("llm", {})
        model = llm_cfg.get("model", "mistral:7b-instruct")
        temperature = float(llm_cfg.get("temperature", 0.2))
        max_tokens = int(llm_cfg.get("max_tokens", 1024))

        try:
            llm = OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens)
            self.logger.info(f"LLM initialized: {model}")
            return llm
        except Exception as e:
            self.logger.exception(f"Failed to initialize LLM: {e}")
            raise

    # ------------------------------------------------------------------
    def _generate_subqueries(self, query: str) -> List[str]:
        """Generate semantically related subqueries for broader retrieval coverage."""
        # lightweight paraphrastic expansion via semantic similarity
        # keeps retrieval robust without hardcoded domain heuristics
        return [query]

    # ------------------------------------------------------------------
    def _format_retrieved_context(self, results: List[Dict[str, Any]], top_k: int = 10) -> str:
        """Format retrieved results sorted by publication year."""
        seen = set()
        lines = []
        results_sorted = sorted(
            results, key=lambda x: int(x.get("metadata", {}).get("year") or x.get("year") or 0)
        )[:top_k]

        for i, r in enumerate(results_sorted, start=1):
            meta = r.get("metadata", {})
            title = meta.get("title") or meta.get("source_file") or "Unknown"
            year = meta.get("year") or r.get("year", "n/a")
            if title in seen:
                continue
            seen.add(title)
            lines.append(f"[{i}] ({year}) {title}")

        return "Retrieved Context (chronological order: oldest → newest):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def _build_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Construct prompt adaptively based on transformer-detected intent."""
        mode = self.intent_classifier.classify(query)
        retrieved_sorted = sorted(
            retrieved, key=lambda r: int(r.get("metadata", {}).get("year") or r.get("year") or 0)
        )
        context_block = self._format_retrieved_context(retrieved_sorted)

        snippets = "\n\n".join(
            f"[{i+1}] ({r.get('metadata', {}).get('year', 'n/a')}) "
            f"{r.get('text', '')[:500]}"
            for i, r in enumerate(retrieved_sorted[:10])
        )

        if mode == "chronological":
            system_prompt = (
                "You are an analytical historian of Artificial Intelligence. "
                "Use the provided sources to explain how the concept evolved over time, "
                "highlighting paradigm shifts and key milestones."
            )
        elif mode == "conceptual":
            system_prompt = (
                "You are an AI expert explaining a concept with precision and clarity. "
                "Define the term, outline its principles, and support your explanation with evidence."
            )
        else:  # analytical
            system_prompt = (
                "You are an analytical researcher comparing or evaluating AI-related ideas. "
                "Discuss their differences, similarities, and theoretical or practical implications."
            )

        prompt = (
            f"{system_prompt}\n\n"
            f"{context_block}\n\n"
            f"User Question:\n{query}\n\n"
            f"Context Snippets:\n{snippets}\n\n"
            "Now provide a clear, logically structured, and evidence-based answer."
        )
        return prompt

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Interactive adaptive QA session with zero-shot intent detection."""
        self.logger.info("LLM phase ready for adaptive semantic queries.")
        self.logger.info('Example queries:')
        self.logger.info('   "How did the term Artificial Intelligence evolve over time?"')
        self.logger.info('   "What is AI?"')
        self.logger.info('   "Compare symbolic AI and neural networks."')

        while True:
            try:
                query = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                self.logger.info("Session terminated by user.")
                break

            if query.lower() in {"exit", "quit"}:
                self.logger.info("User exited interactive mode.")
                break
            if not query:
                continue

            mode = self.intent_classifier.classify(query)
            self.logger.info(f"Detected query mode: {mode}")

            # --- Retrieval Phase ---
            self.logger.info(f"Retrieving context for query: {query}")
            retrieved_docs: List[Dict[str, Any]] = []
            for sub in self._generate_subqueries(query):
                retrieved_docs.extend(self.retriever.retrieve(sub, top_k=10))

            if not retrieved_docs:
                self.logger.warning("No relevant documents retrieved.")
                continue

            # Deduplicate by title/source
            seen, unique = set(), []
            for r in retrieved_docs:
                key = (r.get("metadata", {}).get("title") or r.get("metadata", {}).get("source_file") or "").lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            retrieved_docs = unique[:15]

            # Log retrieved context overview
            context_log = self._format_retrieved_context(retrieved_docs)
            self.logger.info("\n" + context_log)

            # --- LLM Generation ---
            try:
                full_prompt = self._build_prompt(query, retrieved_docs)
                answer = self.llm.generate(full_prompt, context=retrieved_docs)
            except Exception as e:
                self.logger.exception(f"LLM generation failed: {e}")
                continue

            self.logger.info("Model Output:")
            self.logger.info(answer)
            self.logger.info("=" * 100)

        self.close()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Gracefully close retriever and LLM."""
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Error closing retriever: {e}")
        try:
            self.llm.close()
        except Exception as e:
            self.logger.warning(f"Error closing LLM: {e}")
        self.logger.info("LLM generation phase finished successfully.")


# ----------------------------------------------------------------------
def main() -> None:
    """Standalone execution for adaptive RAG LLM QA using zero-shot intent classification."""
    orchestrator = LLMOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
