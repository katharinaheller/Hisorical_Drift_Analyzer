# src/core/llm/llm_orchestrator.py
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.prompt.query.prompt_builder import PromptBuilder


class LLMOrchestrator:
    """Full orchestration: Prompt → Retrieval → Reranking → LLM → IEEE-style Output + unified logging for evaluation."""

    def __init__(self, config_path: str = "configs/llm.yaml"):
        # Load configuration
        self.cfg = ConfigLoader(config_path).config
        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()

        # Initialize pipeline components
        self.prompt_phase = PromptOrchestrator()
        self.retriever = self._init_retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = self._init_llm()

        self.logger.info("Initialized full LLM pipeline orchestrator (Prompt → Retrieval → LLM).")

    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        # Configure global logging based on configuration file
        log_level = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")
        self.logger.info("Logging configured.")

    # ------------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        # Initialize the retrieval orchestrator
        try:
            retriever = RetrievalOrchestrator()
            self.logger.info("Retriever ready.")
            return retriever
        except Exception as e:
            self.logger.exception(f"Failed to initialize retriever: {e}")
            raise

    # ------------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        # Initialize the LLM backend (Ollama) from configuration
        try:
            profile_name = self.cfg.get("generation", {}).get("llm", {}).get("profile", "default")
            llm = OllamaLLM(config_path="configs/llm.yaml", profile=profile_name)
            self.logger.info(f"LLM initialized (profile={profile_name}).")
            return llm
        except Exception as e:
            self.logger.exception(f"Failed to initialize LLM: {e}")
            raise

    # ------------------------------------------------------------------
    def run_interactive(self) -> None:
        # Interactive mode for continuous Prompt → Retrieval → LLM interaction
        self.logger.info("Ready for interactive mode. Press Ctrl+C to exit.")
        while True:
            try:
                query_obj = self.prompt_phase.get_prompt_object()
                if not query_obj:
                    continue
                answer = self.process_query(query_obj)
                if answer:
                    print("\n=== MODEL OUTPUT ===\n")
                    print(answer)
                    print("\n====================\n")
            except KeyboardInterrupt:
                self.logger.info("Session terminated by user.")
                break
            except Exception as e:
                self.logger.error(f"Unexpected runtime error: {e}")

    # ------------------------------------------------------------------
    def process_query(self, query_obj: Optional[Dict[str, Any]]) -> str:
        # End-to-end query execution: retrieval, reranking, prompt building, generation, logging
        if not query_obj or not query_obj.get("processed_query"):
            self.logger.warning("Invalid or empty query object.")
            return ""

        query = query_obj["processed_query"]
        intent = query_obj.get("intent", "conceptual")
        temporal_mode = intent in ["chronological", "temporal"]

        # Retrieval
        try:
            self.logger.info(f"Retrieving context (temporal_mode={temporal_mode}) for: '{query}'")
            retrieved_docs = self.retriever.retrieve(query, intent)
        except Exception as e:
            self.logger.exception(f"Retrieval failed: {e}")
            return ""

        if not retrieved_docs:
            self.logger.warning("No relevant documents retrieved.")
            return ""

        # Ensure evaluation-ready fields
        for i, ch in enumerate(retrieved_docs, start=1):
            ch.setdefault("rank", i)  # # deterministic rank
            if "text" not in ch:
                ch["text"] = ch.get("snippet", "")  # # ensure text field exists
            if not ch.get("id"):
                meta = ch.get("metadata", {}) or {}
                src = meta.get("source_file") or "unknown"
                year = meta.get("year", "na")
                h = abs(hash((ch.get("text") or "")[:120])) % (10**8)
                ch["id"] = f"{src}::{year}::{h}"  # # last-resort stable id

        # Prompt building
        try:
            full_prompt = self.prompt_builder.build_prompt(query, intent, retrieved_docs)
        except Exception as e:
            self.logger.exception(f"Prompt building failed: {e}")
            return ""

        # Generation + log
        try:
            output = self.llm.generate(full_prompt, context=retrieved_docs)
            qid = self._log_llm_run(query, retrieved_docs, output, full_prompt)
            self.logger.info(f"LLM generation successful. Run logged (query_id={qid}).")
            refs = self._format_references_grouped(retrieved_docs)
            return output.strip() + "\n\n" + refs
        except Exception as e:
            self.logger.exception(f"LLM generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    def run_with_context(self, query: str, intent: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        # Run generation phase with externally provided retrieval context
        self.logger.info(f"Running automatic contextual generation for intent='{intent}'")
        if not retrieved_chunks:
            return "No context provided."

        try:
            full_prompt = self.prompt_builder.build_prompt(query, intent, retrieved_chunks)
        except Exception as e:
            self.logger.error(f"Prompt construction failed: {e}")
            return f"Prompt build failed: {e}"

        try:
            output = self.llm.generate(full_prompt, context=retrieved_chunks)
            qid = self._log_llm_run(query, retrieved_chunks, output, full_prompt)
            self.logger.info(f"Contextual generation completed successfully. Run logged (query_id={qid}).")
            refs = self._format_references_grouped(retrieved_chunks)
            return output.strip() + "\n\n" + refs
        except Exception as e:
            self.logger.error(f"Contextual LLM generation failed: {e}")
            return f"[Error] Contextual generation failed: {e}"

    # ------------------------------------------------------------------
    def _log_llm_run(self, query: str, retrieved: List[Dict[str, Any]], output: str, prompt_text: str) -> str:
        # Persist a single run log consumed by evaluation (unified schema)
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        qid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in query)[:80] or "query"
        log_dir = Path("data/logs"); log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": ts,
            "query_id": qid,
            "query": query,
            "prompt": prompt_text,
            "retrieved_chunks": retrieved,   # # unified schema for evaluation
            "context_snippets": retrieved,   # # alias to support parsers relying on 'context_snippets'
            "model_output": output
        }
        (log_dir / f"llm_{ts}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return qid

    # ------------------------------------------------------------------
    def _format_references_grouped(self, results: List[Dict[str, Any]]) -> str:
        # Create IEEE-style reference list (one per unique PDF)
        if not results:
            return "References: none"

        grouped = {}
        for r in results:
            meta = r.get("metadata", {}) or {}
            src = meta.get("source_file") or "Unknown.pdf"
            year = meta.get("year") or r.get("year", "n/a")
            grouped[src] = year

        ordered = sorted(grouped.items(), key=lambda x: str(x[1]))
        refs = [f"[{i}] {src}, {year}" for i, (src, year) in enumerate(ordered, start=1)]
        return "References:\n" + "\n".join(refs)

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Gracefully close all components
        try:
            self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Retriever close failed: {e}")
        try:
            self.llm.close()
        except Exception as e:
            self.logger.warning(f"LLM close failed: {e}")
        self.logger.info("Pipeline shutdown complete.")


def main() -> None:
    # Standalone entry point for full interactive LLM orchestration
    orchestrator = LLMOrchestrator()
    orchestrator.run_interactive()
    orchestrator.close()


if __name__ == "__main__":
    main()
