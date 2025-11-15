# src/core/llm/llm_orchestrator.py
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.core.config.config_loader import ConfigLoader
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.prompt.query.prompt_builder import PromptBuilder


class LLMOrchestrator:
    """High-level orchestrator: query → retrieval → prompt assembly → LLM generation."""

    def __init__(self, config_path: str = "configs/llm.yaml", logs_dir: str = "data/logs"):
        # Load high-level configuration
        self.cfg = ConfigLoader(config_path).config

        # Logger
        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()

        # Dynamic log directory passed from benchmark
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Prompt pipeline
        self.prompt_phase = PromptOrchestrator()

        # Retriever
        self.retriever = self._init_retriever()

        # Citation style for prompt builder (falls back to IEEE if missing)
        citation_style = (
            self.cfg.get("generation", {})
            .get("citations", {})
            .get("style", "ieee")
            .lower()
        )
        self.prompt_builder = PromptBuilder(citation_style=citation_style)

        # LLM backend (model + sampling read from YAML)
        self.llm = self._init_llm()

        # Embedding model for factual consistency scoring
        self._ff_model = SentenceTransformer(
            self.cfg.get("generation", {}).get(
                "faith_embed_model", "multi-qa-mpnet-base-dot-v1"
            )
        )

        self.logger.info(
            f"LLMOrchestrator initialized successfully "
            f"(citations={citation_style}, logs_dir={self.logs_dir})."
        )

    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        # Configure logging from YAML
        log_level = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(levelname)s | %(message)s",
        )
        self.logger.info("Logging configured.")

    # ------------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        # Initialize retrieval orchestrator
        retriever = RetrievalOrchestrator()
        self.logger.info("Retriever initialized.")
        return retriever

    # ------------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        # Configure LLM backend from YAML profile
        profile_name = (
            self.cfg.get("generation", {})
            .get("llm", {})
            .get("profile", "mistral_7b")
        )
        llm = OllamaLLM(config_path="configs/llm.yaml", profile=profile_name)
        self.logger.info(f"LLM backend ready (profile='{profile_name}').")
        return llm

    # ------------------------------------------------------------------
    def _split_into_sentences(self, text: str) -> list[str]:
        # Simple sentence splitter
        import re
        s = re.split(r"(?<=[.!?])\s+", text.strip())
        return [t for t in s if t]

    # ------------------------------------------------------------------
    def _fact_check_and_revise(
        self,
        answer: str,
        evidence_texts: list[str],
        min_sim: float = 0.22,
        drop_below: float = 0.15,
    ) -> str:
        # Factual consistency filtering based on embedding similarity
        if not answer or not evidence_texts:
            return answer

        sents = self._split_into_sentences(answer)
        if not sents:
            return answer

        ans_emb = self._ff_model.encode(
            sents, normalize_embeddings=True, convert_to_tensor=True
        )
        ev_emb = self._ff_model.encode(
            evidence_texts, normalize_embeddings=True, convert_to_tensor=True
        )
        sims = util.cos_sim(ans_emb, ev_emb).cpu().numpy()

        keep, revise = [], []
        for i, sent in enumerate(sents):
            max_sim = float(np.max(sims[i])) if sims.shape[1] > 0 else 0.0
            if max_sim >= min_sim:
                keep.append(sent)
            elif max_sim >= drop_below:
                revise.append(sent)

        if not revise:
            return " ".join(keep) if keep else answer

        instruction = (
            "Revise the following sentences to align strictly with the provided evidence snippets. "
            "If support is insufficient, replace with: 'insufficient evidence'. Keep numeric citations [n]."
        )
        ev_join = "\n\n".join(f"[EVID{i+1}] {e}" for i, e in enumerate(evidence_texts[:8]))

        q = (
            instruction
            + "\n\n"
            + ev_join
            + "\n\nSentences:\n"
            + "\n".join(f"- {s}" for s in revise)
        )

        try:
            revised_block = self.llm.generate(q)
            return " ".join(keep + [revised_block])
        except Exception:
            return " ".join(keep) if keep else answer

    # ------------------------------------------------------------------
    def process_query(self, query_obj: Optional[Dict[str, Any]]) -> str:
        # Main pipeline execution
        if not query_obj:
            self.logger.warning("Empty query object received.")
            return ""

        query = query_obj.get("refined_query") or query_obj.get("processed_query")
        intent = query_obj.get("intent", "conceptual")

        if not query or not query.strip():
            self.logger.warning("Query text missing or empty; skipping request.")
            return ""

        query = query.strip()
        self.logger.info(
            f"Starting full LLM pipeline | intent={intent} | query='{query}'"
        )

        # Retrieval phase
        try:
            retrieved_docs = self.retriever.retrieve(query, intent)
        except Exception as e:
            self.logger.exception(f"Retrieval failed: {e}")
            return ""

        if not retrieved_docs:
            self.logger.warning("No documents retrieved.")
            retrieved_docs = []

        # Prompt assembly
        try:
            final_prompt = self._compose_full_prompt(query, intent, retrieved_docs)
        except Exception as e:
            self.logger.exception(f"Prompt construction failed: {e}")
            return ""

        llm_input = {
            "system_prompt": final_prompt,
            "query_refined": query,
            "intent": intent,
            "context_chunks": retrieved_docs,
        }
        self._log_llm_input(llm_input)

        # LLM inference
        try:
            ans = self.llm.generate(final_prompt.strip())
            evidence_texts = [c.get("text", "") for c in retrieved_docs if c.get("text")]
            final_answer = self._fact_check_and_revise(ans, evidence_texts)

            qid = self._log_llm_run(
                query, intent, retrieved_docs, final_answer, final_prompt
            )
            self.logger.info(f"LLM generation successful (query_id={qid}).")
            return final_answer.strip()
        except Exception as e:
            self.logger.exception(f"LLM generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    def _compose_full_prompt(
        self,
        refined_query: str,
        intent: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        # Assemble full system prompt with citations and constraints
        system_prompt = self.prompt_builder.build_prompt(refined_query, intent)

        # Chronological ordering if required by intent
        if intent == "chronological":

            def safe_year(meta: Dict[str, Any]) -> int:
                y = meta.get("year")
                try:
                    return int(y)
                except Exception:
                    return 9999

            retrieved_chunks = sorted(
                retrieved_chunks,
                key=lambda c: safe_year(c.get("metadata", {})),
            )
            self.logger.info("Chunks sorted chronologically.")

        unique_sources: Dict[str, int] = {}
        ref_data: Dict[int, Dict[str, Any]] = {}

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file", "Unknown.pdf")
            year = meta.get("year", "n/a")
            if src not in unique_sources:
                ref_id = len(unique_sources) + 1
                unique_sources[src] = ref_id
                ref_data[ref_id] = {"source_file": src, "year": year}

        lines: List[str] = []
        lines.append(system_prompt.strip())
        lines.append("")
        lines.append(f"Refined query:\n{refined_query.strip()}\n")
        lines.append(
            "You are given the following context snippets from historical AI-related documents."
        )
        lines.append("")
        lines.append("Context snippets:")

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file", "Unknown.pdf")
            year = meta.get("year", "n/a")
            ref_id = unique_sources.get(src, 0)
            header = f"[{ref_id}] {src} ({year})"
            text = chunk.get("text", "").strip()
            lines.append(header)
            lines.append(text if text else "[Empty chunk]")
            lines.append("")

        lines.append(
            "Now answer the refined query using ONLY the context above. "
            "Use IEEE-style numeric citations [n]. If a claim lacks sufficient evidence, "
            "write 'insufficient evidence'."
        )
        lines.append("")
        lines.append("Reference index:")
        for ref_id, meta in ref_data.items():
            lines.append(f"[{ref_id}] {meta['source_file']} ({meta['year']})")

        lines.append("")
        lines.append("IMPORTANT OUTPUT REQUIREMENTS:")
        lines.append("Your final answer MUST end with a separate section titled 'References'.")
        lines.append(
            "This final section MUST list all unique PDFs exactly once in the format:"
        )
        lines.append("[n] FILENAME.pdf (YEAR)")
        lines.append("Do NOT omit this section. Do NOT invent filenames or years.")
        lines.append("The 'References' section MUST appear at the very end of your output.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _log_llm_input(self, llm_input: Dict[str, Any]) -> Path:
        # Save refined prompt and retrieved chunks
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_path = self.logs_dir / f"llm_input_{ts}.json"

        payload = {
            "timestamp": ts,
            "query_refined": llm_input["query_refined"],
            "intent": llm_input["intent"],
            "prompt_final_to_llm": llm_input["system_prompt"],
            "chunks_final_to_llm": llm_input["context_chunks"],
        }

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info(f"Logged refined LLM input → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    def _log_llm_run(
        self,
        query: str,
        intent: str,
        retrieved: List[Dict[str, Any]],
        output: str,
        final_prompt: str,
    ) -> str:
        # Save final LLM output
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        qid = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in query
        )[:80] or "query"

        out_path = self.logs_dir / f"llm_{ts}.json"
        payload = {
            "timestamp": ts,
            "query_id": qid,
            "query": query,
            "query_refined": query,
            "intent": intent,
            "prompt_final_to_llm": final_prompt,
            "retrieved_chunks": retrieved,
            "model_output": output,
        }

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return qid

    # ------------------------------------------------------------------
    def run_interactive(self) -> None:
        # Interactive shell mode
        self.logger.info("Starting interactive LLM phase...")
        try:
            query = input("Enter your query (leave empty to use default): ").strip()
        except EOFError:
            query = ""

        if not query:
            query = "How has the term 'AI' evolved over time?"
            self.logger.info(f"No input provided. Using default query: '{query}'")

        query_obj = {
            "refined_query": query,
            "intent": "chronological" if "evolve" in query.lower() else "conceptual",
        }

        self.logger.info(f"Processing query in interactive mode | query='{query}'")
        result = self.process_query(query_obj)

        print("\n=== LLM OUTPUT ===\n")
        print(result if result else "[No response generated]")
        self.logger.info("Interactive LLM phase completed successfully.")

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Close all subcomponents safely
        self.logger.info("Closing LLMOrchestrator and freeing resources.")
        try:
            if hasattr(self.retriever, "close"):
                self.retriever.close()
        except Exception as e:
            self.logger.warning(f"Failed to close retriever: {e}")

        try:
            if hasattr(self.llm, "close"):
                self.llm.close()
        except Exception as e:
            self.logger.warning(f"Failed to close LLM backend: {e}")

        try:
            if hasattr(self.prompt_phase, "close"):
                self.prompt_phase.close()
        except Exception:
            pass

        self.logger.info("LLMOrchestrator closed successfully.")
