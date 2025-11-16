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
from src.core.retrieval.orchestrator.retrieval_orchestrator import RetrievalOrchestrator
from src.core.llm.ollama_llm import OllamaLLM
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.prompt.query.prompt_builder import PromptBuilder


class LLMOrchestrator:
    """Coherent LLM orchestration pipeline with validated logging and deterministic retrieval."""

    def __init__(self, config_path: str = "configs/llm.yaml", logs_dir: str = "data/logs"):
        self.cfg = ConfigLoader(config_path).config

        self.logger = logging.getLogger("LLMOrchestrator")
        self._setup_logging()

        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_phase = PromptOrchestrator()

        self.retriever = self._init_retriever()

        citation_style = (
            self.cfg.get("generation", {})
            .get("citations", {})
            .get("style", "ieee")
            .lower()
        )
        self.prompt_builder = PromptBuilder(citation_style=citation_style)

        self.llm = self._init_llm()

        # Embedding model for factual consistency revision
        self._ff_model = SentenceTransformer(
            self.cfg.get("generation", {}).get(
                "faith_embed_model", "multi-qa-mpnet-base-dot-v1"
            )
        )

        self.logger.info(
            f"LLMOrchestrator initialized (citations={citation_style}, logs_dir={self.logs_dir})."
        )

    # --------------------------------------------------------------
    def _setup_logging(self) -> None:
        # Configure logging level from YAML
        lvl = self.cfg.get("global", {}).get("log_level", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, lvl, logging.INFO),
            format="%(levelname)s | %(message)s",
        )
        self.logger.info("Logging configured.")

    # --------------------------------------------------------------
    def _init_retriever(self) -> RetrievalOrchestrator:
        # Initialize retrieval subsystem
        r = RetrievalOrchestrator()
        self.logger.info("Retriever initialized.")
        return r

    # --------------------------------------------------------------
    def _init_llm(self) -> OllamaLLM:
        # Initialize LLM backend
        profile = (
            self.cfg.get("generation", {})
            .get("llm", {})
            .get("profile", "mistral_7b")
        )
        llm = OllamaLLM(config_path="configs/llm.yaml", profile=profile)
        self.logger.info(f"LLM backend ready (profile={profile}).")
        return llm

    # --------------------------------------------------------------
    def _split_into_sentences(self, text: str) -> list[str]:
        # Very simple sentence splitter
        import re
        s = re.split(r"(?<=[.!?])\s+", text.strip())
        return [t for t in s if t]

    # --------------------------------------------------------------
    def _fact_check_and_revise(
        self,
        answer: str,
        evidence_texts: List[str],
        min_sim: float = 0.22,
        drop_below: float = 0.15,
    ) -> str:
        # Embedding based factual consistency refinement
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
            "Revise the following sentences to align strictly with the evidence snippets. "
            "If support is insufficient write 'insufficient evidence'. Preserve numeric citations."
        )

        ev_join = "\n\n".join(f"[EVID{i+1}] {e}" for i, e in enumerate(evidence_texts[:8]))
        task = instruction + "\n\n" + ev_join + "\n\nSentences:\n" + "\n".join(f"- {s}" for s in revise)

        try:
            revised = self.llm.generate(task)
            return " ".join(keep + [revised])
        except Exception:
            return " ".join(keep) if keep else answer

    # --------------------------------------------------------------
    def _safe_atomic_write(self, path: Path, payload: Dict[str, Any]) -> None:
        # Atomic write of JSON files to prevent partial logs
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # --------------------------------------------------------------
    def _compose_full_prompt(
        self,
        refined_query: str,
        intent: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        # Assemble system prompt with context and citation constraints
        sys_p = self.prompt_builder.build_prompt(refined_query, intent)

        if intent == "chronological":
            def sy(meta: Dict[str, Any]) -> int:
                try:
                    return int(meta.get("year", 9999))
                except Exception:
                    return 9999

            retrieved_chunks = sorted(retrieved_chunks, key=lambda c: sy(c.get("metadata", {})))
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

        lines = []
        lines.append(sys_p.strip())
        lines.append("")
        lines.append(f"Refined query:\n{refined_query.strip()}\n")
        lines.append("Context snippets:")

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {}) or {}
            src = meta.get("source_file", "Unknown.pdf")
            year = meta.get("year", "n/a")
            ref_id = unique_sources.get(src, 0)

            header = f"[{ref_id}] {src} ({year})"
            text = chunk.get("text", "").strip() or "[Empty chunk]"

            lines.append(header)
            lines.append(text)
            lines.append("")

        lines.append(
            "Answer the refined query using only the context above. "
            "Use numeric citations. If a claim lacks evidence write 'insufficient evidence'."
        )
        lines.append("")
        lines.append("Reference index:")

        for rid, meta in ref_data.items():
            lines.append(f"[{rid}] {meta['source_file']} ({meta['year']})")

        lines.append("")
        lines.append("IMPORTANT OUTPUT REQUIREMENTS:")
        lines.append("Your final answer must end with a section titled 'References'.")
        lines.append("List all unique PDFs exactly once in the format:")
        lines.append("[n] FILENAME.pdf (YEAR)")
        lines.append("This section must be at the end of your output.")

        return "\n".join(lines)

    # --------------------------------------------------------------
    def _log_llm_input(self, query: str, intent: str, prompt: str, chunks: List[Dict[str, Any]]) -> None:
        # Log input with atomic write
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        path = self.logs_dir / f"llm_input_{ts}.json"

        payload = {
            "timestamp": ts,
            "query_refined": query,
            "intent": intent,
            "prompt_final_to_llm": prompt,
            "chunks_final_to_llm": chunks,
        }

        self._safe_atomic_write(path, payload)
        self.logger.info(f"Logged refined LLM input → {path}")

    # --------------------------------------------------------------
    def _log_llm_run(
        self,
        query: str,
        intent: str,
        retrieved: List[Dict[str, Any]],
        output: str,
        prompt: str,
    ) -> str:
        # Log completed LLM output
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        qid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in query)[:80] or "query"

        path = self.logs_dir / f"llm_{ts}.json"
        payload = {
            "timestamp": ts,
            "query_id": qid,
            "query": query,
            "query_refined": query,
            "intent": intent,
            "prompt_final_to_llm": prompt,
            "retrieved_chunks": retrieved,
            "model_output": output,
        }

        self._safe_atomic_write(path, payload)
        return qid

    # --------------------------------------------------------------
    def process_query(self, query_obj: Optional[Dict[str, Any]]) -> str:
        # Full pipeline execution
        if not query_obj:
            self.logger.warning("Empty query object received.")
            return ""

        query = query_obj.get("refined_query") or query_obj.get("processed_query")
        intent = query_obj.get("intent", "conceptual")

        if not query or not query.strip():
            self.logger.warning("Query text missing or empty.")
            return ""

        query = query.strip()

        self.logger.info(f"Starting pipeline | intent={intent} | query='{query}'")

        try:
            retrieved = self.retriever.retrieve(query, intent)
        except Exception as e:
            self.logger.exception(f"Retrieval failed: {e}")
            return ""

        try:
            prompt = self._compose_full_prompt(query, intent, retrieved)
        except Exception as e:
            self.logger.exception(f"Prompt construction failed: {e}")
            return ""

        self._log_llm_input(query, intent, prompt, retrieved)

        try:
            raw_ans = self.llm.generate(prompt.strip())
            evidence = [c.get("text", "") for c in retrieved if c.get("text")]
            final_ans = self._fact_check_and_revise(raw_ans, evidence)

            qid = self._log_llm_run(query, intent, retrieved, final_ans, prompt)
            self.logger.info(f"LLM generation successful (id={qid}).")

            return final_ans.strip()
        except Exception as e:
            self.logger.exception(f"LLM generation failed: {e}")
            return ""

    # --------------------------------------------------------------
    def close(self) -> None:
        # Safe close of all subcomponents
        self.logger.info("Closing LLMOrchestrator.")
        try:
            if hasattr(self.retriever, "close"):
                self.retriever.close()
        except Exception:
            pass

        try:
            if hasattr(self.llm, "close"):
                self.llm.close()
        except Exception:
            pass

        try:
            if hasattr(self.prompt_phase, "close"):
                self.prompt_phase.close()
        except Exception:
            pass

        self.logger.info("LLMOrchestrator closed.")
