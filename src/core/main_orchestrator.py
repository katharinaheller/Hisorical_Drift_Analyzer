# src/core/main_orchestrator.py
from __future__ import annotations
import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.config.config_loader import ConfigLoader
from src.core.ingestion.ingestion_orchestrator import main as run_ingestion
from src.core.embedding.embedding_orchestrator import main as run_embedding
from src.core.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.core.prompt.prompt_orchestrator import PromptOrchestrator
from src.core.llm.llm_orchestrator import LLMOrchestrator

# Optional import: evaluator module for KPIs (NDCG@k + Faithfulness)
try:
    from src.core.evaluation.evaluator import Evaluator
except Exception:
    Evaluator = None  # # Gracefully handle missing evaluator module


class MainOrchestrator:
    """Central controller coordinating all pipeline phases of the Historical Drift Analyzer."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Ensure UTF-8 runtime consistency
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["LANG"] = "C.UTF-8"

        # Reconfigure stdout/stderr for UTF-8
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        # Load global configuration and setup logging
        self.cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = self.cfg_loader.config
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        """Initialize global logger."""
        opts = self.cfg.get("global", {})
        level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
        logger = logging.getLogger("MainOrchestrator")
        logger.info("Initialized main orchestrator")
        return logger

    # ------------------------------------------------------------------
    def run_phase(self, phase: str) -> None:
        """Dispatch the orchestrator to the specified phase."""
        self.logger.info(f"Starting phase: {phase.upper()}")

        base_dir = Path(self.cfg["global"]["base_dir"]).resolve()
        sys.path.append(str(base_dir / "src"))

        try:
            if phase == "ingestion":
                run_ingestion()

            elif phase == "embedding":
                run_embedding()

            elif phase == "retrieval":
                self._run_prompt_retrieval_chain()

            elif phase == "llm":
                self._run_llm_interactive()

            elif phase == "evaluation":
                self._run_evaluation()

            elif phase == "all":
                self.logger.info("Running full pipeline (ingestion → embedding → interactive LLM phase)")
                run_ingestion()
                run_embedding()
                self._run_llm_interactive()

            else:
                self.logger.error(f"Unknown phase: {phase}")
                sys.exit(1)

            self.logger.info(f"Phase '{phase}' completed successfully.")

        except UnicodeDecodeError as ue:
            self.logger.error(f"Unicode decoding failed: {ue}. Retrying with UTF-8 replacement.")
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
            raise

        except KeyboardInterrupt:
            self.logger.info("Execution manually interrupted by user.")
            sys.exit(0)

        except Exception as e:
            self.logger.exception(f"Phase '{phase}' failed: {e}")
            raise

    # ------------------------------------------------------------------
    def _run_prompt_retrieval_chain(self) -> None:
        """Execute prompt → retrieval → automatic LLM generation (single-shot mode)."""
        self.logger.info("Executing prompt → retrieval → LLM phase")

        prompt_orch = PromptOrchestrator()
        prompt_data = prompt_orch.get_prompt_object()
        if not prompt_data or "processed_query" not in prompt_data:
            self.logger.warning("Prompt phase returned no valid query. Aborting retrieval.")
            return

        query = prompt_data["processed_query"]
        intent = prompt_data["intent"]

        retrieval = RetrievalOrchestrator(config_path="configs/retrieval.yaml")
        self.logger.info(f"Query intent='{intent}' → executing retrieval flow")
        retrieved: List[Dict[str, Any]] = retrieval.retrieve(query, intent)
        retrieval.close()

        if not retrieved:
            self.logger.warning("No documents retrieved.")
            return

        print("\n" + "=" * 80)
        print(f"Retrieved Top-{len(retrieved)} Chunks (intent={intent})")
        for i, r in enumerate(retrieved, start=1):
            meta = r.get("metadata", {}) or {}
            year = meta.get("year", "n/a")
            src = meta.get("source_file") or meta.get("title") or "Unknown"
            score = r.get("final_score", r.get("score", 0.0))
            rel = r.get("relevance", None)
            rel_str = f" | rel={rel}" if rel is not None else ""
            print(f"[{i}] ({year}) {src} | score={float(score):.3f}{rel_str}")
        print("=" * 80 + "\n")

        try:
            self.logger.info("Launching Ollama model for contextual generation...")
            llm_orch = LLMOrchestrator()
            output = llm_orch.run_with_context(query, intent, retrieved)
            print("\n=== MODEL OUTPUT ===\n")
            print(output)
            print("\n====================\n")

            # Inline evaluation if evaluator is available
            if Evaluator is not None:
                self.logger.info("Running inline evaluation (NDCG@k + Faithfulness)...")
                evaluator = Evaluator(output_dir="data/eval_logs", k=int(self.cfg.get("evaluation", {}).get("k", 5)))
                metrics = evaluator.evaluate(
                    query_id=self._safe_query_id(query),
                    retrieved_chunks=retrieved,
                    answer=output or ""
                )
                self.logger.info(f"Inline Eval → NDCG@k={metrics.get('ndcg@k', 0.0):.3f} | Faithfulness={metrics.get('faithfulness', 0.0):.3f}")
            else:
                self.logger.warning("Evaluator module not available; skipping inline evaluation.")

            llm_orch.close()
        except Exception as e:
            self.logger.error(f"Automatic LLM generation failed: {e}")

    # ------------------------------------------------------------------
    def _run_llm_interactive(self) -> None:
        """Start an interactive prompt → retrieval → LLM loop until Ctrl+C."""
        self.logger.info("Starting interactive LLM session. Press Ctrl+C to exit.")
        llm_orch = LLMOrchestrator()
        try:
            llm_orch.run_interactive()
        except KeyboardInterrupt:
            self.logger.info("Interactive session terminated by user.")
        finally:
            llm_orch.close()

    # ------------------------------------------------------------------
    def _run_evaluation(self) -> None:
        """Batch evaluation over existing logs (end-to-end and retrieval KPIs)."""
        if Evaluator is None:
            self.logger.error("Evaluator module not available. Please add src/core/evaluation/evaluator.py.")
            return

        # Resolve directories from config with sensible defaults
        paths_cfg = self.cfg.get("paths", {})
        logs_dir = Path(paths_cfg.get("logs_dir", "data/logs")).resolve()
        eval_out = Path(paths_cfg.get("eval_logs_dir", "data/eval_logs")).resolve()
        eval_out.mkdir(parents=True, exist_ok=True)

        k_default = int(self.cfg.get("evaluation", {}).get("k", 5))
        evaluator = Evaluator(output_dir=str(eval_out), k=k_default)

        # Iterate over LLM run logs (assumed to contain retrieved_docs + model_output)
        pattern = self.cfg.get("evaluation", {}).get("glob", "llm_*.json")
        files = sorted(logs_dir.glob(pattern))
        if not files:
            self.logger.warning(f"No evaluation logs found under {logs_dir} matching '{pattern}'.")
            return

        # Aggregation buffers
        ndcgs: List[float] = []
        faiths: List[float] = []

        self.logger.info(f"Evaluating {len(files)} run(s) from {logs_dir} ...")
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                qid = data.get("query_id") or self._safe_query_id(data.get("query", fp.stem))
                retrieved = data.get("retrieved_docs") or data.get("retrieved_chunks") or []
                answer = data.get("model_output") or data.get("answer") or ""

                # Run evaluation for this query
                metrics = evaluator.evaluate(query_id=qid, retrieved_chunks=retrieved, answer=answer)
                ndcgs.append(float(metrics.get("ndcg@k", 0.0)))
                faiths.append(float(metrics.get("faithfulness", 0.0)))

            except Exception as e:
                self.logger.warning(f"Failed to evaluate {fp.name}: {e}")

        # Print summary
        def mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        self.logger.info(f"Evaluation Summary → files={len(files)} | k={k_default}")
        self.logger.info(f"Mean NDCG@k = {mean(ndcgs):.3f} | Mean Faithfulness = {mean(faiths):.3f}")
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Files evaluated : {len(files)}")
        print(f"k (NDCG@k)     : {k_default}")
        print(f"Mean NDCG@k    : {mean(ndcgs):.3f}")
        print(f"Mean Faithfulness: {mean(faiths):.3f}")
        print("==========================\n")

    # ------------------------------------------------------------------
    def _safe_query_id(self, query: str) -> str:
        """Generate a filesystem-safe query id."""
        if not query:
            return "unknown_query"
        q = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in query.strip())
        return q[:80] or "query"


    # ------------------------------------------------------------------
    def run(self, args: argparse.Namespace) -> None:
        """Entrypoint for orchestrator execution."""
        if args.phase:
            self.run_phase(args.phase)
        else:
            self.logger.warning("No phase specified. Use --phase <name> or --phase all.")


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Historical Drift Analyzer – Main Orchestrator")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["ingestion", "embedding", "retrieval", "llm", "evaluation", "all"],
        help="Select which phase of the pipeline to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the master configuration YAML file",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    orchestrator = MainOrchestrator(config_path=args.config)
    orchestrator.run(args)
