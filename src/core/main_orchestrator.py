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

# Import evaluation orchestrator via abstraction
try:
    from src.core.evaluation.evaluation_orchestrator import EvaluationOrchestrator
except Exception:
    EvaluationOrchestrator = None  # # Allow pipeline to run without evaluation module


class MainOrchestrator:
    """Central controller coordinating all pipeline phases."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Ensure consistent UTF-8 runtime
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["LANG"] = "C.UTF-8"

        # Reconfigure streams for UTF-8
        if hasattr(sys, "stdout"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys, "stderr"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        # Load merged configuration and setup logging
        self.cfg_loader = ConfigLoader(config_path)
        self.cfg: Dict[str, Any] = self.cfg_loader.config
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        """Initialize process-wide logger based on config."""
        opts = self.cfg.get("global", {})
        level = getattr(logging, opts.get("log_level", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
        logger = logging.getLogger("MainOrchestrator")
        logger.info("Initialized main orchestrator")
        return logger

    # ------------------------------------------------------------------
    def run_phase(self, phase: str) -> None:
        """Dispatch to the selected pipeline phase."""
        self.logger.info(f"Starting phase: {phase.upper()}")

        # Ensure project src in sys.path
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
                self.logger.info("Launching LLM phase — interactive mode active.")
                orchestrator = LLMOrchestrator()
                orchestrator.run_interactive()
                orchestrator.close()

            elif phase == "evaluation":
                self._run_evaluation()

            elif phase == "all":
                self.logger.info("Running full pipeline (ingestion → embedding → prompt → retrieval → llm)")
                run_ingestion()
                run_embedding()
                self._run_prompt_retrieval_chain()
                orchestrator = LLMOrchestrator()
                orchestrator.run_interactive()
                orchestrator.close()

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
        """Execute prompt → retrieval phase."""
        self.logger.info("Executing prompt → retrieval phase")

        # Prompt phase
        prompt_orch = PromptOrchestrator()
        prompt_data = prompt_orch.get_prompt_object()
        if not prompt_data or "processed_query" not in prompt_data:
            self.logger.warning("Prompt phase returned no valid query. Aborting retrieval.")
            return

        query = prompt_data["processed_query"]
        intent = prompt_data["intent"]

        # Retrieval phase
        retrieval = RetrievalOrchestrator(config_path="configs/retrieval.yaml")
        self.logger.info(f"Query intent='{intent}' → executing retrieval flow")
        retrieved: List[Dict[str, Any]] = retrieval.retrieve(
            query, temporal_mode=(intent == "chronological")
        )
        retrieval.close()

        if not retrieved:
            self.logger.warning("No results retrieved.")
            return

        # Output results summary
        print("\n" + "=" * 80)
        print(f"Retrieved Top-{len(retrieved)} Chunks (intent={intent})")
        for i, r in enumerate(retrieved, start=1):
            meta = r.get("metadata", {}) or {}
            year = meta.get("year", "n/a")
            title = meta.get("source_file") or meta.get("title") or "Unknown"
            score = r.get("final_score", r.get("score", 0.0))
            print(f"[{i}] ({year}) {title} | score={float(score):.3f}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    def _run_evaluation(self) -> None:
        """Batch evaluation over prior LLM logs using registered metrics."""
        if EvaluationOrchestrator is None:
            self.logger.error("Evaluation module not available. Please ensure src/core/evaluation exists.")
            return

        # Resolve evaluation config with sensible defaults
        eval_cfg = self.cfg.get("evaluation", {}) if isinstance(self.cfg, dict) else {}
        logs_dir = Path(eval_cfg.get("logs_dir", "data/logs")).resolve()
        out_dir = Path(eval_cfg.get("eval_logs_dir", "data/eval_logs")).resolve()
        k = int(eval_cfg.get("k", 5))
        glob_pat = eval_cfg.get("glob", "llm_*.json")
        gt_path = eval_cfg.get("ground_truth_path", "data/eval/ground_truth.json")

        # Create orchestrator and run batch evaluation
        self.logger.info(
            f"Evaluation settings → logs_dir={logs_dir} | out_dir={out_dir} | k={k} | glob={glob_pat}"
        )
        orch = EvaluationOrchestrator(
            output_dir=str(out_dir),
            k=k,
            ground_truth_path=gt_path
        )
        summary = orch.evaluate_batch_from_logs(logs_dir=str(logs_dir), pattern=glob_pat)

        # Human-readable console summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"files             : {summary.get('files', 0)}")
        print(f"mean NDCG@{k:>2}   : {summary.get('mean_ndcg@k', 0.0):.3f}")
        print(f"mean Faithfulness : {summary.get('mean_faithfulness', 0.0):.3f}")
        print("==========================\n")

    # ------------------------------------------------------------------
    def run(self, args: argparse.Namespace) -> None:
        """Entrypoint dispatcher using parsed CLI args."""
        if args.phase:
            self.run_phase(args.phase)
        else:
            self.logger.warning("No phase specified. Use --phase <name> or --phase all")


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
