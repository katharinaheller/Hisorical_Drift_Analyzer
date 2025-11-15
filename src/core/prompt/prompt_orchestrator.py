from __future__ import annotations
import logging
import re
from typing import Dict, Any, Optional

from src.core.prompt.query.query_input import QueryInput
from src.core.prompt.query.query_preprocessor import QueryPreprocessor
from src.core.prompt.query.prompt_builder import PromptBuilder


class PromptOrchestrator:
    """
    Coordinates the full prompt understanding pipeline:
    (1) user input → (2) preprocessing → (3) intent classification → (4) intent-guided query reformulation.
    Returns a refined query that aligns with the detected analytical intent.
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("PromptOrchestrator")
        if not self.logger.handlers:
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format="%(levelname)s | %(message)s"
            )

        self.query_input = QueryInput()
        self.preprocessor = QueryPreprocessor()
        self.prompt_builder = PromptBuilder()
        self.logger.info("PromptOrchestrator initialized successfully.")

    # ------------------------------------------------------------------
    def get_prompt_object(self) -> Dict[str, Any]:
        """
        Execute the complete prompt phase: read, clean, classify, reformulate.

        Returns
        -------
        dict
            {
                "refined_query": str,
                "intent": str
            }
        """
        try:
            raw_query: Optional[str] = self.query_input.read_interactive()
            if not raw_query or not raw_query.strip():
                self.logger.warning("No input provided. Skipping prompt phase.")
                return {}

            # Preprocess & classify
            result: Dict[str, Any] = self.preprocessor.process(raw_query)
            if not result or "processed_query" not in result:
                self.logger.warning("Preprocessing failed or returned empty result.")
                return {}

            processed_query = result["processed_query"]
            intent = result["intent"]

            # Intent-guided reformulation
            refined_query = self.prompt_builder.reformulate_query(processed_query, intent)

            # Cosmetic cleanup: remove duplicated “over time” patterns
            refined_query = re.sub(r"over time\?\s*over time", "over time", refined_query, flags=re.IGNORECASE)
            refined_query = re.sub(r"over time\s+over time", "over time", refined_query, flags=re.IGNORECASE)

            self.logger.info(f"Prompt phase complete → intent='{intent}', refined query='{refined_query}'")
            return {"refined_query": refined_query, "intent": intent}

        except KeyboardInterrupt:
            self.logger.info("Prompt input cancelled by user (Ctrl+C). Exiting interactive mode.")
            raise

        except EOFError:
            self.logger.info("Input stream closed (EOF). Terminating prompt phase.")
            raise KeyboardInterrupt

        except Exception as e:
            self.logger.exception(f"Unexpected error in prompt phase: {e}")
            return {}
