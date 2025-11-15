# src/core/llm/ollama_llm.py
from __future__ import annotations
import subprocess
import logging
from typing import Any
from src.core.llm.interfaces.i_llm import ILLM
from src.core.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class OllamaLLM(ILLM):
    """
    Lightweight local Ollama backend.

    - Model selection is controlled via `profiles[profile].model`.
    - Sampling hyperparameters (temperature, max_tokens) are shared globally
      via the `sampling` section in configs/llm.yaml for fair comparisons.
    """

    def __init__(self, config_path: str = "configs/llm.yaml", profile: str | None = None):
        # Load full configuration
        cfg = ConfigLoader(config_path).config
        global_cfg = cfg.get("global", {})
        profiles = cfg.get("profiles", {})
        sampling_cfg = cfg.get("sampling", {})

        # Resolve active profile for this instance
        self.profile = profile or "mistral_7b"
        profile_cfg = profiles.get(self.profile, {})

        # Load model configuration from profile
        self.model = profile_cfg.get("model", "mistral:7b-instruct")
        self.auto_pull = bool(profile_cfg.get("auto_pull", True))

        # Shared sampling configuration for all models
        self.temperature = float(sampling_cfg.get("temperature", 0.1))
        self.max_tokens = int(sampling_cfg.get("max_tokens", 2048))

        # Configure logging
        log_level = global_cfg.get("log_level", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(levelname)s | %(message)s",
        )

        logger.info(
            f"OllamaLLM ready (profile={self.profile}, model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens})"
        )
        self._ensure_model_available()

    # ------------------------------------------------------------------
    def _ensure_model_available(self) -> None:
        # Ensure the configured Ollama model is available locally
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            if self.model.lower() not in result.stdout.lower():
                if not self.auto_pull:
                    logger.warning(
                        f"Model '{self.model}' missing (auto_pull disabled)."
                    )
                    return
                logger.info(f"Pulling model '{self.model}' ...")
                pull_result = subprocess.run(
                    ["ollama", "pull", self.model],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if pull_result.returncode != 0:
                    raise RuntimeError(pull_result.stderr.strip())
                logger.info(f"Successfully pulled '{self.model}'.")
            else:
                logger.info(f"Model '{self.model}' available locally.")
        except Exception as e:
            logger.error(f"Model check failed: {e}")

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        Run the Ollama model and return its output.

        Expectation:
        - `prompt` is the complete composed prompt
          (instruction + query + all context chunks).
        - Sampling hyperparameters are configured globally in YAML.
        """
        try:
            # Note: the basic CLI form does not expose all sampling options.
            # Here we rely on Ollama's defaults plus model-level configuration.
            cmd = ["ollama", "run", self.model, prompt]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            output = result.stdout.strip()
            logger.info(
                f"Ollama generation successful | model={self.model} | len={len(output)}"
            )
            return output

        except Exception as e:
            logger.exception(f"Ollama model run failed: {e}")
            raise

    # ------------------------------------------------------------------
    def close(self) -> None:
        """No persistent connections or open streams."""
        logger.info("OllamaLLM closed cleanly.")
        return
