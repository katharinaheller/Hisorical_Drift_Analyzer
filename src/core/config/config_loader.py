from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    Universal YAML configuration loader for all pipeline phases.
    Expands ${PROJECT_ROOT}, ${BASE_DIR}, ${base_dir}, etc.
    and gracefully handles missing domain-specific sections.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

        # Load YAML file
        with open(self.path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        if not isinstance(self._raw, dict):
            raise ValueError("Configuration file must contain a top-level mapping")

        # Detect project root (the directory above 'configs')
        self.project_root = self._detect_project_root()

        # Determine base_dir, defaulting to project root
        global_section = self._raw.get("global", {})
        base_dir_value = global_section.get("base_dir", "${PROJECT_ROOT}")
        self.base_dir = self._expand_single_var(base_dir_value)

        # Expand placeholders recursively in all sections
        self.config = self._expand_vars(self._raw)

        # ------------------------------------------------------------------
        # Graceful handling for domain-specific sections (e.g. chunking)
        # Instead of raising errors, provide safe defaults.
        # ------------------------------------------------------------------
        if "chunking" not in self.config:
            self.config["chunking"] = {}

        if "options" not in self.config:
            self.config["options"] = {}

        if "paths" not in self.config:
            self.config["paths"] = {}

        # Expand again after adding defaults
        self.config = self._expand_vars(self.config)

    # ------------------------------------------------------------------
    def _detect_project_root(self) -> Path:
        """Return the root directory of the project (one level above configs)."""
        p = self.path.resolve()
        if "configs" in p.parts:
            idx = p.parts.index("configs")
            return Path(*p.parts[:idx])
        # fallback: use parent directory
        return p.parent

    # ------------------------------------------------------------------
    def _expand_single_var(self, value: Any) -> Any:
        """Replace placeholders only when ${...} is explicitly present."""
        if not isinstance(value, str) or "${" not in value:
            return value  # unchanged non-string or no placeholder

        replacements = {
            "${PROJECT_ROOT}": str(self.project_root),
            "${project_root}": str(self.project_root),
            "${BASE_DIR}": str(self.project_root),
            "${base_dir}": str(self.project_root),
        }

        for placeholder, real in replacements.items():
            value = value.replace(placeholder, real)
        return str(Path(value).resolve())

    # ------------------------------------------------------------------
    def _expand_vars(self, data: Any) -> Any:
        """Recursively expand placeholders in nested dicts/lists."""
        if isinstance(data, dict):
            return {k: self._expand_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_vars(v) for v in data]
        elif isinstance(data, str):
            return self._expand_single_var(data)
        else:
            return data

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Access top-level config sections safely."""
        return self.config.get(key, default)

    # ------------------------------------------------------------------
    @property
    def raw(self) -> Dict[str, Any]:
        """Return unexpanded raw YAML structure."""
        return self._raw


# ----------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Convenience function returning a parsed, expanded config dictionary."""
    return ConfigLoader(path).config
