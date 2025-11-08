from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class IMetric(ABC):
    """Interface defining a unified metric contract for all evaluators."""

    @abstractmethod
    def compute(self, **kwargs: Any) -> float:
        """Compute the metric based on explicit input parameters."""
        pass

    @abstractmethod
    def describe(self) -> Dict[str, str]:
        """Return metadata about the metric (name, type, short description)."""
        pass
