from __future__ import annotations
from transformers import pipeline


class QueryIntentClassifier:
    """Transformer-based zero-shot intent classifier without any heuristics."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        candidate_labels: list[str] | None = None,
    ):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.labels = candidate_labels or ["chronological", "conceptual", "analytical"]

    def classify(self, query: str) -> str:
        """Infer intent purely via semantic entailment."""
        result = self.classifier(query, self.labels, multi_label=False)
        return result["labels"][0]
