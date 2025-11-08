from src.core.evaluation.metrics.faithfulness_metric import FaithfulnessMetric
import numpy as np

def test_faithfulness_empty_input_returns_zero():
    metric = FaithfulnessMetric()
    result = metric.compute(context_chunks=[], answer="")
    assert result == 0.0

def test_faithfulness_identical_texts_high_score():
    metric = FaithfulnessMetric()
    text = "Artificial intelligence enables new retrieval-augmented generation systems."
    score = metric.compute(context_chunks=[text], answer=text)
    assert 0.9 <= score <= 1.0  # identical sentences → near-perfect similarity

def test_faithfulness_different_texts_lower_score():
    metric = FaithfulnessMetric()
    ctx = ["Quantum computing uses qubits and superposition."]
    ans = "Neural networks require large datasets for training."
    score = metric.compute(context_chunks=ctx, answer=ans)
    assert 0.0 <= score < 0.8

def test_faithfulness_multiple_chunks_mean_similarity():
    metric = FaithfulnessMetric()
    ctx = [
        "The retrieval system fetches relevant context.",
        "Large language models generate text from context."
    ]
    ans = "The model retrieves context and generates an answer."
    score = metric.compute(context_chunks=ctx, answer=ans)
    assert 0.5 <= score <= 1.0

def test_faithfulness_description():
    metric = FaithfulnessMetric()
    desc = metric.describe()
    assert desc["name"] == "Faithfulness"
    assert "similarity" in desc["description"].lower()
