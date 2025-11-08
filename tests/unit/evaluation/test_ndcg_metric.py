import math
from src.core.evaluation.metrics.ndcg_metric import NDCGMetric

def test_ndcg_perfect_ranking():
    # Perfect ranking → NDCG = 1.0
    metric = NDCGMetric(k=5)
    scores = [3, 2, 1, 0, 0]
    result = metric.compute(relevance_scores=scores)
    assert math.isclose(result, 1.0, rel_tol=1e-9)

def test_ndcg_partial_reversal():
    # Slightly worse order → lower NDCG
    metric = NDCGMetric(k=5)
    scores = [1, 3, 2, 0, 0]
    result = metric.compute(relevance_scores=scores)
    assert 0.7 < result < 1.0

def test_ndcg_zero_relevance():
    metric = NDCGMetric(k=3)
    result = metric.compute(relevance_scores=[0, 0, 0])
    assert result == 0.0

def test_ndcg_empty_input():
    metric = NDCGMetric(k=3)
    result = metric.compute(relevance_scores=[])
    assert result == 0.0

def test_ndcg_description():
    metric = NDCGMetric(k=3)
    desc = metric.describe()
    assert desc["name"] == "NDCG@k"
    assert "ranking" in desc["description"].lower()
