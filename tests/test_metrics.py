import os
import sys
import math
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)
os.environ["DATA_PATH"] = os.path.join(repo_root, "data", "products.csv")
from backend.app import compute_metrics, compute_ndcg


def test_compute_metrics_threshold():
    ids = [1, 2, 3]
    scores = [0.7, 0.2, 0.8]
    metrics = compute_metrics("query", ids, scores)

    expected_flags = [1, 0, 1]
    expected_precision = sum(expected_flags) / len(expected_flags)
    expected_ndcg = compute_ndcg(expected_flags)

    assert metrics.precision == pytest.approx(expected_precision)
    assert metrics.recall == pytest.approx(expected_precision)
    assert metrics.ndcg == pytest.approx(expected_ndcg)


def test_compute_metrics_empty():
    metrics = compute_metrics("query", [], [])
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.ndcg == 0.0
