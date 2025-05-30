import os
import sys
import importlib
from pathlib import Path

import pytest

# Helper fixture to load backend.app with correct DATA_PATH
@pytest.fixture
def backend_app():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    os.environ['DATA_PATH'] = str(repo_root / 'data' / 'products.csv')
    import backend.app as app
    return importlib.reload(app)

@pytest.mark.parametrize('query,expected', [
    ('widget', [1, 2]),
    ('gadget', [3]),
    ('gizmo', [4]),
])
def test_vector_search_returns_expected_ids(backend_app, query, expected):
    results = backend_app.vector_search(query)
    ids = [r['product_id'] for r in results]
    assert ids[: len(expected)] == expected

def test_evaluate_and_metrics_with_mock(monkeypatch, backend_app):
    # Mapping from product name to mocked score
    scores_by_name = {
        'Widget A': 1.0,   # relevant
        'Widget B': 0.4,
        'Gadget C': 0.6,  # relevant
        'Gizmo D': 0.0,
    }

    def fake_call_llm(query: str, text: str) -> float:
        for name, score in scores_by_name.items():
            if name in text:
                return score
        return 0.0

    monkeypatch.setattr(backend_app, 'call_llm', fake_call_llm)

    result_ids = [1, 2, 3]
    scores = backend_app.evaluate_with_llm('test', result_ids)
    assert scores == [1.0, 0.4, 0.6]

    metrics = backend_app.compute_metrics('test', result_ids, scores)
    assert metrics.precision == pytest.approx(2 / 3)
    # Recall is approximated using only the retrieved items
    assert metrics.recall == pytest.approx(metrics.precision)
    expected_ndcg = backend_app.compute_ndcg([1, 0, 1])
    assert metrics.ndcg == pytest.approx(expected_ndcg)
