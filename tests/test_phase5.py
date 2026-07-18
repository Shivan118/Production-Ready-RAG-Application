"""Phase 5 unit tests — golden dataset loader, eval schemas, RAGAS wrapper
helpers. Offline: no RAGAS LLM calls (evaluate_samples is only exercised on the
empty-input short-circuit)."""

import math
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-0000000000")

from app.evaluation import dataset  # noqa: E402
from app.models.schemas import (  # noqa: E402
    EvalMetric,
    EvalRequest,
    RetrievalStrategy,
)


class TestGoldenDataset:
    def test_size_matches_items(self):
        assert dataset.size() == len(dataset.golden_items())
        assert dataset.size() >= 10

    def test_items_have_required_fields(self):
        for item in dataset.golden_items():
            assert item.question.strip()
            assert item.ground_truth.strip()

    def test_limit(self):
        assert len(dataset.golden_items(limit=3)) == 3

    def test_description_present(self):
        assert dataset.description()


class TestEvalSchemas:
    def test_request_defaults(self):
        req = EvalRequest()
        assert RetrievalStrategy.HYBRID in req.strategies
        assert RetrievalStrategy.ADVANCED in req.strategies
        assert set(req.metrics) == set(EvalMetric)
        assert req.use_rerank is True

    def test_metric_values(self):
        assert {m.value for m in EvalMetric} == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }

    def test_num_questions_bounds(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalRequest(num_questions=0)


class TestRagasWrapper:
    def test_metric_classes_cover_all_metrics(self):
        from app.evaluation import ragas_eval

        assert set(ragas_eval._METRIC_CLASSES) == {m.value for m in EvalMetric}

    def test_to_float(self):
        from app.evaluation.ragas_eval import _to_float

        assert _to_float(0.5) == 0.5
        assert _to_float("not a number") is None
        assert _to_float(math.nan) is None

    def test_empty_samples_short_circuit(self):
        from app.evaluation.ragas_eval import evaluate_samples

        per_sample, aggregate = evaluate_samples([], ["faithfulness"])
        assert per_sample == []
        assert aggregate == {"faithfulness": None}
