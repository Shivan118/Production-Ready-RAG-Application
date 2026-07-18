"""RAGAS metric computation.

Given samples of (question, answer, contexts, ground_truth), compute the
selected RAGAS metrics using our OpenAI LLM + embeddings (resolved with the
effective per-request key). Imports use fallbacks because RAGAS renames metric
classes across 0.2.x releases; scores are read back by each metric's own
`.name`, so we never hard-code the output column strings.
"""

import logfire

from app.generation.generator import _llm
from app.ingestion.embedder import get_embeddings

# --- RAGAS imports with cross-version fallbacks ---
from ragas import evaluate  # noqa: E402

try:  # dataset schema location moved between versions
    from ragas import EvaluationDataset
except ImportError:  # pragma: no cover
    from ragas.dataset_schema import EvaluationDataset

from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness  # noqa: E402

try:
    from ragas.metrics import ResponseRelevancy as _AnswerRelevancy
except ImportError:  # pragma: no cover
    from ragas.metrics import AnswerRelevancy as _AnswerRelevancy

try:
    from ragas.metrics import LLMContextPrecisionWithReference as _ContextPrecision
except ImportError:  # pragma: no cover
    from ragas.metrics import ContextPrecision as _ContextPrecision

try:
    from ragas.metrics import LLMContextRecall as _ContextRecall
except ImportError:  # pragma: no cover
    from ragas.metrics import ContextRecall as _ContextRecall


# our metric key -> RAGAS metric class
_METRIC_CLASSES = {
    "faithfulness": Faithfulness,
    "answer_relevancy": _AnswerRelevancy,
    "context_precision": _ContextPrecision,
    "context_recall": _ContextRecall,
}


def _to_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # RAGAS uses NaN for a metric it could not compute
    return None if f != f else round(f, 4)


def evaluate_samples(
    samples: list[dict], metric_keys: list[str]
) -> tuple[list[dict], dict]:
    """Return (per_sample_scores, aggregate_scores).

    per_sample_scores[i] maps metric_key -> score|None for samples[i].
    aggregate maps metric_key -> mean score over computable samples.
    """
    if not samples:
        return [], {m: None for m in metric_keys}

    metric_keys = [m for m in metric_keys if m in _METRIC_CLASSES]
    metric_instances = [_METRIC_CLASSES[m]() for m in metric_keys]

    with logfire.span(
        "ragas.evaluate", samples=len(samples), metrics=metric_keys
    ) as span:
        llm = LangchainLLMWrapper(_llm())
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s["question"],
                    response=s["answer"],
                    retrieved_contexts=s["contexts"] or [""],
                    reference=s["ground_truth"],
                )
                for s in samples
            ]
        )

        result = evaluate(
            dataset=dataset,
            metrics=metric_instances,
            llm=llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()

        # map our keys to the actual output columns via each metric's .name
        per_sample: list[dict] = [{} for _ in samples]
        aggregate: dict[str, float | None] = {}
        for key, metric in zip(metric_keys, metric_instances):
            col = metric.name if metric.name in df.columns else key
            if col not in df.columns:
                aggregate[key] = None
                for row in per_sample:
                    row[key] = None
                continue
            values = [_to_float(v) for v in df[col].tolist()]
            for row, v in zip(per_sample, values):
                row[key] = v
            valid = [v for v in values if v is not None]
            aggregate[key] = round(sum(valid) / len(valid), 4) if valid else None

        span.set_attribute("aggregate", aggregate)
        return per_sample, aggregate
