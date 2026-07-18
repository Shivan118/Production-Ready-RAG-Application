"""Evaluation runner — generate answers per strategy over the golden dataset,
then score them with RAGAS.

For each (strategy, golden question) we run the real RAG pipeline (guardrails
off, so PII redaction/blocking never alters an answer under test), collect the
answer and the retrieved contexts, and hand the batch to `ragas_eval`.
"""

import time

import logfire

from app.evaluation import dataset
from app.models.schemas import (
    EvalMetric,
    EvalRequest,
    EvalResponse,
    PerQuestionResult,
    QueryRequest,
    RetrievalStrategy,
    StrategyEvalResult,
)
from app.pipeline import run_query


def _run_strategy(
    strategy: RetrievalStrategy,
    items,
    metrics: list[str],
    top_k: int,
    use_rerank: bool,
) -> StrategyEvalResult:
    from app.evaluation import ragas_eval

    samples: list[dict] = []
    latencies: list[float] = []

    with logfire.span("eval.strategy", strategy=strategy.value, n=len(items)):
        for item in items:
            response = run_query(
                QueryRequest(
                    question=item.question,
                    strategy=strategy,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    use_compression=False,
                    use_guardrails=False,
                )
            )
            latencies.append(response.latency_ms)
            samples.append(
                {
                    "question": item.question,
                    "answer": response.answer,
                    "contexts": [c.content for c in response.sources],
                    "ground_truth": item.ground_truth,
                }
            )

        per_sample_scores, aggregate = ragas_eval.evaluate_samples(samples, metrics)

    per_question = [
        PerQuestionResult(
            question=s["question"],
            answer=s["answer"],
            ground_truth=s["ground_truth"],
            num_contexts=len(s["contexts"]),
            scores=scores,
        )
        for s, scores in zip(samples, per_sample_scores)
    ]

    return StrategyEvalResult(
        strategy=strategy.value,
        num_questions=len(items),
        aggregate=aggregate,
        per_question=per_question,
        avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
    )


def run_evaluation(request: EvalRequest) -> EvalResponse:
    start = time.perf_counter()
    items = dataset.golden_items(limit=request.num_questions)
    metrics = [m.value for m in (request.metrics or list(EvalMetric))]

    with logfire.span(
        "eval.run",
        strategies=[s.value for s in request.strategies],
        metrics=metrics,
        num_questions=len(items),
    ) as span:
        results = [
            _run_strategy(strategy, items, metrics, request.top_k, request.use_rerank)
            for strategy in request.strategies
        ]
        total_latency_ms = round((time.perf_counter() - start) * 1000, 1)
        span.set_attribute("total_latency_ms", total_latency_ms)

        return EvalResponse(
            metrics=metrics,
            dataset_size=dataset.size(),
            num_questions=len(items),
            results=results,
            total_latency_ms=total_latency_ms,
        )
