"""PII detection and redaction with Microsoft Presidio.

Presidio + spaCy are optional: if either is missing or the model isn't
downloaded, PII guardrails report as unavailable and the app keeps running.
"""

from functools import lru_cache

import logfire

from app.config import get_settings

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    _IMPORTED = True
except ImportError:
    _IMPORTED = False


@lru_cache
def _engines():
    """(analyzer, anonymizer) or None if Presidio/spaCy are unavailable."""
    if not _IMPORTED:
        logfire.warn("presidio_not_installed")
        return None

    settings = get_settings()
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": settings.pii_language, "model_name": settings.spacy_model}
                ],
            }
        )
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=[settings.pii_language]
        )
        return analyzer, AnonymizerEngine()
    except Exception as e:
        logfire.warn("presidio_init_failed", error=str(e))
        return None


def is_available() -> bool:
    return _engines() is not None


def _analyze(text: str):
    engines = _engines()
    if engines is None:
        return []
    analyzer, _ = engines
    settings = get_settings()
    return analyzer.analyze(
        text=text,
        language=settings.pii_language,
        score_threshold=settings.pii_score_threshold,
        entities=settings.pii_entities or None,  # None = all built-in recognizers
    )


def detect(text: str) -> list[str]:
    """Return the distinct PII entity types found (e.g. ['EMAIL_ADDRESS'])."""
    return sorted({r.entity_type for r in _analyze(text)})


def redact(text: str) -> tuple[str, list[str]]:
    """Replace PII spans with <ENTITY_TYPE> tags.

    Returns (redacted_text, entity_types_found). No-op when unavailable.
    """
    engines = _engines()
    if engines is None:
        return text, []
    _, anonymizer = engines
    results = _analyze(text)
    if not results:
        return text, []

    with logfire.span("pii.redact", entities=len(results)) as span:
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        found = sorted({r.entity_type for r in results})
        span.set_attribute("entity_types", found)
        return anonymized.text, found
