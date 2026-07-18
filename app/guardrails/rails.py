"""Guardrail orchestration — input rails (before retrieval) and output rails
(after generation).

Input rails can *block* (injection, moderation, optionally PII); output rails
*transform/flag* (PII redaction, grounding). Every individual guardrail is
config-toggleable and degrades gracefully, so the pipeline is never brought
down by a guardrail component.
"""

import logfire

from app.config import get_settings
from app.guardrails import grounding, injection, moderation, pii
from app.models.schemas import GuardrailCheck, GuardrailReport, SourceChunk


def status() -> dict:
    """Which guardrails are active — for /health."""
    settings = get_settings()
    return {
        "enabled": settings.guardrails_enabled,
        "injection": settings.guardrail_injection,
        "moderation": settings.guardrail_moderation,
        "input_pii": settings.guardrail_input_pii and pii.is_available(),
        "output_pii": settings.guardrail_output_pii and pii.is_available(),
        "grounding": settings.guardrail_grounding,
        "pii_backend": "presidio" if pii.is_available() else "unavailable",
    }


def check_input(question: str) -> GuardrailReport:
    """Run input rails. report.passed is False when the request must be blocked."""
    settings = get_settings()
    checks: list[GuardrailCheck] = []

    with logfire.span("guardrails.input") as span:
        if settings.guardrail_injection:
            attack = injection.detect(question)
            checks.append(
                GuardrailCheck(
                    name="prompt_injection", passed=attack is None, detail=attack
                )
            )
            if attack:
                span.set_attribute("blocked", "prompt_injection")
                return GuardrailReport(
                    passed=False,
                    blocked_reason="Your message looks like a prompt-injection "
                    "attempt and was blocked.",
                    checks=checks,
                )

        if settings.guardrail_moderation:
            flagged, categories = moderation.check(question)
            checks.append(
                GuardrailCheck(
                    name="moderation",
                    passed=not flagged,
                    detail=", ".join(categories) or None,
                )
            )
            if flagged:
                span.set_attribute("blocked", "moderation")
                return GuardrailReport(
                    passed=False,
                    blocked_reason="Your message was flagged by content "
                    f"moderation ({', '.join(categories)}).",
                    checks=checks,
                )

        pii_found: list[str] = []
        if settings.guardrail_input_pii and pii.is_available():
            pii_found = pii.detect(question)
            blocked = bool(pii_found) and settings.block_on_input_pii
            checks.append(
                GuardrailCheck(
                    name="input_pii",
                    passed=not blocked,
                    detail=", ".join(pii_found) or None,
                )
            )
            if blocked:
                span.set_attribute("blocked", "input_pii")
                return GuardrailReport(
                    passed=False,
                    blocked_reason="Your message contains personal data "
                    f"({', '.join(pii_found)}); please remove it and retry.",
                    checks=checks,
                    pii_detected=pii_found,
                )

        span.set_attribute("passed", True)
        return GuardrailReport(passed=True, checks=checks, pii_detected=pii_found)


def apply_output(
    answer: str, chunks: list[SourceChunk], report: GuardrailReport
) -> tuple[str, bool | None]:
    """Redact PII from the answer and (optionally) grounding-check it.

    Extends `report` in place; returns (possibly-redacted answer, grounded?).
    """
    settings = get_settings()
    grounded: bool | None = None

    with logfire.span("guardrails.output"):
        if settings.guardrail_output_pii and pii.is_available():
            redacted, found = pii.redact(answer)
            report.checks.append(
                GuardrailCheck(
                    name="output_pii",
                    passed=not found,
                    detail=", ".join(found) or None,
                )
            )
            if found:
                report.pii_detected = sorted(set(report.pii_detected) | set(found))
            answer = redacted

        if settings.guardrail_grounding:
            grounded, reason = grounding.check(answer, chunks)
            report.checks.append(
                GuardrailCheck(
                    name="grounding",
                    passed=grounded,
                    detail=None if grounded else reason,
                )
            )

    return answer, grounded
