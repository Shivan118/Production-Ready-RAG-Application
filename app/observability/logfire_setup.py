"""Pydantic Logfire setup — logs, traces and metrics for the whole app.

With LOGFIRE_TOKEN set, everything streams to the Logfire dashboard.
Without a token, spans still print locally so development is fully observable.
"""

import logfire

from app.config import get_settings

_configured = False


def setup_logfire() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    logfire.configure(
        service_name="rag-end-to-end",
        environment=settings.environment,
        token=settings.logfire_token or None,
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions(min_log_level="info"),
    )
    # Auto-trace every OpenAI call (chat + embeddings): tokens, latency, prompts
    logfire.instrument_openai()
    _configured = True


def instrument_fastapi(app) -> None:
    """Trace every HTTP request/response through the API."""
    logfire.instrument_fastapi(app)
