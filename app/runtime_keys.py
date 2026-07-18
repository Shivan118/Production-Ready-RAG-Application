"""Per-request API key overrides.

The Streamlit UI (or any client) can send its own keys via headers
(X-OpenAI-Api-Key / X-Cohere-Api-Key). They are stored in request-scoped
contextvars; every client factory resolves the effective key at call time
and falls back to the server's .env settings.
"""

from contextvars import ContextVar

from app.config import get_settings

openai_key_override: ContextVar[str | None] = ContextVar(
    "openai_key_override", default=None
)
cohere_key_override: ContextVar[str | None] = ContextVar(
    "cohere_key_override", default=None
)


def effective_openai_key() -> str:
    return openai_key_override.get() or get_settings().openai_api_key


def effective_cohere_key() -> str:
    return cohere_key_override.get() or get_settings().cohere_api_key
