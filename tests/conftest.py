"""Shared pytest setup.

Loads the project's .env so tests read API keys/config from a single source
(never hardcoded). The offline unit tests don't make network calls, so having
real keys present is harmless; the `live` tests (tests/test_live.py) use them
to hit the real APIs and are skipped when no real key is configured.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# API keys + config come from .env only.
load_dotenv(ROOT / ".env")

# Quiet Logfire when it isn't configured during tests.
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: exercises real external APIs (OpenAI/Cohere/Neo4j) using .env keys; "
        "run with `pytest -m live`",
    )


def has_real_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return key.startswith("sk-") and "test" not in key.lower() and len(key) > 20


@pytest.fixture(scope="session")
def require_live() -> None:
    """Skip a live test unless a real OpenAI key is present in .env."""
    if not has_real_openai_key():
        pytest.skip("no real OPENAI_API_KEY in .env — skipping live test")


@pytest.fixture
def graph_disabled(monkeypatch: pytest.MonkeyPatch):
    """Force Neo4j off deterministically, regardless of .env, and restore after."""
    from app.config import get_settings
    from app.graph import client as graph_client

    monkeypatch.setenv("NEO4J_URI", "")
    get_settings.cache_clear()
    graph_client.reset()
    yield
    get_settings.cache_clear()
    graph_client.reset()
