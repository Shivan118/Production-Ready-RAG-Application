"""Neo4j driver — graceful singleton with auto-recovery.

Graph features are optional: without NEO4J_URI (or with Neo4j down) the app
runs normally and graph strategies report clearly instead of erroring.

A *transient* connection failure must not disable graph for the whole process
lifetime — so failures are cached only briefly (RETRY_COOLDOWN); the next call
after the cooldown re-attempts and recovers automatically once Neo4j is back.
An empty NEO4J_URI (feature off by config) is treated as a permanent, cheap no-op.
"""

import threading
import time

import logfire
from neo4j import Driver, GraphDatabase

from app.config import get_settings

RETRY_COOLDOWN = 30.0  # seconds to back off after a failed connection attempt

_lock = threading.Lock()
_driver: Driver | None = None
_last_fail_at: float = 0.0


def get_driver() -> Driver | None:
    """Connected Neo4j driver, or None if unconfigured/unreachable.

    Retries automatically after RETRY_COOLDOWN following a transient failure.
    """
    global _driver, _last_fail_at
    with _lock:
        if _driver is not None:
            return _driver

        settings = get_settings()
        if not settings.neo4j_uri:
            return None  # feature off by config — nothing to retry

        # back off briefly after a recent failure to avoid per-request hammering
        if _last_fail_at and (time.monotonic() - _last_fail_at) < RETRY_COOLDOWN:
            return None

        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            driver.verify_connectivity()
            _driver = driver
            _last_fail_at = 0.0
            logfire.info("neo4j_connected", uri=settings.neo4j_uri)
            return _driver
        except Exception as e:
            _last_fail_at = time.monotonic()
            logfire.error("neo4j_unreachable", uri=settings.neo4j_uri, error=str(e))
            return None


def is_enabled() -> bool:
    return get_driver() is not None


def reset() -> None:
    """Drop any cached driver/backoff so the next access re-checks connectivity."""
    global _driver, _last_fail_at
    with _lock:
        if _driver is not None:
            _driver.close()
        _driver = None
        _last_fail_at = 0.0
