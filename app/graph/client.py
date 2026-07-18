"""Neo4j driver — graceful singleton.

Graph features are optional: without NEO4J_URI (or with Neo4j down) the app
runs normally and graph strategies report clearly instead of erroring.
"""

import threading

import logfire
from neo4j import Driver, GraphDatabase

from app.config import get_settings

_lock = threading.Lock()
_driver: Driver | None = None
_checked = False


def get_driver() -> Driver | None:
    """Connected Neo4j driver, or None if unconfigured/unreachable."""
    global _driver, _checked
    with _lock:
        if _checked:
            return _driver

        settings = get_settings()
        if not settings.neo4j_uri:
            logfire.warn("neo4j_disabled_no_uri")
            _checked = True
            return None

        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            driver.verify_connectivity()
            _driver = driver
            logfire.info("neo4j_connected", uri=settings.neo4j_uri)
        except Exception as e:
            logfire.error("neo4j_unreachable", uri=settings.neo4j_uri, error=str(e))
            _driver = None
        _checked = True
        return _driver


def is_enabled() -> bool:
    return get_driver() is not None


def reset() -> None:
    """Re-check connectivity on next access (used by tests / reconnects)."""
    global _driver, _checked
    with _lock:
        if _driver is not None:
            _driver.close()
        _driver = None
        _checked = False
