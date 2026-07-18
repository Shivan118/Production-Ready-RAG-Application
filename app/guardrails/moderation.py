"""Content moderation via OpenAI's (free) moderation endpoint.

Flags disallowed content categories (hate, violence, self-harm, sexual, …).
Degrades gracefully: any API error is logged and treated as "not flagged"
so moderation never takes the whole request down.
"""

import logfire
import openai

from app.runtime_keys import effective_openai_key

MODERATION_MODEL = "omni-moderation-latest"


def check(text: str) -> tuple[bool, list[str]]:
    """Return (flagged, categories). (False, []) on any error."""
    key = effective_openai_key()
    if not key:
        return False, []

    with logfire.span("guardrail.moderation") as span:
        try:
            client = openai.OpenAI(api_key=key)
            result = client.moderations.create(
                model=MODERATION_MODEL, input=text
            ).results[0]
        except Exception as e:  # noqa: BLE001 — never fail the request on moderation
            logfire.warn("moderation_failed", error=str(e))
            return False, []

        flagged_categories = [
            name
            for name, is_flagged in result.categories.model_dump().items()
            if is_flagged
        ]
        span.set_attribute("flagged", result.flagged)
        span.set_attribute("categories", flagged_categories)
        return bool(result.flagged), flagged_categories
