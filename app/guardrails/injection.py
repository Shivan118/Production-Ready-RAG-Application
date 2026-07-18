"""Heuristic prompt-injection detection.

Fast, zero-cost first line of defense: regex patterns for the common
"ignore your instructions / reveal your system prompt / act as" attacks.
Not a complete solution (an LLM classifier would catch more) but it blocks
the obvious cases before any tokens are spent.
"""

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b"
        r".{0,30}\b(instruction|prompt|context|rule|direction)", re.I)),
    ("reveal_system_prompt", re.compile(
        r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}"
        r"\b(system\s+prompt|your\s+instructions|initial\s+prompt)", re.I)),
    ("role_override", re.compile(
        r"\byou\s+are\s+now\b|\bnew\s+(instructions?|persona|role)\s*:|"
        r"\bpretend\s+(to\s+be|you\s+are)\b|\bact\s+as\s+(a|an|if)\b", re.I)),
    ("mode_jailbreak", re.compile(
        r"\b(developer\s+mode|jailbreak|DAN\s+mode|unrestricted\s+mode|"
        r"do\s+anything\s+now)\b", re.I)),
    ("override_guardrails", re.compile(
        r"\b(override|bypass|disable|turn\s+off)\b.{0,30}"
        r"\b(guardrail|safety|filter|restriction|rule)", re.I)),
]


def detect(text: str) -> str | None:
    """Return the name of the first matching attack pattern, or None."""
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            return name
    return None
