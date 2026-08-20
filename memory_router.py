"""Deterministic, conservative routing for long-term memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryDecision:
    """A deterministic decision about whether a user turn belongs in memory."""

    action: str
    reason: str
    kind: str | None = None
    topic: str | None = None
    value: str | None = None


def _clean_value(value: str) -> str:
    return re.sub(r"[.!?]+$", "", value.strip(), flags=re.IGNORECASE)


def route_memory_write(message: str) -> MemoryDecision:
    """Classify explicit durable facts as keep, drop, or uncertain without an LLM."""
    text = " ".join(message.strip().split())
    lower = text.lower()

    if not text:
        return MemoryDecision("drop", "empty message")
    if re.fullmatch(r"(?:thanks|thank you|hi|hello|hey)[!. ]*", lower):
        return MemoryDecision("drop", "social message")
    if any(marker in lower for marker in ("maybe", "might", "not sure", "perhaps")):
        return MemoryDecision("uncertain", "the statement is not a firm preference")
    if "how many calories" in lower or "how much protein" in lower:
        return MemoryDecision("drop", "nutrition question, not a durable fact")
    calorie_match = re.search(
        r"(?:usual(?:ly)?\s+)?(?P<meal>breakfast|lunch|dinner)?\s*(?:calorie\s+)?target\s+(?:is|to|at)?\s*(?P<value>\d{2,4})\s*(?:calories|kcal)?",
        lower,
    )
    if calorie_match:
        meal = calorie_match.group("meal") or "daily"
        return MemoryDecision(
            "keep",
            "explicit recurring calorie target",
            "calorie_target",
            f"{meal}_calorie_target",
            f"{calorie_match.group('value')} calories",
        )

    if re.search(r"\b(?:make|recommend|suggest)\b.*\b(?:today|tonight|now)\b", lower):
        return MemoryDecision("drop", "one-time request")

    allergy_match = re.search(r"(?:i am|i'm) allerg(?:ic to|y to) (?P<value>[^,.!?]+)", lower)
    noun_allergy_match = re.search(r"i have (?:an? )?(?P<value>[^,.!?]+?) allerg(?:y|ies)\b", lower)
    if allergy_match or noun_allergy_match:
        match = allergy_match or noun_allergy_match
        value = _clean_value(match.group("value"))
        return MemoryDecision("keep", "explicit allergy", "allergy", f"allergy:{value}", value)

    dietary_match = re.search(r"\b(i am|i'm)\s+(?P<value>vegan|vegetarian|pescatarian)\b", lower)
    if dietary_match:
        value = dietary_match.group("value")
        return MemoryDecision("keep", "explicit dietary constraint", "dietary_constraint", "dietary_pattern", value)

    dislike_match = re.search(r"\bi (?:dislike|hate|avoid)\s+(?P<value>[^,.!?]+)", lower)
    if dislike_match:
        value = _clean_value(dislike_match.group("value"))
        return MemoryDecision(
            "keep", "explicit ingredient aversion", "disliked_ingredient", f"disliked_ingredient:{value}", value
        )

    preference_match = re.search(
        r"\bi (?:usually|generally|always) (?:prefer|like|choose)\s+(?P<value>[^,.!?]+)", lower
    )
    if preference_match:
        value = _clean_value(preference_match.group("value"))
        meal_match = re.search(r"\b(breakfast|lunch(?:es)?|dinner(?:s)?)\b", value)
        meal = meal_match.group(1).rstrip("es") if meal_match else "general"
        return MemoryDecision(
            "keep", "explicit recurring meal preference", "meal_preference", f"{meal}_style", value
        )

    return MemoryDecision("uncertain", "no supported durable-memory pattern matched")
