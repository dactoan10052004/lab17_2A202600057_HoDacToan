from __future__ import annotations

from typing import Dict, Any, List

from memory_agent.config import MEMORY_BUDGET_TOKENS, OPENAI_MODEL
from memory_agent.utils.tokens import count_tokens, trim_to_tokens


def format_memory_context(
    summary: str,
    user_profile: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    semantic_hits: List[str],
    recent_conversation: str = "",
    budget_tokens: int = MEMORY_BUDGET_TOKENS,
) -> str:
    """Priority-based context packing.

    Priority:
    1. Short-term summary/recent context
    2. Long-term profile/facts
    3. Episodic
    4. Semantic

    Khi quá budget: trim semantic trước, sau đó episodic, sau đó long-term.
    """

    blocks = []

    if recent_conversation:
        blocks.append(("RECENT_CONVERSATION", recent_conversation, 0.25))

    if summary:
        blocks.append(("SHORT_TERM_SUMMARY", summary, 0.20))

    if user_profile:
        profile_text = []
        prefs = user_profile.get("preferences", {})
        facts = user_profile.get("facts", [])
        if prefs:
            profile_text.append("Preferences: " + str(prefs))
        if facts:
            if isinstance(facts, dict):
                fact_text = "; ".join(f"{key}={value}" for key, value in facts.items())
            else:
                fact_text = "; ".join(str(fact) for fact in facts)
            profile_text.append("Facts: " + fact_text)
        if profile_text:
            blocks.append(("LONG_TERM_PROFILE", "\n".join(profile_text), 0.25))

    if episodes:
        ep_text = "\n".join(
            f"- Task: {e.get('task')} | Outcome: {e.get('outcome')} | Reflection: {e.get('reflection')}"
            for e in episodes
        )
        blocks.append(("EPISODIC_MEMORY", ep_text, 0.15))

    if semantic_hits:
        sem_text = "\n".join(f"- {s}" for s in semantic_hits)
        blocks.append(("SEMANTIC_MEMORY", sem_text, 0.15))

    final_parts = []
    used = 0
    for name, text, ratio in blocks:
        max_for_block = max(80, int(budget_tokens * ratio))
        clipped = trim_to_tokens(text, max_for_block, OPENAI_MODEL)
        t = count_tokens(clipped, OPENAI_MODEL)
        if used + t > budget_tokens:
            clipped = trim_to_tokens(clipped, budget_tokens - used, OPENAI_MODEL)
        if clipped:
            final_parts.append(f"## {name}\n{clipped}")
            used += count_tokens(clipped, OPENAI_MODEL)
        if used >= budget_tokens:
            break

    return "\n\n".join(final_parts)
