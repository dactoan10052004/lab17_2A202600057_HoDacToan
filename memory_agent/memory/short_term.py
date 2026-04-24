from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from memory_agent.config import OPENAI_API_KEY, OPENAI_MODEL
from memory_agent.utils.tokens import count_tokens, trim_to_tokens


class ShortTermMemory:
    """Conversation buffer + summary + sliding window."""

    def __init__(self, max_tokens: int = 900, recent_turns: int = 6):
        self.max_tokens = max_tokens
        self.recent_turns = recent_turns
        self.summary = ""

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        lines = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def build(self, messages: List[BaseMessage]) -> tuple[str, List[BaseMessage]]:
        if not messages:
            return self.summary, []

        recent = messages[-self.recent_turns :]
        old = messages[: -self.recent_turns]

        recent_text = self._format_messages(recent)
        total = count_tokens((self.summary or "") + recent_text, OPENAI_MODEL)

        if old and total > self.max_tokens:
            self.summary = self.summarize(old, previous_summary=self.summary)

        available = max(100, self.max_tokens - count_tokens(recent_text, OPENAI_MODEL))
        self.summary = trim_to_tokens(self.summary, available, OPENAI_MODEL)
        return self.summary, recent

    def summarize(self, old_messages: List[BaseMessage], previous_summary: str = "") -> str:
        raw_history = self._format_messages(old_messages)
        fallback = "\n".join(part for part in [previous_summary, trim_to_tokens(raw_history, 240, OPENAI_MODEL)] if part)
        if not OPENAI_API_KEY:
            return fallback

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        prompt = f"""Summarize old conversation history for an agent memory buffer.
Keep durable preferences, constraints, decisions, tried fixes, outcomes, and open tasks.

Previous summary:
{previous_summary}

Old history:
{raw_history}

Short Vietnamese summary:
"""
        try:
            return llm.invoke(prompt).content.strip()
        except Exception:
            return fallback
