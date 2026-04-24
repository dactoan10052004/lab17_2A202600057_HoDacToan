from __future__ import annotations

from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage


class MemoryState(TypedDict, total=False):
    messages: List[BaseMessage]
    user_id: str

    # Retrieved memory blocks
    user_profile: Dict[str, Any]       # long-term
    episodes: List[Dict[str, Any]]     # episodic
    semantic_hits: List[str]           # semantic
    summary: str                       # short-term summary

    # Runtime metadata
    memory_budget: int
    memory_context: str
    answer: str
    route: Dict[str, bool]
    long_term_update: Dict[str, Any]
