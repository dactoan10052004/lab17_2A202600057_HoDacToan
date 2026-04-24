from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from memory_agent.config import MEMORY_BUDGET_TOKENS, OPENAI_MODEL, USER_ID
from memory_agent.context import format_memory_context
from memory_agent.memory.episodic import EpisodicMemoryJSON
from memory_agent.memory.profile import LongTermMemoryRedis
from memory_agent.memory.semantic import SemanticMemoryChroma
from memory_agent.memory.short_term import ShortTermMemory
from memory_agent.routing import route_memory
from memory_agent.state import MemoryState


short_memory = ShortTermMemory()
long_memory = LongTermMemoryRedis()
episodic_memory = EpisodicMemoryJSON()
semantic_memory = SemanticMemoryChroma()


def retrieve_memory(state: MemoryState) -> MemoryState:
    """Router node: retrieve each memory backend and pack it into state."""
    messages = state.get("messages", [])
    user_id = state.get("user_id", USER_ID)
    query = messages[-1].content if messages else ""

    route = route_memory(query)
    summary, recent = short_memory.build(messages)
    recent_conversation = short_memory._format_messages(recent)

    profile = long_memory.load_profile(user_id) if route["long_term"] else {}
    episodes = episodic_memory.search(query, k=3, user_id=user_id) if route["episodic"] else []
    semantic_hits = semantic_memory.query(query, k=4) if route["semantic"] else []

    memory_context = format_memory_context(
        summary=summary,
        user_profile=profile,
        episodes=episodes,
        semantic_hits=semantic_hits,
        recent_conversation=recent_conversation,
        budget_tokens=state.get("memory_budget", MEMORY_BUDGET_TOKENS),
    )

    return {
        **state,
        "messages": recent,
        "summary": summary,
        "user_profile": profile,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "route": route,
        "memory_context": memory_context,
    }


load_memory = retrieve_memory


def call_llm(state: MemoryState) -> MemoryState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    memory_context = state.get("memory_context", "")

    system = SystemMessage(
        content=f"""You are a Multi-Memory Agent.
Use memory context only when relevant. Do not invent facts when memory is
missing. If the latest user message conflicts with older memory, prefer the
latest explicit user statement.

When memory contains a user-stated outcome, correction, failed attempt, or
lesson learned, use that concrete memory before giving generic advice. If the
user asks what to try first, answer with the remembered successful direction
first, then add short supporting checks.

For benchmark or report requests, do not ask for more requirements when the
memory context already contains lab requirements. Include the required groups
from memory when present: profile recall, conflict update, episodic recall,
semantic retrieval, and trim/token budget.

The packed memory context may contain these sections:
- RECENT_CONVERSATION
- SHORT_TERM_SUMMARY
- LONG_TERM_PROFILE
- EPISODIC_MEMORY
- SEMANTIC_MEMORY

MEMORY CONTEXT:
{memory_context}
"""
    )

    result = llm.invoke([system] + state.get("messages", []))
    return {**state, "answer": result.content, "messages": state.get("messages", []) + [AIMessage(content=result.content)]}


def save_memory(state: MemoryState) -> MemoryState:
    """Write memory after a turn has a clear assistant outcome."""
    user_id = state.get("user_id", USER_ID)
    messages = state.get("messages", [])

    long_term_update = long_memory.extract_and_save(user_id, messages)

    if messages:
        user_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m.content for m in messages if isinstance(m, AIMessage)]
        if user_msgs and ai_msgs:
            episodic_memory.add_episode(
                task=user_msgs[-1],
                trajectory=f"Route={state.get('route', {})}; retrieved memory and generated a response.",
                outcome=ai_msgs[-1][:500],
                reflection="Reuse useful retrieval choices and avoid stale profile facts after conflict updates.",
                user_id=user_id,
            )
            episodic_memory.consolidate()

    return {**state, "long_term_update": long_term_update}


def build_agent():
    graph = StateGraph(MemoryState)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("call_llm", call_llm)
    graph.add_node("save_memory", save_memory)

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "call_llm")
    graph.add_edge("call_llm", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


def run_turn(
    user_input: str,
    history: List[BaseMessage] | None = None,
    user_id: str = USER_ID,
) -> tuple[str, List[BaseMessage], MemoryState]:
    app = build_agent()
    history = history or []
    messages = history + [HumanMessage(content=user_input)]
    state: MemoryState = {
        "messages": messages,
        "user_id": user_id,
        "memory_budget": MEMORY_BUDGET_TOKENS,
    }
    result = app.invoke(state)
    return result["answer"], result["messages"], result
