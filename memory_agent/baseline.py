from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory_agent.config import OPENAI_MODEL


def run_baseline_turn(user_input: str, history: List[BaseMessage] | None = None):
    """No external memory baseline. It only sees the passed conversation history."""
    history = history or []
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    messages = [SystemMessage(content="You are a helpful assistant with no external memory.")] + history + [
        HumanMessage(content=user_input)
    ]
    result = llm.invoke(messages)
    new_history = history + [HumanMessage(content=user_input), AIMessage(content=result.content)]
    return result.content, new_history
