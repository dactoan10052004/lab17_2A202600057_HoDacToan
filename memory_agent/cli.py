from langchain_core.messages import BaseMessage

from memory_agent.config import USER_ID
from memory_agent.graph import run_turn


def main() -> None:
    history: list[BaseMessage] = []

    print("Multi-Memory Agent. Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        answer, history, state = run_turn(user_input, history=history, user_id=USER_ID)
        print("\nAgent:", answer)
        print("Memory route:", state.get("route"))
        print("-" * 80)


if __name__ == "__main__":
    main()
