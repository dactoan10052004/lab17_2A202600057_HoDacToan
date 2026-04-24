from memory_agent.memory.semantic import SemanticMemoryChroma

docs = [
    "Short-term memory in agents is the current context window buffer. It is fast but temporary and should be trimmed when near the token limit.",
    "Long-term memory stores durable user preferences and facts across sessions. Redis hashes are useful for preferences; Redis sets are useful for user facts.",
    "Episodic memory stores past trajectories: task, actions tried, outcome, and reflection. It helps agents avoid repeating failed approaches.",
    "Semantic memory stores domain knowledge in a vector database. Query is embedded and compared with chunks using cosine similarity.",
    "Context priority for memory packing: short-term first, long-term facts second, relevant episodes third, semantic knowledge fourth.",
    "Privacy-by-design: do not persist sensitive personal data unless user explicitly opts in. Support deletion and TTL-based storage limitation.",
    "Lab 17 benchmark must contain exactly 10 multi-turn conversations comparing no-memory and with-memory results. Required groups: profile recall, conflict update, episodic recall, semantic retrieval, and trim/token budget.",
]

metas = [
    {"source": "lab17_seed", "confidence": 0.95, "type": "short_term"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "long_term"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "episodic"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "semantic"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "context"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "privacy"},
    {"source": "lab17_seed", "confidence": 0.95, "type": "benchmark"},
]

def main() -> None:
    sm = SemanticMemoryChroma()
    sm.add_documents(docs, metas)
    print(f"Seeded {len(docs)} semantic documents into {sm.backend} semantic store.")


if __name__ == "__main__":
    main()
