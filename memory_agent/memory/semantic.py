from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from memory_agent.config import CHROMA_DIR, OPENAI_API_KEY, SEMANTIC_DOC_PATH


DEFAULT_SEMANTIC_DOCS = [
    {
        "text": "Short-term memory is the current conversation buffer plus a sliding window or summary. It is fast, temporary, and must be trimmed when the context budget is tight.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "short_term"},
    },
    {
        "text": "Long-term profile memory stores durable user preferences and facts across sessions. Redis hashes or a JSON key-value store are good backends.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "long_term"},
    },
    {
        "text": "Episodic memory stores past trajectories: task, actions tried, outcome, and reflection. It helps an agent avoid repeating failed approaches.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "episodic"},
    },
    {
        "text": "Semantic memory stores reusable domain knowledge in chunks. A vector database such as Chroma retrieves chunks by embedding similarity; keyword search is a simple fallback.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "semantic"},
    },
    {
        "text": "Memory context packing should prioritize recent conversation, profile facts, relevant episodes, then semantic knowledge, while enforcing a token budget.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "context"},
    },
    {
        "text": "Privacy-by-design for memory agents means consent before storing sensitive PII, deletion support across all backends, TTLs, and care with wrong retrieval.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "privacy"},
    },
    {
        "text": "Lab 17 benchmark must contain exactly 10 multi-turn conversations comparing no-memory and with-memory results. Required groups: profile recall, conflict update, episodic recall, semantic retrieval, and trim/token budget.",
        "metadata": {"source": "lab17_seed", "confidence": 0.95, "type": "benchmark"},
    },
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


class SemanticMemoryChroma:
    """Semantic memory with Chroma primary backend and keyword fallback."""

    def __init__(self, collection_name: str = "lab17_semantic", fallback_path: Path = SEMANTIC_DOC_PATH):
        self.fallback_path = fallback_path
        self.fallback_path.parent.mkdir(exist_ok=True)
        self.collection = None
        self.backend = "keyword"

        if OPENAI_API_KEY:
            try:
                import chromadb
                from chromadb.utils import embedding_functions

                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-3-small",
                )
                self.collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
                self.backend = "chroma"
            except Exception:
                self.collection = None

        if not self.fallback_path.exists():
            self._write_fallback(DEFAULT_SEMANTIC_DOCS)

    def _read_fallback(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_fallback(self, docs: List[Dict[str, Any]]) -> None:
        self.fallback_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[str], metadatas: List[Dict[str, Any]] | None = None) -> None:
        if not docs:
            return
        metadatas = metadatas or [{} for _ in docs]

        current = self._read_fallback()
        by_text = {item.get("text", ""): item for item in current}
        for doc, metadata in zip(docs, metadatas):
            by_text[doc] = {"text": doc, "metadata": metadata}
        self._write_fallback(list(by_text.values()))

        if self.collection:
            try:
                ids = [f"doc_{abs(hash(d))}" for d in docs]
                self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            except Exception:
                self.collection = None
                self.backend = "keyword"

    def delete_user_memory(self, user_id: str) -> None:
        docs = [item for item in self._read_fallback() if item.get("metadata", {}).get("user_id") != user_id]
        self._write_fallback(docs)
        if self.collection:
            try:
                self.collection.delete(where={"user_id": user_id})
            except Exception:
                pass

    def _query_keyword(self, query: str, k: int) -> List[str]:
        query_tokens = _tokens(query)
        docs = self._read_fallback()

        def score(item: Dict[str, Any]) -> float:
            doc_tokens = _tokens(item.get("text", ""))
            overlap = len(query_tokens & doc_tokens)
            confidence = float(item.get("metadata", {}).get("confidence", 0.5))
            return overlap + confidence * 0.01

        ranked = sorted(docs, key=score, reverse=True)
        out = []
        for item in ranked[:k]:
            metadata = item.get("metadata", {})
            src = metadata.get("source", "local_keyword")
            conf = metadata.get("confidence", "n/a")
            out.append(f"[source={src}; confidence={conf}; backend={self.backend}] {item.get('text', '')}")
        return out

    def query(self, query: str, k: int = 4) -> List[str]:
        if self.collection:
            try:
                result = self.collection.query(query_texts=[query], n_results=k)
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                out = []
                for doc, meta in zip(docs, metas):
                    src = meta.get("source", "unknown")
                    conf = meta.get("confidence", "n/a")
                    out.append(f"[source={src}; confidence={conf}; backend=chroma] {doc}")
                return out
            except Exception:
                self.backend = "keyword"
        return self._query_keyword(query, k)
