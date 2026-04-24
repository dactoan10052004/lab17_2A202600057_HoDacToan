from __future__ import annotations

import unicodedata
from typing import Dict


PREFERENCE_WORDS = {"thich", "khong thich", "prefer", "preference", "style", "ngon ngu", "language"}
EPISODIC_WORDS = {"lan truoc", "truoc do", "da lam", "da thu", "kinh nghiem", "bug cu", "similar task"}
SEMANTIC_WORDS = {
    "giai thich",
    "khai niem",
    "tai lieu",
    "knowledge",
    "domain",
    "factual",
    "dinh nghia",
    "so sanh",
    "faq",
    "benchmark",
    "report",
    "context priority",
}
FACT_WORDS = {"toi la", "toi dang", "toi co", "nho rang", "remember", "fact", "di ung"}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D").lower()


def route_memory(query: str) -> Dict[str, bool]:
    q = _normalize(query)

    want_long = any(word in q for word in PREFERENCE_WORDS | FACT_WORDS)
    want_episode = any(word in q for word in EPISODIC_WORDS)
    want_semantic = any(word in q for word in SEMANTIC_WORDS)

    if not (want_episode or want_semantic):
        want_semantic = True

    return {
        "short_term": True,
        "long_term": True,
        "episodic": want_episode,
        "semantic": want_semantic,
    }
