from __future__ import annotations

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def trim_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    if max_tokens <= 0:
        return ""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[-max_tokens:])
