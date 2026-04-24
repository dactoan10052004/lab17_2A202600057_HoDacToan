from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

try:
    import redis
except Exception:  # pragma: no cover - exercised when optional dependency is absent.
    redis = None

from memory_agent.config import (
    ENABLE_LLM_MEMORY_EXTRACTION,
    FACT_TTL_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PREF_TTL_SECONDS,
    PROFILE_STORE_PATH,
    REDIS_URL,
)
from memory_agent.memory.short_term import ShortTermMemory


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def _clean_value(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip(" .,!?:;\"'")
    return value.lower()


class LongTermMemoryRedis:
    """Long-term profile memory.

    Primary backend: Redis hashes with TTL.
    Fallback backend: JSON KV store in data/profiles.json.

    Facts are stored by stable field name, not appended as free text. That lets
    a later correction replace the older value instead of creating conflict.
    """

    def __init__(
        self,
        url: str = REDIS_URL,
        fallback_path: Path = PROFILE_STORE_PATH,
        use_llm_extraction: bool = ENABLE_LLM_MEMORY_EXTRACTION,
    ):
        self.fallback_path = fallback_path
        self.fallback_path.parent.mkdir(exist_ok=True)
        self.use_llm_extraction = use_llm_extraction and bool(OPENAI_API_KEY)
        self.r: Any | None = None
        self.backend = "json"
        if redis is not None:
            try:
                client = redis.Redis.from_url(url, decode_responses=True)
                client.ping()
                self.r = client
                self.backend = "redis"
            except Exception:
                self.r = None

    def _pref_key(self, user_id: str) -> str:
        return f"user:{user_id}:preferences"

    def _fact_key(self, user_id: str) -> str:
        return f"user:{user_id}:facts"

    def _meta_key(self, user_id: str) -> str:
        return f"user:{user_id}:memory_meta"

    def _read_json_store(self) -> Dict[str, Any]:
        if not self.fallback_path.exists():
            return {}
        try:
            return json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json_store(self, store: Dict[str, Any]) -> None:
        self.fallback_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _safe_hgetall(self, key: str) -> Dict[str, str]:
        if not self.r:
            return {}
        try:
            return self.r.hgetall(key)
        except Exception:
            return {}

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        if self.r:
            prefs = self._safe_hgetall(self._pref_key(user_id))
            facts = self._safe_hgetall(self._fact_key(user_id))
            meta = self._safe_hgetall(self._meta_key(user_id))
            return {"preferences": prefs, "facts": facts, "meta": meta, "backend": self.backend}

        profile = self._read_json_store().get(user_id, {})
        return {
            "preferences": profile.get("preferences", {}),
            "facts": profile.get("facts", {}),
            "meta": profile.get("meta", {}),
            "backend": self.backend,
        }

    def save_preferences(self, user_id: str, prefs: Dict[str, str]) -> None:
        prefs = {k: str(v) for k, v in prefs.items() if v not in (None, "")}
        if not prefs:
            return
        if self.r:
            try:
                self.r.hset(self._pref_key(user_id), mapping=prefs)
            except Exception:
                self.r.delete(self._pref_key(user_id))
                self.r.hset(self._pref_key(user_id), mapping=prefs)
            self.r.expire(self._pref_key(user_id), PREF_TTL_SECONDS)
            return

        store = self._read_json_store()
        profile = store.setdefault(user_id, {"preferences": {}, "facts": {}, "meta": {}})
        profile.setdefault("preferences", {}).update(prefs)
        profile.setdefault("meta", {})["updated_at"] = time.time()
        self._write_json_store(store)

    def save_facts(self, user_id: str, facts: Dict[str, str]) -> None:
        facts = {k: str(v) for k, v in facts.items() if v not in (None, "")}
        if not facts:
            return
        if self.r:
            try:
                self.r.hset(self._fact_key(user_id), mapping=facts)
            except Exception:
                self.r.delete(self._fact_key(user_id))
                self.r.hset(self._fact_key(user_id), mapping=facts)
            self.r.expire(self._fact_key(user_id), FACT_TTL_SECONDS)
            self.r.hset(self._meta_key(user_id), mapping={"updated_at": str(time.time())})
            self.r.expire(self._meta_key(user_id), FACT_TTL_SECONDS)
            return

        store = self._read_json_store()
        profile = store.setdefault(user_id, {"preferences": {}, "facts": {}, "meta": {}})
        profile.setdefault("facts", {}).update(facts)
        profile.setdefault("meta", {})["updated_at"] = time.time()
        self._write_json_store(store)

    def delete_user_memory(self, user_id: str) -> None:
        """Deletion hook for privacy requests."""
        if self.r:
            self.r.delete(self._pref_key(user_id), self._fact_key(user_id), self._meta_key(user_id))
            return

        store = self._read_json_store()
        store.pop(user_id, None)
        self._write_json_store(store)

    def _extract_deterministic(self, raw: str) -> Dict[str, Any]:
        normalized = _strip_accents(raw).lower()
        prefs: Dict[str, str] = {}
        facts: Dict[str, str] = {}
        conflicts: List[Dict[str, str]] = []

        correction_matches = list(re.finditer(
            r"(?:^|\n|user:\s*)(?:a nham,?\s*)?toi di ung (?P<new>[^\n.]+?) chu khong phai (?P<old>[^\n.]+?)(?:[.\n]|$)",
            normalized,
            flags=re.DOTALL,
        ))
        correction = correction_matches[-1] if correction_matches else None
        if correction:
            new_value = _clean_value(raw[correction.start("new") : correction.end("new")])
            old_value = _clean_value(raw[correction.start("old") : correction.end("old")])
            facts["allergy"] = new_value
            conflicts.append(
                {
                    "field": "allergy",
                    "old_value": old_value,
                    "new_value": new_value,
                    "resolution": "latest_explicit_correction_wins",
                }
            )
        else:
            allergy_matches = list(re.finditer(r"(?:^|\n|user:\s*)toi di ung (?P<item>[^\n.]+?)(?:[.\n]|$)", normalized, flags=re.DOTALL))
            allergy = allergy_matches[-1] if allergy_matches else None
            if allergy:
                facts["allergy"] = _clean_value(raw[allergy.start("item") : allergy.end("item")])

        if re.search(r"tra loi bang tieng viet|tieng viet", normalized):
            prefs["language"] = "Vietnamese"
        if re.search(r"ngan|ngan gon", normalized):
            prefs["answer_style"] = "concise"
        if re.search(r"code don gian|style code don gian", normalized):
            prefs["code_style"] = "simple runnable code"
        if "thich python" in normalized:
            prefs["preferred_language"] = "Python"
        if "khong thich java" in normalized:
            prefs["avoid_language"] = "Java"
        if "redis cho cache" in normalized or "dung redis cho cache" in normalized:
            facts["project_cache_backend"] = "Redis"
        if "khong muon luu pii" in normalized or "khong luu pii" in normalized:
            prefs["privacy"] = "do not persist PII unless the user explicitly consents"
        if "benchmark co bang ro rang" in normalized:
            prefs["benchmark_format"] = "clear table"
        if "thieu semantic retrieval test" in normalized:
            facts["benchmark_missing_group"] = "semantic retrieval test"

        return {"preferences": prefs, "facts": facts, "conflicts": conflicts}

    def _extract_with_llm(self, raw: str) -> Dict[str, Any]:
        if not self.use_llm_extraction:
            return {"preferences": {}, "facts": {}, "conflicts": []}

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        prompt = f"""Extract durable memory from the conversation below.
Store only useful long-term preferences or stable facts. Do not store sensitive
PII unless the user explicitly asked the assistant to remember it.

Return strict JSON only:
{{
  "preferences": {{"language": "...", "answer_style": "..."}},
  "facts": {{"allergy": "...", "project_cache_backend": "..."}},
  "conflicts": [
    {{"field": "allergy", "old_value": "...", "new_value": "...", "resolution": "..."}}
  ]
}}

Conversation:
{raw}
"""
        try:
            text = llm.invoke(prompt).content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            return {
                "preferences": data.get("preferences", {}) or {},
                "facts": data.get("facts", {}) or {},
                "conflicts": data.get("conflicts", []) or [],
            }
        except Exception as exc:
            return {
                "preferences": {},
                "facts": {},
                "conflicts": [{"field": "llm_extraction", "resolution": f"parse_failed: {exc.__class__.__name__}"}],
            }

    def extract_and_save(self, user_id: str, messages: List[BaseMessage]) -> Dict[str, Any]:
        formatter = ShortTermMemory(max_tokens=999999)
        raw = formatter._format_messages(messages)

        llm_data = self._extract_with_llm(raw)
        deterministic = self._extract_deterministic(raw)

        prefs = {**llm_data.get("preferences", {}), **deterministic.get("preferences", {})}
        facts = {**llm_data.get("facts", {}), **deterministic.get("facts", {})}
        conflicts = [*llm_data.get("conflicts", []), *deterministic.get("conflicts", [])]

        self.save_preferences(user_id, prefs)
        self.save_facts(user_id, facts)
        return {"preferences": prefs, "facts": facts, "conflicts": conflicts, "backend": self.backend}
