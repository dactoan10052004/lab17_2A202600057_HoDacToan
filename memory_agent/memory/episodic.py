from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

from memory_agent.config import EPISODE_LOG_PATH


class EpisodicMemoryJSON:
    """JSONL backend lưu past trajectories.

    Mỗi episode:
    {
      "task": "...",
      "trajectory": "...",
      "outcome": "...",
      "reflection": "...",
      "created_at": 123,
      "used": 0
    }
    """

    def __init__(self, path: Path = EPISODE_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

    def add_episode(self, task: str, trajectory: str, outcome: str, reflection: str, user_id: str = "demo_user") -> None:
        ep = {
            "user_id": user_id,
            "task": task,
            "trajectory": trajectory,
            "outcome": outcome,
            "reflection": reflection,
            "created_at": time.time(),
            "used": 0,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        episodes = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    def search(self, query: str, k: int = 3, user_id: str | None = None) -> List[Dict[str, Any]]:
        episodes = self._load()
        if user_id is not None:
            episodes = [ep for ep in episodes if ep.get("user_id", user_id) == user_id]
        now = time.time()

        def score(ep: Dict[str, Any]) -> float:
            text = f"{ep.get('task', '')} {ep.get('reflection', '')} {ep.get('outcome', '')}"
            sim = SequenceMatcher(None, query.lower(), text.lower()).ratio()
            age_days = max(0.0, (now - ep.get("created_at", now)) / 86400)
            decay = 1 / (1 + 0.03 * age_days)
            importance = 1 + 0.05 * ep.get("used", 0)
            return sim * decay * importance

        ranked = sorted(episodes, key=score, reverse=True)[:k]
        return ranked

    def delete_user_memory(self, user_id: str) -> None:
        episodes = [ep for ep in self._load() if ep.get("user_id") != user_id]
        with self.path.open("w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    def consolidate(self, max_items: int = 200) -> None:
        """LRU-ish cleanup: giữ recent/used episodes nếu log quá dài."""
        episodes = self._load()
        if len(episodes) <= max_items:
            return
        episodes = sorted(
            episodes,
            key=lambda e: (e.get("used", 0), e.get("created_at", 0)),
            reverse=True,
        )[:max_items]
        with self.path.open("w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
