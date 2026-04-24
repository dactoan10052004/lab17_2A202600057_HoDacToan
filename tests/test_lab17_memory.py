from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage

from memory_agent.context import format_memory_context
from memory_agent.memory.profile import LongTermMemoryRedis
from memory_agent.memory.semantic import SemanticMemoryChroma
from memory_agent.utils.tokens import count_tokens


class LongTermProfileTests(unittest.TestCase):
    def test_allergy_correction_overwrites_stale_fact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            memory = LongTermMemoryRedis(
                url="redis://localhost:1/0",
                fallback_path=Path(temp_dir) / "profiles.json",
                use_llm_extraction=False,
            )

            result = memory.extract_and_save(
                "rubric_user",
                [
                    HumanMessage(content="Tôi dị ứng sữa bò."),
                    HumanMessage(content="À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò."),
                ],
            )

            profile = memory.load_profile("rubric_user")
            self.assertEqual(result["facts"]["allergy"], "đậu nành")
            self.assertEqual(profile["facts"]["allergy"], "đậu nành")
            self.assertIn("latest_explicit_correction_wins", result["conflicts"][0]["resolution"])


class ContextPackingTests(unittest.TestCase):
    def test_context_has_required_sections_and_respects_budget(self) -> None:
        context = format_memory_context(
            summary="Earlier turns established concise Vietnamese answers.",
            user_profile={"preferences": {"language": "Vietnamese"}, "facts": {"allergy": "đậu nành"}},
            episodes=[
                {
                    "task": "API timeout",
                    "outcome": "Docker service name was wrong.",
                    "reflection": "Check DNS/service name before increasing timeout.",
                }
            ],
            semantic_hits=["[source=lab17_seed] Semantic memory stores domain knowledge chunks."],
            recent_conversation="user: Hãy tư vấn theo memory của tôi.",
            budget_tokens=180,
        )

        self.assertIn("## RECENT_CONVERSATION", context)
        self.assertIn("## LONG_TERM_PROFILE", context)
        self.assertIn("## EPISODIC_MEMORY", context)
        self.assertIn("## SEMANTIC_MEMORY", context)
        self.assertLessEqual(count_tokens(context), 220)


class SemanticFallbackTests(unittest.TestCase):
    def test_keyword_semantic_backend_returns_seed_chunk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            memory = SemanticMemoryChroma(fallback_path=Path(temp_dir) / "semantic_docs.json")

            hits = memory.query("semantic memory vector database chunks", k=2)

            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(any("Semantic memory" in hit for hit in hits))


if __name__ == "__main__":
    unittest.main()
