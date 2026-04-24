from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from memory_agent.baseline import run_baseline_turn
from memory_agent.config import OPENAI_API_KEY, OPENAI_MODEL, REPORTS_DIR
from memory_agent.graph import episodic_memory, long_memory, run_turn
from memory_agent.utils.tokens import count_tokens


BENCHMARK_CASES = [
    {
        "id": 1,
        "category": "profile recall",
        "scenario": "Recall preferred coding language after filler turns",
        "turns": [
            "Tôi thích Python, không thích Java. Hãy nhớ style code đơn giản.",
            "Giải thích list comprehension thật ngắn.",
            "Cho ví dụ đọc file text.",
            "Tôi nên dùng ngôn ngữ nào cho script nhỏ theo sở thích của tôi?",
        ],
        "expected_keywords": ["Python"],
    },
    {
        "id": 2,
        "category": "conflict update",
        "scenario": "Allergy correction overwrites older fact",
        "turns": [
            "Tôi dị ứng sữa bò.",
            "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
            "Khi gợi ý món ăn, tôi cần tránh gì?",
        ],
        "expected_keywords": ["đậu nành"],
    },
    {
        "id": 3,
        "category": "episodic recall",
        "scenario": "Reuse previous debug lesson",
        "turns": [
            "Lần trước tôi debug API timeout bằng cách tăng timeout nhưng không hiệu quả.",
            "Kết quả đúng là kiểm tra DNS/service name trong Docker network.",
            "API lại timeout, nên thử hướng nào trước?",
        ],
        "expected_keywords": ["DNS", "service", "Docker"],
    },
    {
        "id": 4,
        "category": "semantic retrieval",
        "scenario": "Retrieve definition of semantic memory",
        "turns": [
            "Tôi đang học memory stack cho agent.",
            "Tài liệu lab nói gì về semantic memory?",
            "Giải thích semantic memory và backend phù hợp.",
        ],
        "expected_keywords": ["semantic", "knowledge", "Chroma"],
    },
    {
        "id": 5,
        "category": "profile recall",
        "scenario": "Project cache backend preference",
        "turns": [
            "Nhớ rằng project của tôi dùng Redis cho cache.",
            "Tôi cũng muốn TTL cho dữ liệu nhạy cảm.",
            "Long-term memory nên chọn backend nào cho project này?",
        ],
        "expected_keywords": ["Redis", "TTL"],
    },
    {
        "id": 6,
        "category": "trim/token budget",
        "scenario": "Important preference survives noisy context",
        "turns": [
            "Tôi muốn câu trả lời tiếng Việt, ngắn, có code chạy được.",
            "Filler: nói về kiến trúc agent trong nhiều đoạn dài.",
            "Filler: thêm log, metrics, retry, tracing, deployment.",
            "Hãy viết helper tính token theo đúng style tôi muốn.",
        ],
        "expected_keywords": ["code", "token"],
    },
    {
        "id": 7,
        "category": "episodic recall",
        "scenario": "Avoid repeated failed retrieval strategy",
        "turns": [
            "Tôi đã thử sliding window và bị mất context cũ quan trọng.",
            "Bài học: phải summary phần cũ trước khi trim.",
            "Làm sao tránh mất context khi conversation dài?",
        ],
        "expected_keywords": ["tóm tắt", "context"],
    },
    {
        "id": 8,
        "category": "semantic retrieval",
        "scenario": "Retrieve privacy design chunk",
        "turns": [
            "Tôi không muốn lưu PII trừ khi user đồng ý.",
            "Agent có memory write-back sau mỗi turn.",
            "Thiết kế privacy cho memory write-back thế nào?",
        ],
        "expected_keywords": ["Chấp thuận", "TTL", "xóa"],
    },
    {
        "id": 9,
        "category": "router coverage",
        "scenario": "Combine profile, episodic, and semantic memory",
        "turns": [
            "Tôi thích benchmark có bảng rõ ràng.",
            "Lần trước report bị thiếu semantic retrieval test.",
            "Tài liệu nói context priority là gì?",
            "Hãy đề xuất bảng benchmark cho lab của tôi.",
        ],
        "expected_keywords": ["profile", "conflict", "semantic", "trim"],
    },
    {
        "id": 10,
        "category": "deletion/TTL",
        "scenario": "User requests memory deletion plan",
        "turns": [
            "Tôi có preference trả lời ngắn.",
            "Agent đang lưu profile trong Redis/JSON, episode trong JSONL, semantic trong Chroma.",
            "Nếu tôi yêu cầu xóa memory, phải xóa ở đâu?",
        ],
        "expected_keywords": ["Redis", "JSONL", "Chroma", "short-term"],
    },
]


def require_openai_api_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required. This benchmark uses the real OpenAI API, not a mock LLM.")


def compact(text: str, limit: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit]


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return hits / max(1, len(expected_keywords))


def run_baseline_conversation(turns: list[str]) -> tuple[str, list[BaseMessage]]:
    history: list[BaseMessage] = []
    answer = ""
    for turn in turns:
        answer, history = run_baseline_turn(turn, history=history)
    return answer, history


def run_memory_conversation(turns: list[str], user_id: str) -> tuple[str, list[BaseMessage], dict[str, Any]]:
    long_memory.delete_user_memory(user_id)
    episodic_memory.delete_user_memory(user_id)

    history: list[BaseMessage] = []
    state: dict[str, Any] = {}
    answer = ""
    for turn in turns:
        answer, history, state = run_turn(turn, history=history, user_id=user_id)
    return answer, history, state


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in BENCHMARK_CASES:
        turns = "\n".join(case["turns"])
        expected_keywords = list(case["expected_keywords"])

        baseline_answer, baseline_history = run_baseline_conversation(case["turns"])
        memory_answer, memory_history, memory_state = run_memory_conversation(
            case["turns"],
            user_id=f"benchmark_case_{case['id']}",
        )

        baseline_score = keyword_score(baseline_answer, expected_keywords)
        memory_score = keyword_score(memory_answer, expected_keywords)

        rows.append(
            {
                "conversation_id": case["id"],
                "category": case["category"],
                "scenario": case["scenario"],
                "turn_count": len(case["turns"]),
                "model": OPENAI_MODEL,
                "input_tokens_estimate": count_tokens(turns, OPENAI_MODEL),
                "baseline_final_answer_tokens": count_tokens(baseline_answer, OPENAI_MODEL),
                "memory_final_answer_tokens": count_tokens(memory_answer, OPENAI_MODEL),
                "memory_context_tokens": count_tokens(str(memory_state.get("memory_context", "")), OPENAI_MODEL),
                "baseline_keyword_score": round(baseline_score, 3),
                "memory_keyword_score": round(memory_score, 3),
                "pass": "Pass" if memory_score >= 0.5 else "Review",
                "route": str(memory_state.get("route", {})),
                "no_memory_result": compact(baseline_answer),
                "with_memory_result": compact(memory_answer),
                "memory_context_preview": compact(str(memory_state.get("memory_context", ""))),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    require_openai_api_key()
    rows = build_rows()
    output_path = REPORTS_DIR / "benchmark_report.csv"
    write_csv(rows, output_path)

    passed = sum(1 for row in rows if row["pass"] == "Pass")
    avg_baseline = sum(float(row["baseline_keyword_score"]) for row in rows) / len(rows)
    avg_memory = sum(float(row["memory_keyword_score"]) for row in rows) / len(rows)

    print("=== Lab 17 Real OpenAI API Benchmark ===")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Conversations: {len(rows)}")
    print(f"Pass: {passed}/{len(rows)}")
    print(f"Average baseline keyword score: {avg_baseline:.3f}")
    print(f"Average memory keyword score: {avg_memory:.3f}")
    print(f"Saved detailed report to {output_path}")


if __name__ == "__main__":
    main()
