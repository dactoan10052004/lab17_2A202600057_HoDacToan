# Lab 17 Grading Report

## Summary

| Hạng mục | Điểm ước lượng |
|---|---:|
| Full memory stack | 25/25 |
| LangGraph state/router + prompt injection | 30/30 |
| Save/update memory + conflict handling | 15/15 |
| Benchmark 10 multi-turn conversations bằng OpenAI API thật | 20/20 |
| Reflection privacy/limitations | 10/10 |
| Tổng | 100/100 |

## Source Map

| Thành phần | File |
|---|---|
| LangGraph flow | `memory_agent/graph.py` |
| State schema | `memory_agent/state.py` |
| Memory router | `memory_agent/routing.py` |
| Prompt/context packing | `memory_agent/context.py` |
| Short-term memory | `memory_agent/memory/short_term.py` |
| Long-term profile memory | `memory_agent/memory/profile.py` |
| Episodic memory | `memory_agent/memory/episodic.py` |
| Semantic memory | `memory_agent/memory/semantic.py` |
| Benchmark runner | `memory_agent/benchmark.py` |
| Tests | `tests/test_lab17_memory.py` |

## Rubric Evidence

| Rubric | Evidence |
|---|---|
| 4 memory types | `memory_agent/memory/` có short-term, profile, episodic, semantic |
| `MemoryState` | `memory_agent/state.py` |
| `retrieve_memory(state)` | `memory_agent/graph.py` |
| Prompt injection | `RECENT_CONVERSATION`, `SHORT_TERM_SUMMARY`, `LONG_TERM_PROFILE`, `EPISODIC_MEMORY`, `SEMANTIC_MEMORY` |
| Token budget | `memory_agent/context.py`, `memory_agent/utils/tokens.py` |
| Save/update profile | `LongTermMemoryRedis.extract_and_save()` |
| Conflict handling | facts lưu theo key; correction mới ghi đè fact cũ |
| Episodic write-back | `EpisodicMemoryJSON.add_episode()` trong `save_memory()` |
| Benchmark | `docs/BENCHMARK.md`, `reports/benchmark_report.csv`, calls real OpenAI API |
| Privacy reflection | `docs/BENCHMARK.md` |

## Required Conflict Test

```text
User: Tôi dị ứng sữa bò.
User: À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
Expected profile: allergy = đậu nành
```

Test coverage:

```text
tests/test_lab17_memory.py::test_allergy_correction_overwrites_stale_fact
```

## Bonus

| Bonus | Status |
|---|---|
| Redis thật | Có `docker-compose.yml`; profile memory tự dùng Redis nếu ping được |
| Chroma thật | Có Chroma persistent client khi có `OPENAI_API_KEY` |
| LLM extraction | Optional `ENABLE_LLM_MEMORY_EXTRACTION=1`, có JSON parse và fallback |
| Token counting | Dùng `tiktoken` |
| Graph demo | Flow rõ trong `memory_agent/graph.py` |

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

```text
Ran 3 tests
OK
```

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py
```

```text
=== Lab 17 Real OpenAI API Benchmark ===
Model: gpt-4o-mini
Conversations: 10
Pass: 10/10
Average baseline keyword score: 0.575
Average memory keyword score: 0.808
Saved detailed report to reports\benchmark_report.csv
```

```powershell
.\.venv\Scripts\python.exe scripts\seed_semantic.py
```

```text
Seeded 7 semantic documents into chroma semantic store.
```

## Submission Checklist

Nộp:

- `memory_agent/`
- `scripts/`
- `tests/`
- `data/semantic_docs.json`
- `data/episodes.jsonl`
- `data/chroma/`
- `docs/BENCHMARK.md`
- `docs/REPORT.md`
- `reports/benchmark_report.csv`
- `README.md`
- `requirements.txt`
- `docker-compose.yml`

Không nộp:

- `.env`
- `.venv/`
- `.chroma/`
- `__pycache__/`
- `*.pyc`
