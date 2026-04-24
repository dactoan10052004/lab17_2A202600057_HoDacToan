# Lab 17 - Multi-Memory Agent With LangGraph

## Mục Tiêu

Agent triển khai full memory stack theo rubric Lab 17:

- Short-term memory: sliding window và summary.
- Long-term profile memory: Redis hash, fallback JSON.
- Episodic memory: JSONL log gồm task, trajectory, outcome, reflection.
- Semantic memory: Chroma vector store, fallback keyword search.
- LangGraph flow: `retrieve_memory -> call_llm -> save_memory`.
- Benchmark: 10 multi-turn conversations, so sánh no-memory và with-memory bằng OpenAI API thật.

## Cấu Trúc

```text
memory_agent/
  graph.py
  state.py
  routing.py
  context.py
  benchmark.py
  cli.py
  memory/
    short_term.py
    profile.py
    episodic.py
    semantic.py
  utils/
    tokens.py
scripts/
  main.py
  benchmark.py
  seed_semantic.py
docs/
  BENCHMARK.md
  REPORT.md
reports/
  benchmark_report.csv
data/
tests/
```

## Cài Đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

OpenAI API key là bắt buộc khi chạy agent hoặc benchmark:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Backend Redis/semantic seed tùy chọn:

```bash
docker compose up -d redis
python scripts/seed_semantic.py
```

Nếu không có Redis, hệ thống dùng JSON profile store. Nếu không dùng OpenAI embedding cho Chroma, semantic memory fallback sang keyword search. LLM trả lời vẫn gọi OpenAI API thật.

## Chạy

```bash
python scripts/main.py
python scripts/benchmark.py
python -m unittest discover -s tests
```

## Tài Liệu Nộp

- `docs/BENCHMARK.md`: 10 benchmark conversations và reflection.
- `docs/REPORT.md`: mapping rubric, bonus, verification.
- `reports/benchmark_report.csv`: output benchmark.

## Conflict Test Bắt Buộc

```text
User: Tôi dị ứng sữa bò.
User: À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
Expected profile: allergy = đậu nành
```

Covered by `tests/test_lab17_memory.py`.
