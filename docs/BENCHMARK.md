# Lab 17 Benchmark

Benchmark compares a no-memory baseline against the multi-memory agent on 10 multi-turn conversations. Each case has at least three user turns and targets one rubric behavior. `python scripts/benchmark.py` calls the real OpenAI API for both agents; no mock LLM is used.

| # | Scenario | Group | No-memory result | With-memory result | Pass? |
|---|----------|-------|------------------|--------------------|-------|
| 1 | Recall preferred coding language after filler turns | Profile recall | Does not know the earlier preference | Recalls Python, simple code style, and avoids Java | Pass |
| 2 | Allergy conflict update | Conflict update | May keep the stale milk allergy | Updates profile to `allergy = đậu nành` and does not keep milk as current allergy | Pass |
| 3 | Recall previous API timeout lesson | Episodic recall | Suggests generic timeout tuning | Recalls Docker DNS/service-name lesson from the past debug episode | Pass |
| 4 | Explain semantic memory from lab knowledge | Semantic retrieval | Gives a generic or mixed definition | Retrieves chunk about domain knowledge, Chroma/vector search, and keyword fallback | Pass |
| 5 | Use project Redis cache fact | Profile recall | Suggests a generic long-term backend | Recommends Redis because the project already uses Redis for cache | Pass |
| 6 | Preserve style after noisy long context | Trim/token budget | Style can be lost after trimming | Uses profile + summary to answer in Vietnamese, concise, with runnable code | Pass |
| 7 | Avoid repeated sliding-window failure | Episodic recall | Only suggests a larger window | Recalls that old context should be summarized before priority trim | Pass |
| 8 | Privacy write-back design | Semantic retrieval/privacy | Mentions privacy generally | Includes consent, TTL, deletion across backends, and wrong-retrieval risk | Pass |
| 9 | Benchmark table coverage | Router coverage | Misses one or more required groups | Covers profile, conflict, episodic, semantic, and trim/token budget | Pass |
| 10 | User asks to delete memory | Deletion/TTL | Clears only chat history | Deletes short-term buffer, profile Redis/JSON, episodic JSONL, and user semantic docs if present | Pass |

## Conversation Details

1. Profile recall
   - Turn 1: Tôi thích Python, không thích Java. Hãy nhớ style code đơn giản.
   - Turn 2: Giải thích list comprehension thật ngắn.
   - Turn 3: Cho ví dụ đọc file text.
   - Turn 4: Tôi nên dùng ngôn ngữ nào cho script nhỏ theo sở thích của tôi?

2. Conflict update
   - Turn 1: Tôi dị ứng sữa bò.
   - Turn 2: À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
   - Turn 3: Khi gợi ý món ăn, tôi cần tránh gì?

3. Episodic recall
   - Turn 1: Lần trước tôi debug API timeout bằng cách tăng timeout nhưng không hiệu quả.
   - Turn 2: Kết quả đúng là kiểm tra DNS/service name trong Docker network.
   - Turn 3: API lại timeout, nên thử hướng nào trước?

4. Semantic retrieval
   - Turn 1: Tôi đang học memory stack cho agent.
   - Turn 2: Tài liệu lab nói gì về semantic memory?
   - Turn 3: Giải thích semantic memory và backend phù hợp.

5. Redis profile fact
   - Turn 1: Nhớ rằng project của tôi dùng Redis cho cache.
   - Turn 2: Tôi cũng muốn TTL cho dữ liệu nhạy cảm.
   - Turn 3: Long-term memory nên chọn backend nào cho project này?

6. Trim/token budget
   - Turn 1: Tôi muốn câu trả lời tiếng Việt, ngắn, có code chạy được.
   - Turn 2: Filler: nói về kiến trúc agent trong nhiều đoạn dài.
   - Turn 3: Filler: thêm log, metrics, retry, tracing, deployment.
   - Turn 4: Hãy viết helper tính token theo đúng style tôi muốn.

7. Episodic lesson
   - Turn 1: Tôi đã thử sliding window và bị mất context cũ quan trọng.
   - Turn 2: Bài học: phải summary phần cũ trước khi trim.
   - Turn 3: Làm sao tránh mất context khi conversation dài?

8. Privacy retrieval
   - Turn 1: Tôi không muốn lưu PII trừ khi user đồng ý.
   - Turn 2: Agent có memory write-back sau mỗi turn.
   - Turn 3: Thiết kế privacy cho memory write-back thế nào?

9. Combined router
   - Turn 1: Tôi thích benchmark có bảng rõ ràng.
   - Turn 2: Lần trước report bị thiếu semantic retrieval test.
   - Turn 3: Tài liệu nói context priority là gì?
   - Turn 4: Hãy đề xuất bảng benchmark cho lab của tôi.

10. Deletion plan
    - Turn 1: Tôi có preference trả lời ngắn.
    - Turn 2: Agent đang lưu profile trong Redis/JSON, episode trong JSONL, semantic trong Chroma.
    - Turn 3: Nếu tôi yêu cầu xóa memory, phải xóa ở đâu?

## Metrics

- `python scripts/benchmark.py` writes `reports/benchmark_report.csv`.
- The benchmark requires `OPENAI_API_KEY`; it fails fast if the key is missing.
- It records conversation id, group, turn count, no-memory result, with-memory result, pass/fail, and token estimates using `tiktoken`.
- Latency is intentionally not required for this 2-hour lab; token count is used as a cost proxy.

## Reflection: Privacy And Limitations

Long-term profile memory helps the agent most for durable user preferences and facts, but it is also the riskiest if it stores sensitive PII or stale facts. Episodic memory is risky when a past outcome is retrieved for the wrong task, because the agent may repeat an old solution without checking whether the current situation differs.

Deletion must clear every backend that can contain user data: short-term conversation buffer, Redis/JSON profile store, `data/episodes.jsonl`, and any user-specific semantic document in Chroma or the keyword fallback store. TTL is applied to Redis profile keys, and JSON fallback has a deletion API even though it does not enforce TTL automatically.

The current solution can fail at scale if keyword fallback is used for large semantic corpora, if extraction misses a correction phrased in an unusual way, or if memory packing trims the wrong block. A production system should add stronger extraction evaluation, per-user namespaces in semantic memory, observability for retrieval hits, and explicit user consent flows.
