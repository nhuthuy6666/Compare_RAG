# Metrics V1

V1 giữ lại toàn bộ metric chi tiết cũ, nhưng đổi công thức xếp hạng cuối cùng sang bộ trọng số mới.

## Bộ test V1 cân bằng

Bộ test V1 hiện có `72` câu và được chia đều thành `3` nhóm:

- `24` câu `dense_shared`: nghiêng về thế mạnh chung của `rag` và `hybrid_rag`
- `24` câu `graph`: nghiêng về thế mạnh kiến trúc của `graph_rag`
- `24` câu `neutral`: nhóm trung lập để tránh benchmark chỉ thiên về một kiến trúc

Evaluator sẽ ghi thêm breakdown theo từng nhóm này vào `evaluation/results_v1/strength_breakdown.csv`.

## Thiết lập `k`

Benchmark hiện dùng:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`

## Công thức Overall V1

`Overall = 0.25 * Recall@k + 0.35 * Faithfulness + 0.25 * Answer_Relevancy + 0.15 * Context_Precision`

## Ý nghĩa từng thành phần

### Recall@k

`Recall@k = relevant_in_top_k / total_relevant`

Ý nghĩa: trong tổng số source relevant kỳ vọng, hệ retrieve được bao nhiêu source trong top-`k`.

### Faithfulness

`Faithfulness = supported_claims / total_claims`

Ý nghĩa: trong các claim của câu trả lời, có bao nhiêu claim được source hỗ trợ.

### Answer_Relevancy

`AnswerRelevance = 0.7 * cosine(question, answer) + 0.3 * keyword_coverage(answer, answer_keywords)`

Ý nghĩa: câu trả lời có bám đúng trọng tâm câu hỏi hay không.

### Context_Precision

`Context_Precision = Precision@k = relevant_in_top_k / k`

Ý nghĩa: trong top-`k` context retrieve, tỷ lệ context relevant là bao nhiêu.

## Metric chi tiết vẫn được giữ

Ngoài `overall_score`, V1 vẫn ghi lại:

- Generation: `exact_match`, `token_f1`, `char_similarity`, `semantic_similarity`, `keyword_coverage`
- RAG: `answer_relevance`, `context_relevance`, `faithfulness`, `hallucination_rate`
- Retrieval: `precision@1/3/5`, `recall@1/3/5`, `f1@1/3/5`, `hit@3`, `mrr`, `map`, `ndcg@3/5`
- System: `latency_ms`, `error`, `source_count`

## Thư mục kết quả

- Output gốc của từng hệ: `evaluation/outputs_v1/`
- Metric và báo cáo tổng hợp: `evaluation/results_v1/`
- Breakdown theo nhóm sức mạnh: `evaluation/results_v1/strength_breakdown.csv`
