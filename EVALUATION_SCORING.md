# Evaluation Scoring And Run Guide

File này tóm tắt:

- cách project tính điểm evaluation
- 2 phương pháp đọc kết quả
- cây thư mục kết quả mới
- câu lệnh để tự chạy từng đánh giá, kể cả ground truth

## 1. Cây thư mục kết quả mới

Từ bây giờ kết quả mặc định được ghi vào `evaluation/results/`:

- `evaluation/results/3_rag_no_fusion/`
- `evaluation/results/3_rag_with_fusion/`
- `evaluation/results/3_rag_best_tuned/`
- `evaluation/results/ground_truth_baseline_no_fusion/`

Mỗi lần chạy `evaluate-v1.py` cho 3 RAG sẽ sinh:

- `comparison.md`
- `retrieval_answer_quality_evaluation.md`
- `system_performance_evaluation.md`

Ground-truth baseline sinh:

- `report.md`

## 2. Hai phương pháp đọc kết quả

### 2.1. Retrieval + Answer Quality Evaluation

Phương pháp này tập trung vào chất lượng truy hồi và chất lượng câu trả lời.

Các cột chính nên đọc:

- `Overall`
- `Answer`
- `Retrieval`
- `Faithfulness`
- `MRR`
- `Semantic`

File tương ứng:

- `evaluation/results/3_rag_no_fusion/retrieval_answer_quality_evaluation.md`
- `evaluation/results/3_rag_with_fusion/retrieval_answer_quality_evaluation.md`
- `evaluation/results/3_rag_best_tuned/retrieval_answer_quality_evaluation.md`

### 2.2. System Performance Evaluation

Phương pháp này tập trung vào hiệu năng hệ thống.

Các cột chính nên đọc:

- `Latency (ms)`
- `Budget Violation`
- `Errors`

File tương ứng:

- `evaluation/results/3_rag_no_fusion/system_performance_evaluation.md`
- `evaluation/results/3_rag_with_fusion/system_performance_evaluation.md`
- `evaluation/results/3_rag_best_tuned/system_performance_evaluation.md`

## 3. Công thức `overall_score`

Ở mức từng câu hỏi, `overall_score` được tính như sau:

```text
Overall = 0.25 * Recall@k
        + 0.35 * Faithfulness
        + 0.25 * Answer_Relevancy
        + 0.15 * Context_Precision
```

Trong project hiện tại:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`

## 4. 4 thành phần đi trực tiếp vào `overall`

### 4.1. `Recall@k`

```text
Recall@k = min(1.0, sum(relevance_score(top_k)) / total_relevant)
```

Ý nghĩa:

- đo lượng evidence liên quan mà hệ retrieve được trong top-`k`

### 4.2. `Faithfulness`

```text
Faithfulness = supported_claims / total_claims
```

Ý nghĩa:

- đo mức độ các claim trong câu trả lời có được source hỗ trợ hay không

### 4.3. `Answer_Relevancy`

```text
Answer_Relevance
= 0.35 * cosine(question, answer)
+ 0.40 * cosine(reference_answer, answer)
+ 0.25 * keyword_coverage(answer, answer_keywords)
```

Ý nghĩa:

- đo câu trả lời có bám câu hỏi và đáp án mong đợi hay không

### 4.4. `Context_Precision`

```text
Context_Precision = Precision@k = mean(relevance_score(top_k))
```

Ý nghĩa:

- đo độ liên quan trung bình của các source trong top-`k`

## 5. `relevance_score` của từng source

Project chấm retrieval theo nội dung source, không dựa chủ yếu vào tên file hay `source_hint`.

Nếu bài có số liệu kỳ vọng:

```text
relevance_score
= 0.30 * semantic_reference
+ 0.20 * semantic_question
+ 0.25 * context_coverage
+ 0.10 * answer_coverage
+ 0.15 * numeric_coverage
```

Nếu bài không có số liệu kỳ vọng:

```text
relevance_score
= 0.40 * semantic_reference
+ 0.25 * semantic_question
+ 0.20 * context_coverage
+ 0.15 * answer_coverage
```

## 6. Metric phụ vẫn được ghi ra

Ngoài `overall_score`, evaluator vẫn ghi lại:

- Generation: `answer_quality`, `exact_match`, `token_f1`, `char_similarity`, `semantic_similarity`, `keyword_coverage`
- RAG: `answer_relevance`, `context_relevance`, `faithfulness`, `hallucination_rate`
- Retrieval: `precision@1/3/5`, `recall@1/3/5`, `f1@1/3/5`, `hit@3`, `mrr`, `map`, `ndcg@3/5`
- System: `latency_ms`, `error`, `source_count`

## 7. `answer_quality` và `retrieval_quality`

### 7.1. `answer_quality`

Với câu hỏi thường:

```text
answer_quality
= 0.05 * exact_match
+ 0.10 * token_f1
+ 0.10 * char_similarity
+ 0.35 * semantic_similarity
+ 0.25 * keyword_coverage
+ 0.15 * answer_relevance
```

Nếu hệ từ chối sai khi không nên từ chối, điểm này còn bị nhân `0.25`.

Với câu hỏi mong đợi từ chối:

```text
answer_quality = 0.6 * refusal_correct + 0.4 * answer_relevance
```

### 7.2. `retrieval_quality`

Với câu hỏi thường:

```text
retrieval_quality
= 0.20 * context_relevance
+ 0.15 * precision@3
+ 0.20 * recall@3
+ 0.10 * f1@3
+ 0.10 * hit@3
+ 0.10 * mrr
+ 0.05 * map
+ 0.10 * ndcg@3
```

Với câu hỏi mong đợi từ chối:

```text
retrieval_quality = hit@3
```

## 8. Câu lệnh chạy từng đánh giá

Chạy từ root project.

### 8.1. 3 RAG no fusion

```powershell
python evaluation/evaluate-v1.py --system all --mode controlled_no_fusion --split all
```

Kết quả mặc định:

- `evaluation/results/3_rag_no_fusion/comparison.md`
- `evaluation/results/3_rag_no_fusion/retrieval_answer_quality_evaluation.md`
- `evaluation/results/3_rag_no_fusion/system_performance_evaluation.md`

### 8.2. 3 RAG with fusion

```powershell
python evaluation/evaluate-v1.py --system all --mode controlled_with_fusion --split all
```

Kết quả mặc định:

- `evaluation/results/3_rag_with_fusion/comparison.md`
- `evaluation/results/3_rag_with_fusion/retrieval_answer_quality_evaluation.md`
- `evaluation/results/3_rag_with_fusion/system_performance_evaluation.md`

### 8.3. 3 RAG best tuned

```powershell
python evaluation/evaluate-v1.py --system all --mode best_tuned --split all
```

Kết quả mặc định:

- `evaluation/results/3_rag_best_tuned/comparison.md`
- `evaluation/results/3_rag_best_tuned/retrieval_answer_quality_evaluation.md`
- `evaluation/results/3_rag_best_tuned/system_performance_evaluation.md`

### 8.4. Chạy nhanh cả 2 shared profile: with fusion và no fusion

```powershell
python evaluation/scripts/run_shared_profile_comparisons.py --split all
```

Kết quả mặc định:

- `evaluation/results/3_rag_with_fusion/`
- `evaluation/results/3_rag_no_fusion/`

### 8.5. Ground-truth baseline no fusion

```powershell
python evaluation/ground_truth_baseline_no_fusion.py --split all
```

Kết quả mặc định:

- `evaluation/results/ground_truth_baseline_no_fusion/report.md`

## 9. Nếu muốn đổi thư mục kết quả

Bạn vẫn có thể override bằng `--results-dir`, ví dụ:

```powershell
python evaluation/evaluate-v1.py --system all --mode best_tuned --split dev --results-dir evaluation/results/tmp_best_tuned_dev
```
