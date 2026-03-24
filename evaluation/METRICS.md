# Hướng dẫn metric

Tài liệu này mô tả công thức và thuật toán chấm cho bộ so sánh 3 RAG.

## 1. Xác định source relevant

Mỗi source retrieve được xem là `relevant = 1` nếu thỏa ít nhất một điều kiện:

- Chuỗi `label/content/metadata` chứa một `expected_source_hint`
- Hoặc chứa đủ `context_keywords`

Ngưỡng keyword:

- `1` nếu chỉ có `1-2` keyword
- `2` nếu có từ `3` keyword trở lên

Ký hiệu:

- `rel_i`: độ phù hợp của source tại vị trí `i`
- `R`: tổng số source relevant được gán nhãn trong ground truth

## 2. Retrieval metrics

### Precision@k

`Precision@k = (sum(rel_i, i=1..k)) / k`

Đo tỷ lệ source relevant trong `top-k`.

### Recall@k

`Recall@k = (sum(rel_i, i=1..k)) / |R|`

Đo tỷ lệ source relevant trong ground truth đã được retrieve.

### F1@k

`F1@k = 2 * Precision@k * Recall@k / (Precision@k + Recall@k)`

### HitRate@k

`HitRate@k = 1` nếu có ít nhất một source relevant trong `top-k`, ngược lại bằng `0`.

### MRR

`MRR = 1 / rank_first_relevant`

Nếu không có source relevant thì `MRR = 0`.

### MAP

`AP = (1 / |R|) * sum(P@i * rel_i)`

Trong đó `P@i` là precision tại vị trí `i`.

### nDCG@k

`DCG@k = sum(rel_i / log2(i + 1))`

`nDCG@k = DCG@k / IDCG@k`

Đo chất lượng xếp hạng có tính đến vị trí.

## 3. Generation metrics

### Exact Match

So sánh câu trả lời và đáp án tham chiếu sau khi normalize text.

### Token Precision / Recall / F1

So khớp token-level giữa câu trả lời và reference:

- `Precision = overlap / answer_tokens`
- `Recall = overlap / reference_tokens`
- `F1 = 2PR / (P + R)`

### Char Similarity

Dùng `SequenceMatcher` để đo mức giống nhau cấp ký tự.

### Semantic Similarity

Dùng embedding `bge-m3:latest` qua Ollama:

`SemanticSimilarity = cosine(embedding(answer), embedding(reference_answer))`

## 4. RAG-specific metrics

### Answer Relevance

Đo mức liên quan giữa câu trả lời và câu hỏi:

`AnswerRelevance = 0.7 * cosine(question, answer) + 0.3 * keyword_coverage(answer, answer_keywords)`

### Context Relevance

Chấm từng source retrieve:

`ContextItemScore = 0.65 * cosine(question, source) + 0.35 * keyword_coverage(source, context_keywords)`

`ContextRelevance = mean(top_3(ContextItemScore))`

### Faithfulness

Thuật toán:

1. Tách câu trả lời thành các `claim` theo dấu câu, xuống dòng, dấu `;`
2. Loại claim quá ngắn
3. Với mỗi claim, tìm source có điểm support cao nhất

Công thức support cho một claim:

`ClaimSupport = 0.6 * cosine(claim, source) + 0.4 * lexical_overlap(claim_terms, source_terms)`

Nếu claim có số liệu:

- Bắt buộc ít nhất một giá trị số trong claim phải xuất hiện trong source

Một claim được xem là `supported` nếu:

- `ClaimSupport >= 0.58`
- Và ràng buộc số liệu được thỏa

Khi đó:

`Faithfulness = supported_claims / total_claims`

### Hallucination Rate

`HallucinationRate = 1 - Faithfulness`

## 5. Tổng hợp điểm

### Answer Quality

Với câu hỏi thông thường:

`AnswerQuality = 0.2*ExactMatch + 0.2*TokenF1 + 0.2*CharSimilarity + 0.2*SemanticSimilarity + 0.2*KeywordCoverage`

Với câu hỏi refusal:

`AnswerQuality = 0.6*RefusalCorrect + 0.4*AnswerRelevance`

### Retrieval Quality

`RetrievalQuality = 0.10*SourceHintHit + 0.10*SourceKeywordCoverage + 0.10*P@3 + 0.10*R@3 + 0.10*F1@3 + 0.15*MRR + 0.15*MAP + 0.10*nDCG@3 + 0.10*nDCG@5`

### Relevance Score

`RelevanceScore = 0.5*AnswerRelevance + 0.5*ContextRelevance`

### Overall Score

`Overall = 0.30*AnswerQuality + 0.20*RetrievalQuality + 0.20*Faithfulness + 0.20*RelevanceScore + 0.10*RefusalCorrect`

## 6. Ghi chú

- Metric lexical tốt cho fact rõ ràng, số liệu, email, số điện thoại
- Metric semantic tốt cho câu hỏi diễn đạt tự do hơn
- Faithfulness hữu ích để bắt hallucination khi model trả lời dài và thêm thông tin ngoài context
- Retrieval metrics hữu ích nhất khi dataset có ground-truth source hint rõ ràng
