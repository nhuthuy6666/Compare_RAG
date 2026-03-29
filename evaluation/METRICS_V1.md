# Metrics V1

V1 giu lai bo metric chi tiet, nhung duoc dieu chinh de giam thien lech theo kieu bieu dien source cua tung kien truc.

## Bo test V1 can bang

Bo test V1 hien co `72` cau va duoc chia deu thanh `3` nhom:

- `24` cau `dense_shared`: nghieng ve the manh chung cua `rag` va `hybrid_rag`
- `24` cau `graph`: nghieng ve the manh kien truc cua `graph_rag`
- `24` cau `neutral`: nhom trung lap de tranh benchmark chi thien ve mot kien truc

Evaluator se ghi them breakdown theo tung nhom nay vao `evaluation/results_v1/strength_breakdown.csv`.

## Thiet lap `k`

Benchmark hien dung:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`

## Cong thuc Overall V1

`Overall = 0.25 * Recall@k + 0.35 * Faithfulness + 0.25 * Answer_Relevancy + 0.15 * Context_Precision`

## Nguyen tac cong bang moi

- Retrieval relevance duoc cham bang noi dung source, khong dua chu yeu vao `source_hint` hay ten file.
- Source co the la chunk dai, doan ngan hoac graph fact; evaluator deu quy ve cung mot relevance score trong `[0, 1]`.
- `source_hint_hit` van duoc ghi lai de debug dataset, nhung khong con la tin hieu chinh de tong hop chat luong retrieval.
- Answer quality uu tien semantic similarity, keyword/fact coverage va answer relevance hon la exact surface form.

## Y nghia tung thanh phan

### Recall@k

`Recall@k = sum(relevance_score(top_k)) / total_relevant`

Y nghia: trong tong evidence ky vong, he retrieve duoc bao nhieu relevance mass trong top-`k`.

### Faithfulness

`Faithfulness = supported_claims / total_claims`

Y nghia: trong cac claim cua cau tra loi, co bao nhieu claim duoc source ho tro.

### Answer_Relevancy

`AnswerRelevance = 0.35 * cosine(question, answer) + 0.40 * cosine(reference_answer, answer) + 0.25 * keyword_coverage(answer, answer_keywords)`

Y nghia: cau tra loi co bam dung cau hoi va dung dap an mong doi, nhung khong bi phat nang chi vi khac wording.

### Context_Precision

`Context_Precision = Precision@k = mean(relevance_score(top_k))`

Y nghia: trong top-`k` context retrieve, muc do lien quan trung binh cao den dau.

## Metric chi tiet van duoc giu

Ngoai `overall_score`, V1 van ghi lai:

- Generation: `exact_match`, `token_f1`, `char_similarity`, `semantic_similarity`, `keyword_coverage`
- RAG: `answer_relevance`, `context_relevance`, `faithfulness`, `hallucination_rate`
- Retrieval: `precision@1/3/5`, `recall@1/3/5`, `f1@1/3/5`, `hit@3`, `mrr`, `map`, `ndcg@3/5`
- System: `latency_ms`, `error`, `source_count`

## Thu muc ket qua

- Output goc cua tung he: `evaluation/outputs_v1/`
- Metric va bao cao tong hop: `evaluation/results_v1/`
- Breakdown theo nhom suc manh: `evaluation/results_v1/strength_breakdown.csv`
