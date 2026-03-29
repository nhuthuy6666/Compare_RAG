# So sánh 3 RAG

- Mode: `controlled_with_fusion`
- Split: `all`
- Run Label: `shared_fusion_all`
- Seed: `0`
- Token Budget: `enforced` | max_output_tokens=`1024`
- Retrieval Budget: top_n=`6`, k=`3`
- Latency Budget: `25000.0` ms

| Hạng | Hệ thống | Profile | Overall | Answer | Retrieval | Faithfulness | MRR | Semantic | Latency (ms) | Budget Violation | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | shared_locked_v1 | 0.7334 | 0.6356 | 0.6542 | 0.7714 | 0.6102 | 0.7909 | 25024.62 | 0.3889 | 0 |
| 2 | hybrid | shared_locked_v1 | 0.7191 | 0.6308 | 0.6452 | 0.7593 | 0.6374 | 0.7739 | 21586.77 | 0.1528 | 0 |
| 3 | graphrag | shared_locked_v1 | 0.6379 | 0.5471 | 0.6099 | 0.5913 | 0.5573 | 0.7146 | 20623.69 | 0.0556 | 0 |

## Breakdown Theo Nhóm Câu Hỏi

### Dense Shared

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_locked_v1 | 24 | 0.7228 | 0.9573 | 0.7361 | 0.6452 | 0.4306 |
| baseline | shared_locked_v1 | 24 | 0.7218 | 0.9601 | 0.7236 | 0.6392 | 0.4585 |
| graphrag | shared_locked_v1 | 24 | 0.6053 | 0.9353 | 0.4806 | 0.5609 | 0.4203 |

### Graph

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_locked_v1 | 24 | 0.8367 | 0.9515 | 0.9375 | 0.7997 | 0.4721 |
| baseline | shared_locked_v1 | 24 | 0.8366 | 0.9675 | 0.9031 | 0.8062 | 0.5142 |
| graphrag | shared_locked_v1 | 24 | 0.7629 | 0.9557 | 0.7656 | 0.7384 | 0.4757 |

### Neutral

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_locked_v1 | 24 | 0.6416 | 0.6976 | 0.6875 | 0.6384 | 0.4467 |
| hybrid | shared_locked_v1 | 24 | 0.5977 | 0.6983 | 0.6042 | 0.5994 | 0.4124 |
| graphrag | shared_locked_v1 | 24 | 0.5456 | 0.6799 | 0.5278 | 0.5348 | 0.3812 |
