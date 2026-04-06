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
| 1 | baseline | shared_locked_v1 | 0.7420 | 0.5983 | 0.6751 | 0.7981 | 0.6038 | 0.7552 | 28884.10 | 0.8889 | 0 |
| 2 | hybrid | shared_locked_v1 | 0.7324 | 0.6229 | 0.6712 | 0.7697 | 0.6129 | 0.7682 | 22226.33 | 0.1667 | 0 |
| 3 | graphrag | shared_locked_v1 | 0.6367 | 0.5352 | 0.6300 | 0.5756 | 0.5572 | 0.7247 | 22136.25 | 0.1528 | 0 |

## Breakdown Theo Nhóm Câu Hỏi

### Dense Shared

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_locked_v1 | 24 | 0.7307 | 0.9694 | 0.7292 | 0.6392 | 0.4892 |
| hybrid | shared_locked_v1 | 24 | 0.7195 | 0.9598 | 0.7083 | 0.6342 | 0.4870 |
| graphrag | shared_locked_v1 | 24 | 0.6165 | 0.9515 | 0.4861 | 0.5571 | 0.4614 |

### Graph

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_locked_v1 | 24 | 0.8366 | 0.9779 | 0.9222 | 0.7348 | 0.5713 |
| hybrid | shared_locked_v1 | 24 | 0.8339 | 0.9703 | 0.8785 | 0.7921 | 0.5719 |
| graphrag | shared_locked_v1 | 24 | 0.7428 | 0.9567 | 0.6990 | 0.7263 | 0.5162 |

### Neutral

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_locked_v1 | 24 | 0.6586 | 0.6999 | 0.7431 | 0.6115 | 0.4714 |
| hybrid | shared_locked_v1 | 24 | 0.6440 | 0.6989 | 0.7222 | 0.5890 | 0.4617 |
| graphrag | shared_locked_v1 | 24 | 0.5508 | 0.6772 | 0.5417 | 0.5238 | 0.4062 |
