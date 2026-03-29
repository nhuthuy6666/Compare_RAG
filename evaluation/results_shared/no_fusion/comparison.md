# So sánh 3 RAG

- Mode: `controlled_no_fusion`
- Split: `all`
- Run Label: `shared_no_fusion_all`
- Seed: `0`
- Token Budget: `enforced` | max_output_tokens=`1024`
- Retrieval Budget: top_n=`6`, k=`3`
- Latency Budget: `25000.0` ms

| Hạng | Hệ thống | Profile | Overall | Answer | Retrieval | Faithfulness | MRR | Semantic | Latency (ms) | Budget Violation | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | shared_no_fusion_v1 | 0.7487 | 0.6235 | 0.6886 | 0.7799 | 0.6256 | 0.8066 | 14096.10 | 0.0139 | 0 |
| 2 | hybrid | shared_no_fusion_v1 | 0.7432 | 0.5800 | 0.6911 | 0.7877 | 0.6508 | 0.7674 | 14733.18 | 0.0000 | 0 |
| 3 | graphrag | shared_no_fusion_v1 | 0.6419 | 0.5267 | 0.6547 | 0.5659 | 0.5703 | 0.7091 | 12367.79 | 0.0000 | 0 |

## Breakdown Theo Nhóm Câu Hỏi

### Dense Shared

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_no_fusion_v1 | 24 | 0.7708 | 0.9778 | 0.7778 | 0.6966 | 0.5329 |
| hybrid | shared_no_fusion_v1 | 24 | 0.7448 | 0.9722 | 0.7361 | 0.6573 | 0.5319 |
| graphrag | shared_no_fusion_v1 | 24 | 0.6115 | 0.9667 | 0.4528 | 0.5337 | 0.5198 |

### Graph

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_no_fusion_v1 | 24 | 0.8346 | 0.9714 | 0.9033 | 0.7441 | 0.5969 |
| baseline | shared_no_fusion_v1 | 24 | 0.8327 | 0.9810 | 0.8953 | 0.7438 | 0.5873 |
| graphrag | shared_no_fusion_v1 | 24 | 0.7923 | 0.9779 | 0.7865 | 0.7469 | 0.5723 |

### Neutral

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_no_fusion_v1 | 24 | 0.6502 | 0.6994 | 0.7236 | 0.6048 | 0.4730 |
| baseline | shared_no_fusion_v1 | 24 | 0.6426 | 0.7001 | 0.6667 | 0.6499 | 0.4784 |
| graphrag | shared_no_fusion_v1 | 24 | 0.5218 | 0.6842 | 0.4583 | 0.5054 | 0.4265 |
