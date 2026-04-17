# So sánh 3 RAG

- Mode: `controlled_no_fusion`
- Split: `all`
- Run Label: `final_controlled_no_fusion_all72`
- Seed: `0`
- Token Budget: `enforced` | max_output_tokens=`1024`
- Retrieval Budget: top_n=`6`, k=`3`
- Latency Budget: `25000.0` ms

| Hạng | Hệ thống | Profile | Overall | Answer | Retrieval | Faithfulness | MRR | Semantic | Latency (ms) | Budget Violation | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | shared_no_fusion_v1 | 0.7487 | 0.6235 | 0.6886 | 0.7799 | 0.6256 | 0.8066 | 12888.25 | 0.0139 | 0 |
| 2 | hybrid | shared_no_fusion_v1 | 0.7440 | 0.5875 | 0.6911 | 0.7900 | 0.6508 | 0.7674 | 13075.95 | 0.0000 | 0 |
| 3 | graphrag | shared_no_fusion_v1 | 0.6475 | 0.5346 | 0.6415 | 0.5817 | 0.5421 | 0.7120 | 12645.32 | 0.0000 | 0 |

## Breakdown Theo Nhóm Câu Hỏi

### Dense Shared

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | shared_no_fusion_v1 | 24 | 0.7708 | 0.9778 | 0.7778 | 0.6966 | 0.5329 |
| hybrid | shared_no_fusion_v1 | 24 | 0.7472 | 0.9722 | 0.7431 | 0.6572 | 0.5319 |
| graphrag | shared_no_fusion_v1 | 24 | 0.6499 | 0.9640 | 0.5486 | 0.5695 | 0.4968 |

### Graph

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_no_fusion_v1 | 24 | 0.8346 | 0.9714 | 0.9033 | 0.7441 | 0.5969 |
| baseline | shared_no_fusion_v1 | 24 | 0.8327 | 0.9810 | 0.8953 | 0.7438 | 0.5873 |
| graphrag | shared_no_fusion_v1 | 24 | 0.7519 | 0.9737 | 0.6964 | 0.7253 | 0.5559 |

### Neutral

| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | shared_no_fusion_v1 | 24 | 0.6502 | 0.6994 | 0.7236 | 0.6048 | 0.4730 |
| baseline | shared_no_fusion_v1 | 24 | 0.6426 | 0.7001 | 0.6667 | 0.6499 | 0.4784 |
| graphrag | shared_no_fusion_v1 | 24 | 0.5408 | 0.6842 | 0.5000 | 0.5269 | 0.4205 |
