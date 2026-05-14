# Evaluation Layout

Folder `evaluation/` is organized by responsibility:

- `dataset/`: benchmark questions, splits, and loaders
- `metrics/`: metric implementations
- `runners/`: adapters that call each RAG system
- `reporting/`: report rendering and result visualization scripts
- `analysis/`: tuning, repeated studies, and human-judgment helpers
- `results/`: current evaluation outputs for each mode

Primary entry scripts kept at the root for backward compatibility:

- `evaluate-v1.py`
- `ground_truth_baseline_no_fusion.py`
- `compare.py`
- `reliability.py`
- `visualize.py`
- `judgments.py`
- `study_v1.py`
- `tune_v1.py`

Default results layout:

- `evaluation/results/3_rag_no_fusion/`
- `evaluation/results/3_rag_with_fusion/`
- `evaluation/results/3_rag_best_tuned/`
- `evaluation/results/ground_truth_baseline_no_fusion/`
