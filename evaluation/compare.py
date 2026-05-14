import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reporting.compare import (
    build_comparison_report,
    build_retrieval_answer_quality_report,
    build_system_performance_report,
    load_rows,
    main,
    parse_args,
    render_metadata,
    render_retrieval_answer_quality_table,
    render_strength_tables,
    render_summary_table,
    render_system_performance_table,
    to_float,
)


if __name__ == "__main__":
    main()
