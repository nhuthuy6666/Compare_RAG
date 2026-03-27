from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from http_app import main  # noqa: E402


__all__ = ["main"]


# Entry point chính của GraphRAG ở project root để đồng bộ với `rag/app.py` và `hybrid_rag/app.py`.
# 1) Thêm `graph_rag/src` vào `sys.path`.
# 2) Import hàm `main` từ `http_app`.
# 3) Khi chạy trực tiếp thì khởi động web UI và API qua `main()`.
if __name__ == "__main__":
    main()
