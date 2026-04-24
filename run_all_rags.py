from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
UI_HOST = "127.0.0.1"
FASTEMBED_CACHE_ROOT = PROJECT_ROOT / ".cache" / "fastembed"
FASTEMBED_BM25_CACHE_DIR = FASTEMBED_CACHE_ROOT / "models--Qdrant--bm25"

RAG_SERVERS = [
    {"name": "hybrid", "path": PROJECT_ROOT / "hybrid_rag" / "app.py", "ui_port": 8000},
    {"name": "baseline", "path": PROJECT_ROOT / "rag" / "app.py", "ui_port": 8001},
    {"name": "graph", "path": PROJECT_ROOT / "graph_rag" / "app.py", "ui_port": 8502},
]


def _ensure_fastembed_cache_root() -> Path:
    FASTEMBED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return FASTEMBED_CACHE_ROOT


def _bm25_cache_metadata_matches(snapshot_dir: Path) -> bool:
    metadata_path = snapshot_dir / "files_metadata.json"
    if not metadata_path.exists():
        return True
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    for rel_path, metadata in payload.items():
        if not isinstance(metadata, dict):
            return False
        file_path = snapshot_dir / rel_path
        expected_size = metadata.get("size")
        if not file_path.exists() or not isinstance(expected_size, int):
            return False
        try:
            if file_path.stat().st_size != expected_size:
                return False
        except OSError:
            return False
    return True


def _repair_fastembed_bm25_cache_if_needed() -> None:
    snapshot_dir = FASTEMBED_BM25_CACHE_DIR
    if not snapshot_dir.exists():
        return
    if _bm25_cache_metadata_matches(snapshot_dir):
        return

    quarantine_dir = snapshot_dir.with_name(f"{snapshot_dir.name}.corrupt-{int(time.time())}")
    snapshot_dir.rename(quarantine_dir)
    print(
        "[launcher] Phat hien cache fastembed BM25 lech metadata. "
        f"Da co lap sang {quarantine_dir} de tai lai sach."
    )


def _is_port_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _ensure_ports_available() -> None:
    conflicts = [
        f"{server['name']}:{server['ui_port']}"
        for server in RAG_SERVERS
        if _is_port_listening(UI_HOST, int(server["ui_port"]))
    ]
    if conflicts:
        raise RuntimeError(
            "Khong the khoi dong launcher vi mot so cong dang duoc su dung: "
            f"{', '.join(conflicts)}. Hay dung cac launcher/backend cu truoc khi chay lai."
        )


# Tạo biến môi trường dùng chung để mỗi process biết port của cả 3 tab giao diện.
def _build_child_env(ui_port: int) -> dict[str, str]:
    """Tạo bộ biến môi trường con cho từng backend RAG."""

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["RAG_UI_HOST"] = UI_HOST
    child_env["HYBRID_UI_PORT"] = "8000"
    child_env["BASELINE_UI_PORT"] = "8001"
    child_env["GRAPH_UI_PORT"] = "8502"
    child_env["UI_PORT"] = str(ui_port)
    child_env["FASTEMBED_CACHE_PATH"] = str(_ensure_fastembed_cache_root())
    child_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return child_env


# In log của từng process kèm prefix để dễ phân biệt trên một terminal.
def _stream_output(name: str, stream) -> None:
    """Đọc stdout của process con và gắn prefix theo tên hệ RAG."""

    try:
        for line in stream:
            print(f"[{name}] {line.rstrip()}")
    finally:
        stream.close()


# Khởi động một RAG backend bằng Python hiện tại và nối stdout vào bộ đọc log.
def _start_process(server: dict[str, object]) -> tuple[subprocess.Popen[str], threading.Thread]:
    """Khởi động một backend RAG và tạo thread đọc log của backend đó."""

    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        # Tách process group để Ctrl+C ở launcher không đẩy KeyboardInterrupt xuống backend.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        [sys.executable, str(server["path"])],
        cwd=str(PROJECT_ROOT),
        env=_build_child_env(int(server["ui_port"])),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )
    reader = threading.Thread(
        target=_stream_output,
        args=(str(server["name"]), process.stdout),
        daemon=True,
    )
    reader.start()
    return process, reader


# Dừng một process con theo cách mềm trước, rồi mới ép kill nếu cần.
def _stop_process(process: subprocess.Popen[str]) -> None:
    """Dừng process con theo thứ tự terminate rồi kill nếu process không thoát."""

    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    process.kill()
    process.wait(timeout=5)


# Kiểm tra nhanh xem có process nào tắt sớm không để dừng cả cụm nếu có lỗi khởi động.
def _check_early_exit(processes: list[tuple[dict[str, object], subprocess.Popen[str]]]) -> str | None:
    """Phát hiện backend nào thoát sớm để launcher dừng toàn bộ cụm."""

    for server, process in processes:
        if process.poll() is not None:
            return str(server["name"])
    return None


# Entry point launcher.
# 1) Khởi động 3 backend với port cố định để tab trong UI tham chiếu đúng nhau.
# 2) In ra danh sách URL để mở và so sánh.
# 3) Giữ process cha sống cho đến khi user nhấn Ctrl+C hoặc một backend bị lỗi.
# 4) Khi dừng thì tắt toàn bộ process con một cách gọn gàng.
def main() -> None:
    """Khởi động đồng thời Baseline, Hybrid và Graph RAG trong một terminal."""

    # Bước 1: ép stdout/stderr sang UTF-8 để log tiếng Việt không bị lỗi mã hóa.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    _ensure_fastembed_cache_root()
    _repair_fastembed_bm25_cache_if_needed()
    _ensure_ports_available()
    processes: list[tuple[dict[str, object], subprocess.Popen[str]]] = []
    exit_code = 0

    try:
        # Bước 2: lần lượt khởi động từng backend và gắn thread đọc log tương ứng.
        for server in RAG_SERVERS:
            process, _reader = _start_process(server)
            processes.append((server, process))

        # Bước 3: in URL truy cập để người dùng mở 3 giao diện song song.
        print("Đã khởi động launcher cho 3 RAG:")
        for server, _ in processes:
            print(f"- {server['name']}: http://{UI_HOST}:{server['ui_port']}/")
        print("Nhấn Ctrl+C để dừng tất cả.")

        # Bước 4: giữ launcher chạy nền và theo dõi backend nào thoát sớm để báo lỗi ngay.
        while True:
            failed_name = _check_early_exit(processes)
            if failed_name:
                raise RuntimeError(f"Backend `{failed_name}` đã dừng sớm. Xem log ở trên để debug.")
            time.sleep(1)
    except RuntimeError as exc:
        exit_code = 1
        print(str(exc))
    except KeyboardInterrupt:
        print("\nĐang dừng tất cả backend...")
    finally:
        # Bước 5: khi launcher thoát, tắt các process con theo thứ tự ngược lại.
        for _, process in reversed(processes):
            _stop_process(process)
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
