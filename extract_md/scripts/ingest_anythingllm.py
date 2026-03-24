import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests import exceptions as req_exc


# In log ngay lập tức để theo dõi tiến trình ingest theo thời gian thực.
def log(message: str) -> None:
    print(message, flush=True)


YEAR_DIR_RE = re.compile(r"^20\d{2}$")
WEB_DIR = "web"


@dataclass(frozen=True)
class ChunkDoc:
    title: str
    text: str
    chunk_path: str
    source_txt: str
    heading_path: list[str]
    chunk_id: int
    category: str  # pdf | web


# Nạp biến môi trường từ file .env dạng KEY=VALUE.
def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.lstrip("\ufeff")
        env[k.strip()] = v.strip()
    return env


class AnythingLLMClient:
    # Khởi tạo HTTP session dùng chung cho các request AnythingLLM.
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # Gọi GET đến API AnythingLLM và trả về JSON.
    def _get(self, path: str) -> Any:
        resp = self.session.get(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.json()

    # Gọi POST đến API AnythingLLM và trả về JSON nếu có.
    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        resp = self.session.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()

    # Kiểm tra API server còn online hay không.
    def ping(self) -> bool:
        data = self._get("/api/ping")
        return bool(data.get("online"))

    # Kiểm tra workspace có tồn tại theo slug.
    def workspace_exists(self, slug: str) -> bool:
        data = self._get(f"/api/v1/workspace/{slug}")
        return bool(data.get("workspace"))

    # Lấy thông tin chi tiết 1 workspace.
    def get_workspace(self, slug: str) -> dict[str, Any]:
        data = self._get(f"/api/v1/workspace/{slug}")
        workspaces = data.get("workspace") or []
        if not workspaces:
            raise ValueError(f"Workspace slug not found: {slug}")
        return dict(workspaces[0])

    # Cập nhật cấu hình workspace.
    def update_workspace(self, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/api/v1/workspace/{slug}/update", payload)

    # Lấy cấu hình hệ thống AnythingLLM.
    def get_system_settings(self) -> dict[str, Any]:
        data = self._get("/api/v1/system")
        return dict(data.get("settings") or {})

    # Lấy danh sách title tài liệu đã tồn tại trong AnythingLLM.
    def list_existing_titles(self) -> set[str]:
        data = self._get("/api/v1/documents")
        root = data.get("localFiles", {})
        titles: set[str] = set()

        # Duyệt đệ quy cây folder/file trả về từ API để gom tất cả title.
        def walk(node: dict[str, Any]) -> None:
            for item in node.get("items", []) or []:
                if isinstance(item, dict) and item.get("type") == "folder":
                    walk(item)
                elif isinstance(item, dict):
                    title = item.get("title")
                    if title:
                        titles.add(str(title))

        if isinstance(root, dict):
            walk(root)
        return titles

    # Lấy danh sách title tài liệu đang thực sự gắn với 1 workspace cụ thể.
    def list_workspace_titles(self, slug: str) -> set[str]:
        workspace = self.get_workspace(slug)
        titles: set[str] = set()
        for item in workspace.get("documents") or []:
            metadata_raw = item.get("metadata")
            if isinstance(metadata_raw, str) and metadata_raw.strip():
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            for candidate in (
                metadata.get("title") if isinstance(metadata, dict) else None,
                item.get("title"),
                item.get("filename"),
            ):
                if candidate:
                    titles.add(str(candidate))
        return titles

    # Gửi một chunk text lên AnythingLLM theo endpoint raw-text.
    def ingest_raw_text(self, workspace_slug: str, label: str, doc: ChunkDoc) -> dict[str, Any]:
        filename = doc.title if doc.title.endswith(".txt") else f"{doc.title}.txt"
        payload = {
            "textContent": doc.text,
            "addToWorkspaces": workspace_slug,
            "filename": filename,
            "metadata": {
                "title": doc.title,
                "filename": filename,
                "docSource": label,
                "chunkSource": doc.chunk_path,
                "sourceTxt": doc.source_txt,
                "headingPath": " > ".join(doc.heading_path) if doc.heading_path else "",
                "chunkId": doc.chunk_id,
                "description": f"{label}:{doc.category}_chunk",
            },
        }
        # Use a plain request for thread-safe parallel ingestion.
        resp = requests.post(
            f"{self.base_url}/api/v1/document/raw-text",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": self.session.headers.get("Authorization", ""),
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


# Chuyển đường dẫn tương đối thành slug để tạo title an toàn.
def slug_from_relpath(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")


# Chuẩn hóa title để so sánh trùng lặp (không phân biệt .txt).
def normalize_title(title: str) -> str:
    t = title.strip().lower()
    if t.endswith(".txt"):
        t = t[:-4]
    return t


# Liệt kê các file chunk JSONL hợp lệ (nhóm năm và web).
def iter_chunk_files(chunk_root: Path) -> list[Path]:
    candidates = sorted(chunk_root.rglob("*.jsonl"))
    if not candidates:
        return []
    chunk_files: list[Path] = []
    for path in candidates:
        rel = path.relative_to(chunk_root)
        if not rel.parts:
            continue
        if YEAR_DIR_RE.fullmatch(rel.parts[0]) or rel.parts[0].lower() == WEB_DIR:
            chunk_files.append(path)
    return chunk_files


# Đọc JSONL chunk và tạo danh sách ChunkDoc sẵn sàng ingest.
def build_docs(chunk_root: Path, label: str) -> list[ChunkDoc]:
    # B1: Duyet cac file chunk JSONL hop le trong chunk_root.
    docs: list[ChunkDoc] = []
    for chunk_path in iter_chunk_files(chunk_root):
        rel = chunk_path.relative_to(chunk_root).as_posix()
        top = rel.split("/", 1)[0].lower() if rel else ""
        category = "pdf" if YEAR_DIR_RE.fullmatch(top) else "web"
        try:
            lines = chunk_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            log(f"[WARN] Skip unreadable chunk file: {chunk_path} ({exc})")
            continue
        # B2: Parse từng dòng JSONL, validate text, tạo ChunkDoc.
        for raw in lines:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            chunk_id = int(row.get("chunk_id") or 0)
            title = f"{label}__chunk__{slug_from_relpath(rel)}__{chunk_id:04d}"
            docs.append(
                ChunkDoc(
                    title=title,
                    text=text,
                    chunk_path=str(chunk_path),
                    source_txt=str(row.get("source_txt", "")),
                    heading_path=list(row.get("heading_path") or []),
                    chunk_id=chunk_id,
                    category=category,
                )
            )
    # B3: Trả danh sách tài liệu chunk đã được chuẩn hóa.
    return docs


# Xử lý toàn bộ quy trình ingest: đọc cấu hình, lọc trùng lặp, gọi API, tổng hợp kết quả.
def ingest_chunk_root(
    env_file: Path,
    chunk_root: Path,
    label: str | None,
    workspace_slug: str | None,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    # B1: Nạp config ingest từ env file + tham số CLI.
    env = load_env_file(env_file)
    base_url = env.get("ANYTHINGLLM_BASE_URL", "").strip()
    api_key = env.get("ANYTHINGLLM_API_KEY", "").strip()
    resolved_workspace_slug = (workspace_slug or env.get("WORKSPACE_SLUG", "")).strip()
    resolved_label = (label or resolved_workspace_slug).strip()

    if not base_url or not api_key or not resolved_workspace_slug:
        raise SystemExit(
            "Missing config. Require ANYTHINGLLM_BASE_URL, ANYTHINGLLM_API_KEY, WORKSPACE_SLUG "
            f"(env file: {env_file})."
        )

    # B2: Build danh sách docs từ chunk files.
    docs = build_docs(chunk_root=chunk_root, label=resolved_label)
    if not docs:
        log("No chunk documents found to ingest.")
        return {"success": 0, "failed": 0, "skipped_existing": 0}

    # B3: Kiểm tra kết nối API/workspace trước khi ingest.
    client = AnythingLLMClient(base_url=base_url, api_key=api_key)
    if not client.ping():
        raise SystemExit("AnythingLLM API is offline.")
    if not client.workspace_exists(resolved_workspace_slug):
        raise SystemExit(f"Workspace slug not found: {resolved_workspace_slug}")

    # B4: Lọc chunk đã tồn tại theo title (trừ khi force).
    existing_titles = {normalize_title(t) for t in client.list_workspace_titles(resolved_workspace_slug)}
    to_ingest: list[ChunkDoc] = []
    skipped_existing = 0
    for doc in docs:
        if (not force) and normalize_title(doc.title) in existing_titles:
            skipped_existing += 1
            continue
        to_ingest.append(doc)

    log(f"Discovered chunks: {len(docs)}")
    log(f"Skip existing: {skipped_existing}")
    log(f"To ingest: {len(to_ingest)}")
    log(f"Workspace: {resolved_workspace_slug}")
    log(f"Label: {resolved_label}")

    if not to_ingest:
        log("Nothing new to ingest.")
        return {"success": 0, "failed": 0, "skipped_existing": skipped_existing}

    # B5: Dry-run chỉ in hành động dự kiến, không gọi API ingest.
    if dry_run:
        for i, doc in enumerate(to_ingest[:15], start=1):
            log(f"[dry-run] {i}. {doc.title} <- {doc.chunk_path}")
        if len(to_ingest) > 15:
            log(f"[dry-run] ... and {len(to_ingest)-15} more")
        return {"success": 0, "failed": 0, "skipped_existing": skipped_existing}

    # B6: Ingest song song và tổng hợp kết quả success/failed.
    success = 0
    failed = 0
    total = len(to_ingest)
    max_workers = min(8, total, (os.cpu_count() or 4))
    futures: dict[Any, ChunkDoc] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for doc in to_ingest:
            future = executor.submit(client.ingest_raw_text, resolved_workspace_slug, resolved_label, doc)
            futures[future] = doc

        completed = 0
        for future in as_completed(futures):
            completed += 1
            doc = futures[future]
            try:
                future.result()
                success += 1
                log(f"[{completed}/{total}] OK: {doc.title} | Ingested={success}/{total}")
            except req_exc.Timeout as e:
                failed += 1
                log(f"[{completed}/{total}] FAIL(timeout): {doc.title} :: {e} | Ingested={success}/{total}")
            except req_exc.ConnectionError as e:
                failed += 1
                log(f"[{completed}/{total}] FAIL(connection): {doc.title} :: {e} | Ingested={success}/{total}")
            except req_exc.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                failed += 1
                log(f"[{completed}/{total}] FAIL(http {status}): {doc.title} :: {e} | Ingested={success}/{total}")
            except Exception as e:
                failed += 1
                log(f"[{completed}/{total}] FAIL: {doc.title} :: {e} | Ingested={success}/{total}")

    log(f"Done. Chunks={len(docs)}, Success={success}, Failed={failed}, SkippedExisting={skipped_existing}")
    return {"success": success, "failed": failed, "skipped_existing": skipped_existing}


# Điểm vào CLI cho quy trình ingest chunk vào AnythingLLM.
def main() -> None:
    # 1) Khai báo tham số CLI.
    parser = argparse.ArgumentParser(description="Ingest chunk JSONL (PDF only) into AnythingLLM workspace.")
    parser.add_argument("--env-file", default=".env.anythingllm", help="Path to env file for AnythingLLM config.")
    parser.add_argument("--chunk-root", default="data_chunks", help="Root folder containing chunk JSONL files.")
    parser.add_argument(
        "--label",
        default=None,
        help="Label written into metadata/doc title prefix. Defaults to workspace slug.",
    )
    parser.add_argument("--workspace-slug", default=None, help="Override workspace slug from env file.")
    parser.add_argument("--force", action="store_true", help="Ingest even if a title already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without ingesting.")

    # 2) Parse args.
    args = parser.parse_args()

    # 3) Chạy pipeline ingest: đọc chunk JSONL -> lọc trùng -> gọi API AnythingLLM.
    ingest_chunk_root(
        env_file=Path(args.env_file),
        chunk_root=Path(args.chunk_root),
        label=args.label,
        workspace_slug=args.workspace_slug,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
