from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = PROJECT_ROOT / "rag_baseline.json"
DEFAULT_MANIFEST_JSON = PROJECT_ROOT / "corpus_manifest.json"
DEFAULT_MANIFEST_MD = PROJECT_ROOT / "corpus_manifest.md"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTIONS = ("ntu_rag", "ntu_hybrid_llamaindex", "ntu_graphrag")


# Cấu hình UTF-8 cho console để script có thể in tiếng Việt có dấu trên Windows.
def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                pass


# Đọc baseline config hiện tại để biết chunk root và scope đang dùng cho các hệ RAG.
def load_baseline_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Đếm số dòng JSONL thực sự có dữ liệu để suy ra số chunk của từng tài liệu.
def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


# Liệt kê danh sách tài liệu nguồn đang có trong `extract_md/data_chunks`.
def collect_expected_docs(chunk_root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for jsonl_path in sorted(chunk_root.rglob("*.jsonl")):
        relative_jsonl = jsonl_path.relative_to(chunk_root)
        docs.append(
            {
                "jsonl_path": relative_jsonl.as_posix(),
                "relative_path": relative_jsonl.with_suffix(".txt").as_posix(),
                "chunk_count": count_jsonl_rows(jsonl_path),
            }
        )
    return docs


# Gọi REST API của Qdrant để lấy tổng số point trong collection.
def fetch_qdrant_count(qdrant_url: str, collection: str) -> int | None:
    response = requests.post(
        f"{qdrant_url}/collections/{collection}/points/count",
        json={"exact": True},
        timeout=30,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    return int(payload["result"]["count"])


# Scroll toàn bộ payload trong collection để lấy danh sách tài liệu thật sự đang nằm trong Qdrant.
def fetch_qdrant_doc_paths(qdrant_url: str, collection: str) -> list[str] | None:
    docs: set[str] = set()
    offset: Any = None

    while True:
        body: dict[str, Any] = {
            "limit": 256,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        response = requests.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json=body,
            timeout=60,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()["result"]
        for point in payload["points"]:
            point_payload = point.get("payload") or {}
            relative_path = point_payload.get("relative_path") or point_payload.get("source_file")
            if relative_path:
                docs.add(str(relative_path))

        offset = payload.get("next_page_offset")
        if offset is None:
            break

    return sorted(docs)


# So sánh một collection Qdrant với danh sách tài liệu kỳ vọng từ extract.
def compare_collection(qdrant_url: str, collection: str, expected_docs: list[str]) -> dict[str, Any]:
    expected_set = set(expected_docs)
    count = fetch_qdrant_count(qdrant_url, collection)
    doc_paths = fetch_qdrant_doc_paths(qdrant_url, collection)

    if doc_paths is None:
        return {
            "collection": collection,
            "exists": False,
            "point_count": None,
            "doc_count": 0,
            "missing_docs": expected_docs,
            "extra_docs": [],
        }

    actual_set = set(doc_paths)
    return {
        "collection": collection,
        "exists": True,
        "point_count": count,
        "doc_count": len(doc_paths),
        "doc_paths": doc_paths,
        "missing_docs": sorted(expected_set - actual_set),
        "extra_docs": sorted(actual_set - expected_set),
    }


# Kết xuất manifest Markdown để nhìn nhanh danh sách tài liệu và trạng thái sync.
def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Corpus Manifest")
    lines.append("")
    lines.append(f"- Scope: `{manifest['scope']}`")
    lines.append(f"- Chunk root: `{manifest['chunk_root']}`")
    lines.append(f"- Tài liệu nguồn: `{manifest['document_count']}`")
    lines.append(f"- Tổng chunk: `{manifest['chunk_count']}`")
    lines.append("")
    lines.append("## Tài liệu trong extract")
    lines.append("")
    for doc in manifest["documents"]:
        lines.append(f"- `{doc['relative_path']}`: {doc['chunk_count']} chunk")

    lines.append("")
    lines.append("## Đối chiếu Qdrant")
    lines.append("")
    for collection in manifest["collections"]:
        lines.append(f"### {collection['collection']}")
        if not collection["exists"]:
            lines.append("- Trạng thái: không tồn tại")
            lines.append("")
            continue
        lines.append(f"- Point count: `{collection['point_count']}`")
        lines.append(f"- Số tài liệu phân biệt: `{collection['doc_count']}`")
        lines.append(f"- Thiếu tài liệu: `{len(collection['missing_docs'])}`")
        lines.append(f"- Dư tài liệu: `{len(collection['extra_docs'])}`")
        if collection["missing_docs"]:
            lines.append("- Danh sách thiếu:")
            for item in collection["missing_docs"]:
                lines.append(f"  - `{item}`")
        if collection["extra_docs"]:
            lines.append("- Danh sách dư:")
            for item in collection["extra_docs"]:
                lines.append(f"  - `{item}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# Entry point: đọc baseline config, lập manifest và đối chiếu với các collection Qdrant.
def main() -> None:
    configure_console_utf8()
    baseline = load_baseline_config(BASELINE_CONFIG_PATH)
    corpus = baseline.get("corpus") or {}
    chunk_root = Path(str(corpus.get("chunk_root") or PROJECT_ROOT / "data_chunks"))
    scope = str(corpus.get("scope") or "all")

    documents = collect_expected_docs(chunk_root)
    expected_paths = [doc["relative_path"] for doc in documents]
    qdrant_url = DEFAULT_QDRANT_URL

    manifest = {
        "scope": scope,
        "chunk_root": str(chunk_root),
        "document_count": len(documents),
        "chunk_count": sum(int(doc["chunk_count"]) for doc in documents),
        "documents": documents,
        "collections": [compare_collection(qdrant_url, collection, expected_paths) for collection in DEFAULT_COLLECTIONS],
    }

    DEFAULT_MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    DEFAULT_MANIFEST_MD.write_text(render_manifest_markdown(manifest), encoding="utf-8")

    print(f"Đã ghi manifest JSON: {DEFAULT_MANIFEST_JSON}")
    print(f"Đã ghi manifest Markdown: {DEFAULT_MANIFEST_MD}")
    print(f"Tài liệu nguồn: {manifest['document_count']} | Tổng chunk: {manifest['chunk_count']}")
    for collection in manifest["collections"]:
        status = "không tồn tại" if not collection["exists"] else "khớp"
        if collection["exists"] and (collection["missing_docs"] or collection["extra_docs"]):
            status = "lệch"
        print(
            f"- {collection['collection']}: {status} | "
            f"missing={len(collection['missing_docs'])} | extra={len(collection['extra_docs'])}"
        )


if __name__ == "__main__":
    main()
