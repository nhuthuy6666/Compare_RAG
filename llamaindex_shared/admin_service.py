from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_ROOT = PROJECT_ROOT / "extract_md"
EXTRACT_SCRIPTS_ROOT = EXTRACT_ROOT / "scripts"
GRAPH_PROCESSED_CHUNK_ROOT = PROJECT_ROOT / "graph_rag" / "data" / "processed" / "chunks"
PDF_ROOT = EXTRACT_ROOT / "data_raw" / "pdf"
WEB_LINK_MD = EXTRACT_ROOT / "data_raw" / "web" / "link.md"
WEB_CACHE_ROOT = EXTRACT_ROOT / "data_raw" / "web_links"
TXT_ROOT = EXTRACT_ROOT / "data_txt"
CHUNK_ROOT = EXTRACT_ROOT / "data_chunks"
BASELINE_CONFIG_PATH = EXTRACT_ROOT / "rag_baseline.json"

if str(EXTRACT_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACT_SCRIPTS_ROOT))


PDF_UPLOAD_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def list_corpus_documents() -> dict[str, Any]:
    web_documents = _list_web_documents()
    pdf_documents = _list_pdf_documents()
    return {
        "documents": [*web_documents, *pdf_documents],
        "counts": {
            "web": len(web_documents),
            "pdf": len(pdf_documents),
            "total": len(web_documents) + len(pdf_documents),
        },
    }


def add_corpus_documents(*, links_text: str = "", pdf_files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    build_web_txt, dedupe_preserve_order, extract_urls_from_markdown, _safe_slug_from_url = _load_web_tools()
    build_pdf_txt = _load_pdf_builder()
    pdf_payloads = pdf_files or []
    added_urls: list[dict[str, Any]] = []
    skipped_urls: list[str] = []
    added_pdfs: list[dict[str, Any]] = []
    skipped_pdfs: list[dict[str, str]] = []
    warnings: list[str] = []

    existing_urls = extract_urls_from_markdown(_read_text(WEB_LINK_MD))
    existing_url_set = set(existing_urls)
    requested_urls = dedupe_preserve_order(extract_urls_from_markdown(links_text or ""))
    new_urls = [url for url in requested_urls if url not in existing_url_set]
    skipped_urls.extend([url for url in requested_urls if url in existing_url_set])

    if new_urls:
        updated_urls = [*existing_urls, *new_urls]
        _write_link_md(updated_urls)

    for url in new_urls:
        try:
            txt_path = build_web_txt(
                url=url,
                out_root=TXT_ROOT,
                cache_root=WEB_CACHE_ROOT,
                allow_fetch=True,
                timeout=_get_int_env("WEB_FETCH_TIMEOUT", 300),
                render_js=False,
            )
            chunk_count, chunk_path = _chunk_txt_file(txt_path)
            added_urls.append(
                {
                    "url": url,
                    "txt_relative_path": _relative_to_extract(txt_path),
                    "chunk_relative_path": _relative_to_extract(chunk_path),
                    "chunk_count": chunk_count,
                }
            )
        except Exception as exc:
            warnings.append(f"Không thể xử lý link {url}: {exc}")

    for file_payload in pdf_payloads:
        try:
            outcome = _store_uploaded_pdf(file_payload)
        except ValueError as exc:
            warnings.append(str(exc))
            continue

        if outcome["status"] == "skipped":
            skipped_pdfs.append(
                {
                    "name": str(file_payload.get("name") or "unknown.pdf"),
                    "reason": outcome["reason"],
                }
            )
            continue

        raw_pdf_path = Path(outcome["raw_pdf_path"])
        try:
            txt_path = build_pdf_txt(raw_pdf_path, TXT_ROOT)
            if txt_path is None:
                warnings.append(f"Không trích xuất được nội dung PDF: {raw_pdf_path.name}")
                continue
            chunk_count, chunk_path = _chunk_txt_file(txt_path)
            added_pdfs.append(
                {
                    "name": raw_pdf_path.name,
                    "raw_relative_path": _relative_to_extract(raw_pdf_path),
                    "txt_relative_path": _relative_to_extract(txt_path),
                    "chunk_relative_path": _relative_to_extract(chunk_path),
                    "chunk_count": chunk_count,
                }
            )
        except Exception as exc:
            warnings.append(f"Không thể xử lý PDF {raw_pdf_path.name}: {exc}")

    _refresh_baseline_chunk_count()
    changed = bool(new_urls or added_pdfs)
    return {
        "changed": changed,
        "added": {
            "web": added_urls,
            "pdf": added_pdfs,
        },
        "skipped": {
            "web": skipped_urls,
            "pdf": skipped_pdfs,
        },
        "warnings": warnings,
        "message": _build_add_message(added_urls, added_pdfs, skipped_urls, skipped_pdfs, warnings),
    }


def delete_corpus_documents(document_ids: list[str]) -> dict[str, Any]:
    dedupe_preserve_order, extract_urls_from_markdown, safe_slug_from_url = _load_web_listing_tools()
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    ids = dedupe_preserve_order([str(item).strip() for item in document_ids if str(item).strip()])
    if not ids:
        return {
            "changed": False,
            "removed": removed,
            "skipped": skipped,
            "warnings": warnings,
            "message": "Chưa chọn tài liệu để xóa.",
        }

    current_urls = extract_urls_from_markdown(_read_text(WEB_LINK_MD))
    remaining_urls = list(current_urls)

    for document_id in ids:
        if document_id.startswith("web::"):
            url = document_id.split("::", 1)[1]
            slug = safe_slug_from_url(url)
            existed = url in remaining_urls
            if existed:
                remaining_urls = [item for item in remaining_urls if item != url]
            removed_any = existed
            removed_any = _remove_file(WEB_CACHE_ROOT / "html" / f"{slug}.html") or removed_any
            removed_any = _remove_file(TXT_ROOT / "web" / f"{slug}.txt") or removed_any
            removed_any = _remove_file(CHUNK_ROOT / "web" / f"{slug}.jsonl") or removed_any
            removed_any = _remove_file(GRAPH_PROCESSED_CHUNK_ROOT / "web" / f"{slug}.jsonl") or removed_any
            if removed_any:
                removed.append({"id": document_id, "kind": "web", "label": url})
            else:
                skipped.append({"id": document_id, "reason": "Tài liệu web không còn tồn tại."})
            continue

        if document_id.startswith("pdf::"):
            raw_relative_path = document_id.split("::", 1)[1]
            raw_pdf_path = EXTRACT_ROOT / raw_relative_path
            if raw_pdf_path.suffix.lower() != ".pdf":
                skipped.append({"id": document_id, "reason": "Định danh PDF không hợp lệ."})
                continue
            txt_path = _pdf_txt_path(raw_pdf_path)
            chunk_path = _txt_to_chunk_path(txt_path)
            graph_chunk_path = GRAPH_PROCESSED_CHUNK_ROOT / chunk_path.relative_to(CHUNK_ROOT)
            removed_any = _remove_file(raw_pdf_path)
            removed_any = _remove_file(txt_path) or removed_any
            removed_any = _remove_file(chunk_path) or removed_any
            removed_any = _remove_file(graph_chunk_path) or removed_any
            if removed_any:
                removed.append({"id": document_id, "kind": "pdf", "label": raw_pdf_path.name})
            else:
                skipped.append({"id": document_id, "reason": "PDF không còn tồn tại."})
            continue

        skipped.append({"id": document_id, "reason": "Loại tài liệu không được hỗ trợ."})

    _write_link_md(remaining_urls)
    _prune_empty_dirs(WEB_CACHE_ROOT)
    _prune_empty_dirs(TXT_ROOT)
    _prune_empty_dirs(CHUNK_ROOT)
    _prune_empty_dirs(PDF_ROOT)
    _prune_empty_dirs(GRAPH_PROCESSED_CHUNK_ROOT)
    _refresh_baseline_chunk_count()
    return {
        "changed": bool(removed),
        "removed": removed,
        "skipped": skipped,
        "warnings": warnings,
        "message": _build_delete_message(removed, skipped),
    }


def load_document_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    raw_documents = payload.get("documents") or payload.get("document_ids") or []
    document_ids: list[str] = []
    if isinstance(raw_documents, list):
        for item in raw_documents:
            if isinstance(item, str):
                document_ids.append(item)
            elif isinstance(item, dict):
                candidate = str(item.get("id") or "").strip()
                if candidate:
                    document_ids.append(candidate)
    return document_ids


def _list_web_documents() -> list[dict[str, Any]]:
    _, extract_urls_from_markdown, safe_slug_from_url = _load_web_listing_tools()
    documents: list[dict[str, Any]] = []
    for url in extract_urls_from_markdown(_read_text(WEB_LINK_MD)):
        slug = safe_slug_from_url(url)
        documents.append(
            {
                "id": f"web::{url}",
                "kind": "web",
                "label": url,
                "source_key": url,
                "raw_relative_path": _relative_to_extract(WEB_LINK_MD),
                "cache_relative_path": f"data_raw/web_links/html/{slug}.html",
                "txt_relative_path": f"data_txt/web/{slug}.txt",
                "chunk_relative_path": f"data_chunks/web/{slug}.jsonl",
            }
        )
    return documents


def _list_pdf_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(PDF_ROOT.rglob("*.pdf")):
        txt_path = _pdf_txt_path(pdf_path)
        chunk_path = _txt_to_chunk_path(txt_path)
        raw_relative_path = _relative_to_extract(pdf_path)
        documents.append(
            {
                "id": f"pdf::{raw_relative_path}",
                "kind": "pdf",
                "label": pdf_path.name,
                "source_key": raw_relative_path,
                "raw_relative_path": raw_relative_path,
                "txt_relative_path": _relative_to_extract(txt_path),
                "chunk_relative_path": _relative_to_extract(chunk_path),
            }
        )
    return documents


def _store_uploaded_pdf(file_payload: dict[str, Any]) -> dict[str, str]:
    file_name = str(file_payload.get("name") or "").strip()
    if not file_name:
        raise ValueError("Thiếu tên file PDF.")
    suffix = Path(file_name).suffix.lower()
    if suffix != ".pdf":
        raise ValueError(f"Chỉ hỗ trợ PDF: {file_name}")

    raw_content = str(file_payload.get("content_base64") or "").strip()
    if not raw_content:
        raise ValueError(f"Thiếu nội dung file PDF: {file_name}")
    if "," in raw_content and raw_content.lower().startswith("data:"):
        raw_content = raw_content.split(",", 1)[1]

    try:
        content_bytes = base64.b64decode(raw_content, validate=True)
    except Exception as exc:
        raise ValueError(f"File PDF {file_name} không đúng định dạng base64.") from exc
    if not content_bytes:
        raise ValueError(f"File PDF {file_name} đang rỗng.")

    content_hash = hashlib.sha1(content_bytes).hexdigest()
    existing = _find_existing_pdf_by_hash(content_hash)
    if existing is not None:
        return {"status": "skipped", "reason": f"Đã có file trùng nội dung: {existing.name}"}

    safe_name = _safe_pdf_filename(file_name)
    target_dir = PDF_ROOT / str(date.today().year)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = _unique_path(target_dir / safe_name)
    target_path.write_bytes(content_bytes)
    return {"status": "stored", "raw_pdf_path": str(target_path)}


def _find_existing_pdf_by_hash(content_hash: str) -> Path | None:
    for pdf_path in PDF_ROOT.rglob("*.pdf"):
        try:
            existing_hash = hashlib.sha1(pdf_path.read_bytes()).hexdigest()
        except OSError:
            continue
        if existing_hash == content_hash:
            return pdf_path
    return None


def _safe_pdf_filename(file_name: str) -> str:
    _, _, slugify, _ = _load_corpus_tools()
    stem = Path(file_name).stem or "tai-lieu-moi"
    stem = PDF_UPLOAD_NAME_RE.sub("-", stem).strip(" .-_")
    safe_stem = slugify(stem) or "tai-lieu-moi"
    return f"{safe_stem}.pdf"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Không thể tạo tên file duy nhất cho {path.name}")


def _chunk_txt_file(txt_path: Path) -> tuple[int, Path]:
    build_chunk_rows, _, _, write_jsonl = _load_corpus_tools()
    settings = _load_chunk_settings()
    chunk_path = _txt_to_chunk_path(txt_path)
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    rows = build_chunk_rows(
        text=text,
        source_txt=txt_path,
        min_chars=settings["min_chars"],
        max_chars=settings["max_chars"],
        max_sections_per_chunk=settings["max_sections_per_chunk"],
    )
    write_jsonl(chunk_path, rows)
    return len(rows), chunk_path


def _load_chunk_settings() -> dict[str, int]:
    default = {
        "min_chars": 120,
        "max_chars": 700,
        "max_sections_per_chunk": 2,
    }
    if not BASELINE_CONFIG_PATH.exists():
        return default
    try:
        payload = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    corpus = payload.get("corpus") or {}
    return {
        "min_chars": int(corpus.get("min_chars") or default["min_chars"]),
        "max_chars": int(corpus.get("max_chars") or default["max_chars"]),
        "max_sections_per_chunk": int(corpus.get("max_sections_per_chunk") or default["max_sections_per_chunk"]),
    }


def _refresh_baseline_chunk_count() -> None:
    if not BASELINE_CONFIG_PATH.exists():
        return
    try:
        payload = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    corpus = payload.setdefault("corpus", {})
    corpus["chunk_count"] = _count_chunk_rows()
    BASELINE_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_chunk_rows() -> int:
    total = 0
    if not CHUNK_ROOT.exists():
        return total
    for jsonl_path in CHUNK_ROOT.rglob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _relative_to_extract(path: Path) -> str:
    return path.relative_to(EXTRACT_ROOT).as_posix()


def _pdf_txt_path(pdf_path: Path) -> Path:
    _, guess_year, slugify, _ = _load_corpus_tools()
    year = guess_year(pdf_path)
    return TXT_ROOT / year / f"{slugify(pdf_path.stem)}.txt"


def _txt_to_chunk_path(txt_path: Path) -> Path:
    return (CHUNK_ROOT / txt_path.relative_to(TXT_ROOT)).with_suffix(".jsonl")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_link_md(urls: list[str]) -> None:
    WEB_LINK_MD.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(urls).strip()
    WEB_LINK_MD.write_text((content + "\n") if content else "", encoding="utf-8")


def _remove_file(path: Path) -> bool:
    if not path.exists() or path.is_dir():
        return False
    path.unlink()
    return True


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            next(directory.iterdir())
        except StopIteration:
            try:
                directory.rmdir()
            except OSError:
                continue


def _build_add_message(
    added_urls: list[dict[str, Any]],
    added_pdfs: list[dict[str, Any]],
    skipped_urls: list[str],
    skipped_pdfs: list[dict[str, str]],
    warnings: list[str],
) -> str:
    parts: list[str] = []
    if added_urls or added_pdfs:
        parts.append(f"Đã thêm {len(added_urls)} link mới và {len(added_pdfs)} PDF mới.")
    if skipped_urls or skipped_pdfs:
        parts.append(f"Bỏ qua {len(skipped_urls) + len(skipped_pdfs)} tài liệu đã có.")
    if warnings:
        parts.append(f"Có {len(warnings)} cảnh báo khi trích xuất dữ liệu.")
    if not parts:
        return "Không có dữ liệu mới để nạp."
    return " ".join(parts)


def _build_delete_message(removed: list[dict[str, Any]], skipped: list[dict[str, str]]) -> str:
    parts: list[str] = []
    if removed:
        parts.append(f"Đã xóa {len(removed)} tài liệu khỏi dữ liệu thô, TXT và chunk.")
    if skipped:
        parts.append(f"Bỏ qua {len(skipped)} mục không còn tồn tại hoặc không hợp lệ.")
    if not parts:
        return "Không có tài liệu nào được xóa."
    return " ".join(parts)


def _load_web_tools():
    from build_links_txt import build_web_txt, dedupe_preserve_order, extract_urls_from_markdown, safe_slug_from_url

    return build_web_txt, dedupe_preserve_order, extract_urls_from_markdown, safe_slug_from_url


def _load_web_listing_tools():
    from build_links_txt import dedupe_preserve_order, extract_urls_from_markdown, safe_slug_from_url

    return dedupe_preserve_order, extract_urls_from_markdown, safe_slug_from_url


def _load_pdf_builder():
    from build_pdf_txt import build_pdf_txt

    return build_pdf_txt


def _load_corpus_tools():
    from corpus_utils import build_chunk_rows, guess_year, slugify, write_jsonl

    return build_chunk_rows, guess_year, slugify, write_jsonl
