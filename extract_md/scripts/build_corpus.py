import argparse
import json
import subprocess
import shutil
from pathlib import Path

from corpus_utils import guess_year, slugify


CHUNK_PROFILES = {
    "balanced": {"min_chars": 120, "max_chars": 700, "max_sections_per_chunk": 2},
    "max": {"min_chars": 80, "max_chars": 550, "max_sections_per_chunk": 1},
}

KNOWN_TEMP_DIRS = ("data_chunks_demo",)

PURE_RAG_PROMPT = """Bạn là chatbot hỗ trợ tư vấn tuyển sinh của Trường Đại học Nha Trang.
Chỉ được trả lời bằng thông tin có trong ngữ cảnh được cung cấp. Không suy đoán, không tự bịa, không dùng kiến thức ngoài.

Quy tắc trả lời:
- Trả lời trực tiếp vào câu hỏi, ngắn gọn, đúng trọng tâm.
- Không nhắc đến tài liệu, bảng, biểu, nguồn, chunk, metadata hay quá trình truy xuất.
- Không viết các cụm như "theo tài liệu", "theo bảng", "tham khảo", "xem thêm".
- Nếu câu hỏi là fact ngắn như email, số điện thoại, mã trường, chỉ tiêu, thời hạn: trả lời đúng fact đó thật gọn.
- Với câu hỏi số liệu, chỉ nêu con số khi tên ngành, chương trình và năm khớp rõ ràng.
- Nếu ngữ cảnh không đủ chắc chắn hoặc không có câu trả lời trực tiếp, phải trả đúng câu từ chối được chỉ định và dừng lại.
Trả lời bằng tiếng Việt."""

PURE_RAG_QUERY_REFUSAL = "Tôi chưa đủ căn cứ để trả lời câu hỏi này từ dữ liệu hiện có."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corpus cho demo AnythingLLM: lam sach output, chunk, ingest va ap cau hinh model."
    )
    # Nhóm tham số input/output
    parser.add_argument("--pdf-root", default="data_raw/pdf", help="Input folder containing PDF files.")
    parser.add_argument("--link-md", default="data_raw/web/link.md", help="Markdown file containing web links to fetch.")
    parser.add_argument("--web-cache-root", default="data_raw/web_links", help="Folder for cached web HTML files.")
    parser.add_argument("--txt-root", default="data_txt", help="Output folder for TXT files.")
    parser.add_argument("--chunk-root", default="data_chunks", help="Output folder for chunk JSONL files.")
    # Nhóm scope xử lý
    parser.add_argument(
        "--scope",
        choices=("all", "web", "pdf"),
        default="web",
        help="Subset du lieu can build. Mac dinh la web vi on dinh hon cho demo hoi-dap.",
    )
    # Nhóm web processing
    parser.add_argument("--offline", action="store_true", help="Do not fetch web pages; use only cached HTML.")
    parser.add_argument("--render-js", action="store_true", help="Render web pages with Playwright and click tab/collapse controls before extraction.")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip TXT chunking.")
    parser.add_argument("--sync", action="store_true", help="Delete stale generated HTML/TXT/JSONL files.")
    # Nhóm chunking
    parser.add_argument(
        "--chunk-profile",
        choices=tuple(CHUNK_PROFILES),
        default="balanced",
        help="Chunk profile: balanced hoac max. Mac dinh balanced de lam baseline RAG thuần gon hon.",
    )
    parser.add_argument("--min-chars", type=int, default=None, help="Override chunk profile min_chars.")
    parser.add_argument("--max-chars", type=int, default=None, help="Override chunk profile max_chars.")
    parser.add_argument("--max-sections-per-chunk", type=int, default=None, help="Override chunk profile max_sections_per_chunk.")
    # Nhóm ingest AnythingLLM
    parser.add_argument("--ingest", action="store_true", help="Ingest chunk JSONL into AnythingLLM after build.")
    parser.add_argument("--env-file", default=".env.anythingllm", help="Env file for AnythingLLM ingest config.")
    parser.add_argument("--label", default=None, help="Metadata label prefix for AnythingLLM ingest.")
    parser.add_argument("--workspace-slug", default=None, help="Override workspace slug for AnythingLLM config/ingest.")
    parser.add_argument("--force-ingest", action="store_true", help="Ingest even if titles already exist.")
    parser.add_argument("--dry-run-ingest", action="store_true", help="Print ingest actions without sending them.")
    # Nhóm apply settings cho workspace
    parser.add_argument("--apply-settings", action="store_true", help="Update AnythingLLM workspace settings.")
    parser.add_argument("--chat-model", default="llama3.1:8b", help="Ollama chat model for the workspace.")
    parser.add_argument("--embed-model", default="bge-m3:latest", help="Preferred AnythingLLM embedding model. If changed, desktop restart may be required.")
    parser.add_argument("--temperature", type=float, default=0.1, help="AnythingLLM workspace temperature.")
    parser.add_argument("--history", type=int, default=1, help="AnythingLLM workspace history length.")
    parser.add_argument("--top-n", type=int, default=6, help="AnythingLLM workspace retrieval topN.")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="AnythingLLM workspace similarity threshold.")
    parser.add_argument(
        "--vector-search-mode",
        choices=("default", "rerank"),
        default="default",
        help="AnythingLLM vector search mode.",
    )
    parser.add_argument(
        "--chat-mode",
        choices=("chat", "query"),
        default="chat",
        help="AnythingLLM chat mode.",
    )
    parser.add_argument("--skip-model-pull", action="store_true", help="Do not auto-pull missing Ollama models.")
    parser.add_argument("--prompt", default=PURE_RAG_PROMPT, help="Workspace prompt to apply when --apply-settings is set.")
    parser.add_argument("--query-refusal-response", default=PURE_RAG_QUERY_REFUSAL, help="Workspace refusal text when retrieval is uncertain.")
    return parser.parse_args()

# Chuyển giá trị scope thành 2 cờ boolean
def resolve_stage_flags(scope: str) -> tuple[bool, bool]:
    run_pdf = scope in {"all", "pdf"}
    run_web = scope in {"all", "web"}
    return run_pdf, run_web

# Xóa 1 file nếu nó tồn tại và thực sự là file
def remove_file(path: Path) -> bool:
    if not path.exists() or path.is_dir():
        return False
    path.unlink() # xóa một file tại đường dẫn path
    return True

# Xóa các thư mục rỗng sau khi xóa file
def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            next(directory.iterdir())
        except StopIteration:
            try:
                directory.rmdir() # Xóa thư mục (khi rỗng)
            except (PermissionError, OSError):
                continue

# Dọn (xóa) các file “stale” trong một thư mục
def sync_files(
    root: Path,
    expected_relative_paths: set[str],
    suffix: str,
    allowed_prefixes: set[str] | None = None,
) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob(f"*{suffix}"):
        rel = path.relative_to(root).as_posix()
        if allowed_prefixes is not None:
            parts = Path(rel).parts
            top_level = parts[0] if parts else ""
            if top_level not in allowed_prefixes:
                continue
        if rel in expected_relative_paths:
            continue
        if remove_file(path):
            removed += 1
            print(f"Removed stale file: {path}")
    prune_empty_dirs(root)
    return removed

# Xóa các thư mục tạm đã biết trong KNOWN_TEMP_DIRS
def cleanup_temp_dirs(project_root: Path) -> int:
    removed = 0
    for name in KNOWN_TEMP_DIRS:
        path = project_root / name
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except PermissionError:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Remove-Item -LiteralPath '{path}' -Recurse -Force",
                ],
                check=True,
            )
        removed += 1
        print(f"Removed temp dir: {path}")
    return removed

# Tính danh sách file web hợp lệ cần tồn tại
def expected_web_relative_paths(link_md: Path) -> tuple[set[str], set[str], set[str]]:
    from build_links_txt import extract_urls_from_markdown, safe_slug_from_url

    if not link_md.exists():
        return set(), set(), set()
    urls = extract_urls_from_markdown(link_md.read_text(encoding="utf-8", errors="ignore"))
    slugs = {safe_slug_from_url(url) for url in urls}
    html_paths = {f"html/{slug}.html" for slug in slugs}
    txt_paths = {f"web/{slug}.txt" for slug in slugs}
    chunk_paths = {f"web/{slug}.jsonl" for slug in slugs}
    return html_paths, txt_paths, chunk_paths

# Tính danh sách file pdf hợp lệ cần tồn tại
def expected_pdf_relative_paths(pdf_root: Path) -> tuple[set[str], set[str]]:
    from build_pdf_txt import analyze_pdf_extractability

    txt_paths: set[str] = set()
    chunk_paths: set[str] = set()
    if not pdf_root.exists():
        return txt_paths, chunk_paths
    for pdf_path in pdf_root.rglob("*.pdf"):
        diagnostics = analyze_pdf_extractability(pdf_path)
        if not diagnostics.get("extractable"):
            continue
        year = guess_year(pdf_path)
        stem = slugify(pdf_path.stem)
        txt_rel = f"{year}/{stem}.txt"
        txt_paths.add(txt_rel)
        chunk_paths.add(f"{year}/{stem}.jsonl")
    return txt_paths, chunk_paths

# Lấy ra tập các prefix top-level theo năm của PDF.
def pdf_relative_prefixes(pdf_root: Path) -> set[str]:
    prefixes: set[str] = set()
    if not pdf_root.exists():
        return prefixes
    for pdf_path in pdf_root.rglob("*.pdf"):
        prefixes.add(guess_year(pdf_path))
    return prefixes

# Lấy workspace_slug để làm việc với AnythingLLM
def load_workspace_slug(env_file: Path, override_slug: str | None) -> str:
    from ingest_anythingllm import load_env_file

    env = load_env_file(env_file)
    return (override_slug or env.get("WORKSPACE_SLUG", "")).strip()

# Quyết định bộ thông số chunk cuối cùng sẽ dùng
def resolve_chunk_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    profile = CHUNK_PROFILES[args.chunk_profile]
    min_chars = args.min_chars if args.min_chars is not None else profile["min_chars"]
    max_chars = args.max_chars if args.max_chars is not None else profile["max_chars"]
    max_sections = (
        args.max_sections_per_chunk
        if args.max_sections_per_chunk is not None
        else profile["max_sections_per_chunk"]
    )
    return min_chars, max_chars, max_sections

# Đếm tổng số dòng JSONL chunk
def count_chunk_rows(chunk_root: Path, scope: str) -> int:
    search_root = chunk_root / scope if scope in {"web", "pdf"} and (chunk_root / scope).exists() else chunk_root
    if not search_root.exists():
        return 0
    total = 0
    for jsonl_path in search_root.rglob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
            total += sum(1 for line in handle if line.strip())
    return total

# Ghi ra file rag_baseline.json chứa toàn bộ cấu hình baseline hiện tại
def write_shared_baseline_config(
    project_root: Path,
    args: argparse.Namespace,
    chunk_root: Path,
    txt_root: Path,
    min_chars: int,
    max_chars: int,
    max_sections: int,
) -> Path:
    config_path = project_root / "rag_baseline.json"
    payload = {
        "name": "ntu-pure-rag-baseline",
        "version": 1,
        "corpus": {
            "scope": args.scope,
            "txt_root": str(txt_root.resolve()),
            "chunk_root": str(chunk_root.resolve()),
            "chunk_profile": args.chunk_profile,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "max_sections_per_chunk": max_sections,
            "chunk_count": count_chunk_rows(chunk_root=chunk_root, scope=args.scope),
        },
        "models": {
            "chat": args.chat_model,
            "embedding": args.embed_model,
        },
        "retrieval": {
            "top_n": args.top_n,
            "similarity_threshold": args.similarity_threshold,
            "vector_search_mode": args.vector_search_mode,
            "chat_mode": args.chat_mode,
        },
        "generation": {
            "temperature": args.temperature,
            "history": args.history,
        },
        "prompt": args.prompt,
        "query_refusal_response": args.query_refusal_response,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path

# Lấy danh sách model hiện có trong Ollama
def list_ollama_models() -> set[str]:
    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    models: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0].strip())
    return models

# Đảm bảo model Ollama đã có sẵn
def ensure_ollama_model(model_name: str) -> bool:
    available = list_ollama_models()
    if model_name in available:
        print(f"Ollama model ready: {model_name}")
        return False
    print(f"Pulling Ollama model: {model_name}")
    subprocess.run(["ollama", "pull", model_name], check=True)
    return True

# Cập nhật file .env dạng KEY=VALUE
def update_key_value_env(path: Path, updates: dict[str, str]) -> bool:
    if not path.exists():
        return False
    original_lines = path.read_text(encoding="utf-8").splitlines()
    rendered: list[str] = []
    seen: set[str] = set()
    changed = False
    for line in original_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key not in updates:
            rendered.append(line)
            continue
        seen.add(key)
        new_line = f"{key}='{updates[key]}'"
        rendered.append(new_line)
        changed = changed or (line != new_line)
    for key, value in updates.items():
        if key in seen:
            continue
        rendered.append(f"{key}='{value}'")
        changed = True
    if changed:
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return changed

# Đẩy setting sang AnythingLLM
def apply_anythingllm_settings(args: argparse.Namespace) -> list[str]:
    from ingest_anythingllm import AnythingLLMClient, load_env_file

    env = load_env_file(Path(args.env_file))
    base_url = env.get("ANYTHINGLLM_BASE_URL", "").strip()
    api_key = env.get("ANYTHINGLLM_API_KEY", "").strip()
    workspace_slug = load_workspace_slug(Path(args.env_file), args.workspace_slug)
    if not base_url or not api_key or not workspace_slug:
        raise SystemExit("Missing AnythingLLM config for applying settings.")

    if not args.skip_model_pull:
        ensure_ollama_model(args.chat_model)
        ensure_ollama_model(args.embed_model)

    client = AnythingLLMClient(base_url=base_url, api_key=api_key)
    if not client.ping():
        raise SystemExit("AnythingLLM API is offline.")

    workspace = client.get_workspace(workspace_slug)
    payload = {
        "chatProvider": "ollama",
        "chatModel": args.chat_model,
        "agentProvider": "ollama",
        "agentModel": args.chat_model,
        "openAiTemp": args.temperature,
        "openAiHistory": args.history,
        "similarityThreshold": args.similarity_threshold,
        "topN": args.top_n,
        "chatMode": args.chat_mode,
        "vectorSearchMode": args.vector_search_mode,
        "openAiPrompt": args.prompt,
        "queryRefusalResponse": args.query_refusal_response,
    }
    client.update_workspace(workspace_slug, payload)

    notes = [
        f"Workspace settings applied: {workspace_slug}",
        f"Chat model: {workspace.get('chatModel')} -> {args.chat_model}",
    ]

    system_settings = client.get_system_settings()
    current_embed_model = str(system_settings.get("EmbeddingModelPref") or "").strip()
    current_chat_model = str(system_settings.get("OllamaLLMModelPref") or system_settings.get("LLMModel") or "").strip()
    if current_embed_model != args.embed_model or current_chat_model != args.chat_model:
        desktop_env = Path.home() / "AppData/Roaming/anythingllm-desktop/storage/.env"
        updated = update_key_value_env(
            desktop_env,
            {
                "EMBEDDING_ENGINE": "ollama",
                "EMBEDDING_MODEL_PREF": args.embed_model,
                "LLM_PROVIDER": "ollama",
                "OLLAMA_MODEL_PREF": args.chat_model,
                "ANYTHINGLLM_MODEL_PREF": args.chat_model,
            },
        )
        if updated:
            notes.append(
                f"Desktop env updated for embedding/chat defaults: {desktop_env} (restart AnythingLLM to fully apply)."
            )
    return notes


def main() -> None:
    # 1) Parse tham số dòng lệnh (CLI).
    args = parse_args()

    # 2) Resolve các đường dẫn. (Mặc định các path là tương đối, nên thường chạy trong folder extract_md/.)
    project_root = Path(__file__).resolve().parents[1]
    pdf_root = Path(args.pdf_root)
    link_md = Path(args.link_md)
    cache_root = Path(args.web_cache_root)
    txt_root = Path(args.txt_root)
    chunk_root = Path(args.chunk_root)

    # 3) Quyết định chạy stage PDF/Web theo --scope.
    run_pdf, run_web = resolve_stage_flags(args.scope)

    # 4) Resolve cấu hình chunk theo profile + các override (nếu có).
    min_chars, max_chars, max_sections = resolve_chunk_settings(args)

    # 5) Gom tóm tắt ngắn theo từng stage để in cuối chương trình.
    summary: list[str] = []

    if args.sync:
        # Stage: dọn output “stale” để khớp input hiện tại (link.md + PDF đang có).
        removed = cleanup_temp_dirs(project_root=project_root)
        html_expected, web_txt_expected, web_chunk_expected = expected_web_relative_paths(link_md=link_md)
        pdf_txt_expected, pdf_chunk_expected = expected_pdf_relative_paths(pdf_root=pdf_root)
        pdf_prefixes = pdf_relative_prefixes(pdf_root=pdf_root)

        # Xóa HTML cache bị thừa (data_raw/web_links/html/*.html).
        if run_web:
            removed += sync_files(cache_root, html_expected, ".html")

        # Xóa TXT/JSONL stale. Nếu chạy cả web + pdf thì cho phép dọn across mọi prefix.
        if run_web and run_pdf:
            expected_txt = set(web_txt_expected) | set(pdf_txt_expected)
            expected_chunk = set(web_chunk_expected) | set(pdf_chunk_expected)
            removed += sync_files(txt_root, expected_txt, ".txt")
            removed += sync_files(chunk_root, expected_chunk, ".jsonl")
        else:
            # Nếu chỉ chạy 1 scope, chỉ dọn trong các prefix top-level tương ứng để tránh xóa nhầm.
            if run_web:
                removed += sync_files(txt_root, web_txt_expected, ".txt", allowed_prefixes={"web"})
                removed += sync_files(chunk_root, web_chunk_expected, ".jsonl", allowed_prefixes={"web"})
            if run_pdf:
                removed += sync_files(txt_root, pdf_txt_expected, ".txt", allowed_prefixes=pdf_prefixes)
                removed += sync_files(chunk_root, pdf_chunk_expected, ".jsonl", allowed_prefixes=pdf_prefixes)
        summary.append(f"Sync removed: {removed}")

    if run_pdf:
        # Stage: PDF -> TXT (data_txt/<năm>/*.txt).
        from build_pdf_txt import process_pdfs

        total_pdfs, written_pdfs = process_pdfs(pdf_root=pdf_root, out_root=txt_root)
        summary.append(f"PDF: {written_pdfs}/{total_pdfs}")

    if run_web:
        # Stage: link.md -> cache HTML -> TXT (data_txt/web/*.txt).
        if link_md.exists():
            from build_links_txt import process_link_md

            total_links, written_links = process_link_md(
                link_md=link_md,
                out_root=txt_root,
                cache_root=cache_root,
                allow_fetch=not args.offline,
                render_js=args.render_js,
            )
            summary.append(f"Web: {written_links}/{total_links}")
        else:
            print(f"[WARN] link.md not found: {link_md}")
            summary.append("Web: 0/0")

    if not args.skip_chunk:
        # Stage: TXT -> JSONL chunks (data_chunks/**.jsonl).
        from chunk_txt import process_txt_files

        chunked_files, chunk_count = process_txt_files(
            txt_root=txt_root,
            chunk_root=chunk_root,
            min_chars=min_chars,
            max_chars=max_chars,
            max_sections_per_chunk=max_sections,
        )
        summary.append(
            f"Chunks: {chunk_count} rows from {chunked_files} files "
            f"(profile={args.chunk_profile}, min={min_chars}, max={max_chars}, sections={max_sections})"
        )

    # Stage: ghi lại baseline config (để dễ tái lập và benchmark).
    baseline_config_path = write_shared_baseline_config(
        project_root=project_root,
        args=args,
        chunk_root=chunk_root,
        txt_root=txt_root,
        min_chars=min_chars,
        max_chars=max_chars,
        max_sections=max_sections,
    )
    summary.append(f"Baseline config: {baseline_config_path}")

    if args.apply_settings:
        # Stage: cập nhật settings workspace AnythingLLM (model, prompt, retrieval params).
        notes = apply_anythingllm_settings(args)
        summary.extend(notes)

    if args.ingest:
        # Stage: ingest chunk JSONL lên AnythingLLM qua API.
        from ingest_anythingllm import ingest_chunk_root

        ingest_stats = ingest_chunk_root(
            env_file=Path(args.env_file),
            chunk_root=chunk_root,
            label=args.label,
            workspace_slug=args.workspace_slug,
            force=args.force_ingest,
            dry_run=args.dry_run_ingest,
        )
        workspace_slug = load_workspace_slug(Path(args.env_file), args.workspace_slug)
        summary.append(
            "Ingest: "
            f"{ingest_stats['success']} ok, {ingest_stats['failed']} fail, {ingest_stats['skipped_existing']} skipped"
            f" -> {workspace_slug}"
        )

    if summary:
        # In tóm tắt gọn theo stage (dễ copy log).
        print("Build summary: " + " | ".join(summary))


if __name__ == "__main__":
    main()
