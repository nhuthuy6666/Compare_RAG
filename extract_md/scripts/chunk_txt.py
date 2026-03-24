import argparse
import re
from pathlib import Path

from corpus_utils import build_chunk_rows, write_jsonl


YEAR_DIR_RE = re.compile(r"^20\d{2}$")
WEB_DIR = "web"


# Liệt kê các file TXT hợp lệ (nhóm PDF theo năm và nhóm web).
def iter_txt_files(txt_root: Path) -> list[Path]:
    candidates = sorted(txt_root.rglob("*.txt"))
    if not candidates:
        return []
    txts: list[Path] = []
    for path in candidates:
        rel = path.relative_to(txt_root)
        if not rel.parts:
            continue
        # PDF outputs are written under data_txt/<year>/..., web outputs under data_txt/web/...
        if YEAR_DIR_RE.fullmatch(rel.parts[0]) or rel.parts[0].lower() == WEB_DIR:
            txts.append(path)
    return txts


# Chunk toàn bộ TXT thành JSONL theo cấu hình độ dài chunk.
def process_txt_files(
    txt_root: Path,
    chunk_root: Path,
    min_chars: int,
    max_chars: int,
    max_sections_per_chunk: int,
) -> tuple[int, int]:
    # B1: Lấy danh sách TXT hợp lệ từ nguồn data_txt.
    txt_files = iter_txt_files(txt_root)
    if not txt_files:
        print(f"No TXT files found in {txt_root}")
        return 0, 0

    # B2: Duyệt từng TXT -> tạo rows chunk -> ghi JSONL theo rel path.
    chunked_files = 0
    chunk_count = 0
    for txt_path in txt_files:
        rel_path = txt_path.relative_to(txt_root)
        out_path = (chunk_root / rel_path).with_suffix(".jsonl")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        rows = build_chunk_rows(
            text=text,
            source_txt=txt_path,
            min_chars=min_chars,
            max_chars=max_chars,
            max_sections_per_chunk=max_sections_per_chunk,
        )
        write_jsonl(out_path, rows)
        chunked_files += 1
        chunk_count += len(rows)
        print(f"Chunked TXT: {txt_path} -> {out_path} ({len(rows)} chunks)")
    # B3: Trả về số file đã chunk và tổng số chunk.
    return chunked_files, chunk_count


# Điểm vào CLI cho quy trình TXT -> chunk JSONL.
def main() -> None:
    # 1) Khai báo tham số CLI.
    parser = argparse.ArgumentParser(description="Chunk PDF-derived TXT files into JSONL files.")
    parser.add_argument("--txt-root", default="data_txt", help="Root folder containing TXT files.")
    parser.add_argument("--chunk-root", default="data_chunks", help="Output folder for chunk JSONL files.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum characters before merging tiny sections.")
    parser.add_argument("--max-chars", type=int, default=700, help="Maximum characters for a chunk.")
    parser.add_argument(
        "--max-sections-per-chunk",
        type=int,
        default=2,
        help="Maximum adjacent sections merged into a chunk.",
    )

    # 2) Parse args.
    args = parser.parse_args()

    # 3) Chạy pipeline TXT -> JSONL chunks.
    process_txt_files(
        txt_root=Path(args.txt_root),
        chunk_root=Path(args.chunk_root),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_sections_per_chunk=args.max_sections_per_chunk,
    )


if __name__ == "__main__":
    main()
