import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import fitz
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: pymupdf. Install it with `python -m pip install pymupdf`."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ModuleNotFoundError:
    RapidOCR = None

from corpus_utils import (
    extract_heading_info,
    filter_noise_lines,
    finalize_document,
    guess_year,
    heading_level_from_text,
    normalize_line,
    render_table_lines,
    slugify,
    source_title_from_stem,
)


ROMAN_SECTION_RE = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)[.)]?\s+\S+", re.IGNORECASE)
OCR_MARKERS = ("tesseract",)
_RAPIDOCR_ENGINE = None


# File này chịu trách nhiệm chuyển PDF sang TXT “markdown-lite”:
# - Trích text theo block, cố gắng sắp lại thứ tự đọc (1 cột/2 cột).
# - Trích bảng (nếu PyMuPDF hỗ trợ find_tables).
# - Fallback OCR khi trang không có text layer.
def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    # Kiểm tra 2 bounding box có giao nhau không (dùng để loại text nằm trong vùng bảng).
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


# Chuẩn hóa text của một block (loại dòng rỗng, normalize ký tự).
def normalize_block_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        cleaned = normalize_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return lines


def cluster_blocks_by_x(blocks: list[dict], tolerance: float) -> list[list[dict]]:
    # Gom các block text theo vị trí X (ước lượng cột trong PDF).
    bands: list[list[dict]] = []
    for block in sorted(blocks, key=lambda item: item["x0"]):
        placed = False
        for band in bands:
            anchor = sum(item["x0"] for item in band) / len(band)
            if abs(block["x0"] - anchor) <= tolerance:
                band.append(block)
                placed = True
                break
        if not placed:
            bands.append([block])
    return [sorted(band, key=lambda item: (item["y0"], item["x0"])) for band in bands]


def reorder_text_blocks(blocks: list[tuple], page_width: float) -> list[dict]:
    # Sắp lại thứ tự block text theo “luồng đọc” (hữu ích cho PDF 2 cột).
    prepared: list[dict] = []
    for block in blocks:
        lines = normalize_block_lines(block[4])
        if not lines:
            continue
        prepared.append(
            {
                "x0": block[0],
                "y0": block[1],
                "x1": block[2],
                "y1": block[3],
                "width": block[2] - block[0],
                "lines": lines,
            }
        )

    if not prepared:
        return []

    narrow_limit = page_width * 0.55
    narrow_blocks = [block for block in prepared if block["width"] <= narrow_limit]
    if len(narrow_blocks) < 6:
        return sorted(prepared, key=lambda item: (item["y0"], item["x0"]))

    x_tolerance = max(36.0, page_width * 0.06)
    x_bands = cluster_blocks_by_x(narrow_blocks, tolerance=x_tolerance)
    meaningful_bands = [band for band in x_bands if len(band) >= 2]
    if len(meaningful_bands) < 2:
        return sorted(prepared, key=lambda item: (item["y0"], item["x0"]))

    bands = [list(band) for band in sorted(meaningful_bands, key=lambda band: min(item["x0"] for item in band))]
    band_blocks = {id(block) for band in bands for block in band}
    remaining_blocks = [block for block in prepared if id(block) not in band_blocks]

    band_top = min(block["y0"] for band in bands for block in band)
    band_bottom = max(block["y1"] for band in bands for block in band)
    top_wide: list[dict] = []
    bottom_wide: list[dict] = []
    band_anchors = [sum(item["x0"] for item in band) / len(band) for band in bands]

    for block in remaining_blocks:
        if block["width"] >= page_width * 0.75 and block["y1"] <= band_top + 20:
            top_wide.append(block)
            continue
        if block["width"] >= page_width * 0.75 and block["y0"] >= band_bottom - 20:
            bottom_wide.append(block)
            continue
        nearest_index = min(range(len(bands)), key=lambda idx: abs(block["x0"] - band_anchors[idx]))
        bands[nearest_index].append(block)

    flattened: list[dict] = sorted(top_wide, key=lambda item: (item["y0"], item["x0"]))
    for band in bands:
        flattened.extend(sorted(band, key=lambda item: (item["y0"], item["x0"])))
    flattened.extend(sorted(bottom_wide, key=lambda item: (item["y0"], item["x0"])))
    return flattened


def _ocr_backend_available() -> bool:
    # Kiểm tra có backend OCR khả dụng (tesseract hoặc RapidOCR + numpy).
    return any(shutil.which(marker) for marker in OCR_MARKERS) or (RapidOCR is not None and np is not None)


def _get_rapidocr_engine():
    # Khởi tạo/lấy singleton RapidOCR engine (nếu dependency có sẵn).
    global _RAPIDOCR_ENGINE
    if RapidOCR is None or np is None:
        return None
    if _RAPIDOCR_ENGINE is None:
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _ocr_page_lines(page) -> list[str]:
    # OCR 1 trang: ưu tiên tesseract (nếu có), fallback RapidOCR.
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            image_path = temp_dir_path / "page.png"
            output_base = temp_dir_path / "ocr"

            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pixmap.save(image_path)

            try:
                subprocess.run(
                    [tesseract_bin, str(image_path), str(output_base), "-l", "vie+eng"],
                    check=True,
                    capture_output=True,
                )
            except Exception:
                pass
            else:
                txt_path = output_base.with_suffix(".txt")
                if txt_path.exists():
                    lines = normalize_block_lines(txt_path.read_text(encoding="utf-8", errors="ignore"))
                    if lines:
                        return lines

    engine = _get_rapidocr_engine()
    if engine is None or np is None:
        return []

    try:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        result, _ = engine(image)
    except Exception:
        return []

    if not result:
        return []

    ordered = sorted(result, key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])))
    lines = [normalize_line(item[1]) for item in ordered if len(item) >= 2 and normalize_line(item[1])]
    return lines


def _extract_page_lines(page) -> tuple[list[str], bool]:
    # Trích lines + tables cho 1 trang; nếu không có text layer thì thử OCR.
    table_boxes: list[tuple[float, float, float, float]] = []
    tables: list[list[list[str]]] = []
    if hasattr(page, "find_tables"):
        try:
            found = page.find_tables()
            for table in found.tables:
                table_boxes.append(tuple(table.bbox))
                rows = table.extract()
                if rows:
                    tables.append(rows)
        except Exception:
            pass

    blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
    lines: list[str] = []
    text_blocks: list[tuple] = []
    for block in blocks:
        bbox = (block[0], block[1], block[2], block[3])
        if any(intersects(bbox, table_box) for table_box in table_boxes):
            continue
        text_blocks.append(block)

    for block in reorder_text_blocks(text_blocks, page_width=page.rect.width):
        lines.extend(block["lines"])

    used_ocr = False
    if not lines:
        ocr_lines = _ocr_page_lines(page)
        if ocr_lines:
            lines = ocr_lines
            used_ocr = True

    return [{"lines": lines, "tables": tables}], used_ocr


def analyze_pdf_extractability(pdf_path: Path) -> dict:
    # Quick scan để biết PDF có trích được text/bảng không (để quyết định có build TXT hay skip).
    summary = {
        "page_count": 0,
        "text_pages": 0,
        "ocr_pages": 0,
        "table_pages": 0,
        "extractable": False,
        "reason": "",
    }

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        summary["reason"] = f"cannot_open: {exc}"
        return summary

    try:
        summary["page_count"] = document.page_count
        if document.page_count == 0:
            summary["reason"] = "zero_pages"
            return summary

        for page in document:
            page_payloads, used_ocr = _extract_page_lines(page)
            payload = page_payloads[0]
            if payload["lines"]:
                summary["text_pages"] += 1
            if used_ocr:
                summary["ocr_pages"] += 1
            if payload["tables"]:
                summary["table_pages"] += 1

        summary["extractable"] = bool(summary["text_pages"] or summary["table_pages"])
        if not summary["extractable"]:
            summary["reason"] = "no_extractable_text_or_tables"
        elif summary["ocr_pages"] > 0:
            summary["reason"] = "ocr_used"
        else:
            summary["reason"] = "ok"
        return summary
    finally:
        document.close()


def extract_pdf_pages(pdf_path: Path) -> tuple[list[dict], dict]:
    # Trích toàn bộ trang thành payload: {"lines": [...], "tables": [...]} + thống kê chẩn đoán.
    pages: list[dict] = []
    diagnostics = {
        "page_count": 0,
        "text_pages": 0,
        "ocr_pages": 0,
        "table_pages": 0,
    }

    document = fitz.open(pdf_path)
    try:
        diagnostics["page_count"] = document.page_count
        for page in document:
            page_payloads, used_ocr = _extract_page_lines(page)
            payload = page_payloads[0]
            pages.append(payload)
            if payload["lines"]:
                diagnostics["text_pages"] += 1
            if used_ocr:
                diagnostics["ocr_pages"] += 1
            if payload["tables"]:
                diagnostics["table_pages"] += 1
    finally:
        document.close()
    return pages, diagnostics


def pdf_pages_to_lines(pages: list[dict]) -> list[str]:
    # Chuyển danh sách pages -> các dòng “markdown-lite”: heading/bullets/bảng.
    page_lines = filter_noise_lines([page["lines"] for page in pages])
    page_tables = [page["tables"] for page in pages]

    output: list[str] = []
    skip_toc = False
    content_started = False
    previous_heading_level = 1
    table_index = 1
    preface_buffer: list[str] = []
    preface_chars = 0

    def flush_preface() -> None:
        nonlocal preface_buffer, preface_chars, content_started
        if not preface_buffer:
            return
        output.extend(preface_buffer)
        preface_buffer = []
        preface_chars = 0
        content_started = True

    for page_index, lines in enumerate(page_lines):
        # 1) Duyệt line text để phát hiện bắt đầu nội dung, bỏ mục lục, và suy ra heading/bullets.
        for raw_line in lines:
            line = normalize_line(raw_line)
            if not line:
                continue

            upper = line.upper()
            if "MỤC LỤC" in upper or "MUC LUC" in upper:
                skip_toc = True
                continue
            if skip_toc:
                if ROMAN_SECTION_RE.match(line):
                    skip_toc = False
                else:
                    continue

            if line.startswith(("-", "*", "•")):
                bullet_line = f"- {line.lstrip('-*• ').strip()}"
                if content_started:
                    output.append(bullet_line)
                else:
                    preface_buffer.append(bullet_line)
                    preface_chars += len(bullet_line)
                continue

            info = extract_heading_info(line)
            if not content_started:
                if info and info["kind"] in {"roman", "numeric"}:
                    flush_preface()
                elif heading_level_from_text(line) == 2:
                    flush_preface()
                else:
                    preface_buffer.append(line)
                    preface_chars += len(line)
                    if preface_chars >= 200 or len(preface_buffer) >= 4:
                        flush_preface()
                    continue

            if info:
                output.append(f"{'#' * info['level']} {info['text']}")
                previous_heading_level = info["level"]
                continue

            uppercase_level = heading_level_from_text(line)
            if uppercase_level == 2 and line == line.upper():
                derived_level = min(previous_heading_level + 1, 4)
                output.append(f"{'#' * derived_level} {line}")
                previous_heading_level = derived_level
                continue

            output.append(line)

        if not content_started and not preface_buffer:
            continue

        # 2) Append bảng (nếu có) sau phần text của trang.
        for table_rows in page_tables[page_index]:
            output.extend(render_table_lines(table_rows, table_index))
            table_index += 1

    if not output and preface_buffer:
        output.extend(preface_buffer)

    return output


def build_pdf_txt(pdf_path: Path, out_root: Path) -> Path | None:
    # 1) Resolve năm + tiêu đề + đường dẫn output (data_txt/<năm>/<slug>.txt).
    year = guess_year(pdf_path)
    title = source_title_from_stem(pdf_path.stem)
    out_path = out_root / year / f"{slugify(pdf_path.stem)}.txt"

    # 2) Trích nội dung theo trang (text blocks + tables; OCR nếu cần).
    try:
        pages, diagnostics = extract_pdf_pages(pdf_path)
    except Exception as exc:
        if out_path.exists():
            out_path.unlink()
        print(f"[WARN] Skip PDF {pdf_path}: extract failed ({exc})")
        return None

    # 3) Guard: file lỗi/không có trang hợp lệ.
    if diagnostics["page_count"] == 0:
        if out_path.exists():
            out_path.unlink()
        print(f"[WARN] Skip PDF {pdf_path}: file has 0 pages or is unreadable for text extraction.")
        return None

    # 4) Chuyển pages -> danh sách dòng (lọc noise, xử lý mục lục, heading, bảng).
    body_lines = pdf_pages_to_lines(pages)
    meaningful_lines = [line for line in body_lines if line.strip()]
    if not meaningful_lines:
        if out_path.exists():
            out_path.unlink()
        extra = ""
        if not _ocr_backend_available():
            extra = " OCR backend not available."
        print(
            f"[WARN] Skip PDF {pdf_path}: no extractable text after removing cover/empty pages."
            f"{extra}"
        )
        return None

    # 5) Finalize thành document TXT hoàn chỉnh và ghi ra disk.
    text = finalize_document(title=title, body_lines=body_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def process_pdfs(pdf_root: Path, out_root: Path) -> tuple[int, int]:
    # Duyệt tất cả PDF dưới pdf_root và build TXT; PDF lỗi thì skip.
    pdf_files = sorted(pdf_root.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_root}")
        return 0, 0

    written = 0
    for pdf_path in pdf_files:
        out_path = build_pdf_txt(pdf_path, out_root)
        if out_path is None:
            continue
        written += 1
        print(f"Built PDF TXT: {pdf_path} -> {out_path}")
    return len(pdf_files), written


def main() -> None:
    # 1) Khai báo tham số CLI.
    parser = argparse.ArgumentParser(description="Convert PDF files directly to TXT markdown-lite files.")
    parser.add_argument("--pdf-root", default="data_raw/pdf", help="Input folder containing PDF files.")
    parser.add_argument("--out-root", default="data_txt", help="Output folder for TXT files.")

    # 2) Parse args.
    args = parser.parse_args()

    # 3) Chạy pipeline PDF -> TXT.
    process_pdfs(pdf_root=Path(args.pdf_root), out_root=Path(args.out_root))


if __name__ == "__main__":
    main()
