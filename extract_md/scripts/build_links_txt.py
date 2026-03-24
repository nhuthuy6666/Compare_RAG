import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from corpus_utils import (
    classify_text_line,
    finalize_document,
    normalize_line,
    normalize_text,
    render_table_lines,
    slugify,
)


URL_RE = re.compile(r"https?://[^\s)>\"]+")
HEADING_MD_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
TABLE_HEADING_RE = re.compile(r"^Bang\s+\d+$", re.IGNORECASE)
CONTENT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "table")


# Loại bỏ URL trùng lặp nhưng vẫn giữ thứ tự xuất hiện ban đầu.
def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = value.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# Trích tất cả URL từ nội dung markdown.
def extract_urls_from_markdown(text: str) -> list[str]:
    urls: list[str] = []
    for line in text.splitlines():
        urls.extend(URL_RE.findall(line))
    return dedupe_preserve_order(urls)


# Tạo slug an toàn từ URL để dùng làm tên file cache/TXT.
def safe_slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw = f"{parsed.netloc}{unquote(parsed.path or '')}"
    candidate = slugify(raw)
    if candidate:
        return candidate[:120]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"link-{digest}"


# Tải HTML bằng trình duyệt (Playwright) để lấy nội dung tab/collapse (render JS).
def fetch_html_rendered(url: str, timeout: int) -> str:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    timeout_ms = max(timeout, 5) * 1000
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1200)

            # Cố gắng click các tab/collapse phổ biến để hiện thêm nội dung ẩn.
            page.evaluate(
                """
                () => {
                  const selectors = [
                    '[data-toggle="tab"]',
                    '[data-bs-toggle="tab"]',
                    '[role="tab"]',
                    '[data-toggle="collapse"]',
                    '[data-bs-toggle="collapse"]',
                    '.accordion-button',
                    '.nav-tabs a'
                  ];
                  const click = (el) => {
                    try { el.click(); } catch (_) {}
                    try {
                      el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                    } catch (_) {}
                  };

                  for (let round = 0; round < 3; round++) {
                    for (const selector of selectors) {
                      for (const el of document.querySelectorAll(selector)) {
                        click(el);
                      }
                    }
                  }

                  for (const detail of document.querySelectorAll('details')) {
                    detail.open = true;
                  }
                }
                """
            )
            page.wait_for_timeout(1200)
            return page.content()
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Timeout rendering HTML: {url}") from exc
        finally:
            browser.close()


# Tải HTML từ web và cố gắng giải mã theo charset khai báo.
def fetch_html(url: str, timeout: int, render_js: bool = False) -> str:
    if render_js:
        return fetch_html_rendered(url, timeout=timeout)

    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ExtractMdBot/1.0"})
    resp.raise_for_status()
    content = resp.content or b""
    header = content[:4096]
    match = re.search(br"charset=([A-Za-z0-9_\-]+)", header)
    if match:
        enc = match.group(1).decode("ascii", errors="ignore").lower()
        try:
            return content.decode(enc, errors="replace")
        except LookupError:
            pass
    return content.decode(resp.encoding or "utf-8", errors="replace")


# Đọc HTML từ cache nếu có; nếu không thì fetch và lưu cache.
def load_or_fetch_html(
    url: str,
    cache_path: Path,
    allow_fetch: bool,
    timeout: int,
    render_js: bool = False,
) -> str:
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8", errors="ignore")
        if cached.strip():
            return cached

    if not allow_fetch:
        raise RuntimeError(f"Missing cached HTML (offline): {cache_path}")

    html = fetch_html(url, timeout=timeout, render_js=render_js)
    if not html.strip():
        raise RuntimeError(f"Fetched HTML is empty: {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


# Chọn các node “nội dung chính” (có thể nhiều khối/tab) theo thứ tự ưu tiên.
def pick_main_containers(soup) -> list:
    selectors = (
        "section.our-agent-single",
        "div.ss_event",
        "div[id^='dnn_BText']",
        "div.DnnModule-HyperLinks",
        "div.DnnModule-Article",
        "div.DNNContainer_Title_h2",
        "div.DNNContainer_Title_h3",
        "div.contents.detail",
        "div.contents",
        "article",
        "main",
        "#dnn_ContentPane",
        "#content",
        ".entry-content",
        ".post",
        ".content",
    )

    def is_descendant(node, ancestor) -> bool:
        parent = getattr(node, "parent", None)
        while parent is not None:
            if parent == ancestor:
                return True
            parent = getattr(parent, "parent", None)
        return False

    def is_rich_container(node) -> bool:
        text = normalize_line(node.get_text(" ", strip=True))
        structured_blocks = len(node.find_all(CONTENT_TAGS, recursive=True))
        return structured_blocks >= 3 or len(text) >= 300

    picked: list = []
    for selector in selectors:
        for node in soup.select(selector):
            # Bo qua node nam ben trong container da chon de tranh parse trung.
            if any(is_descendant(node, existing) for existing in picked):
                continue
            # Bo qua node bao ngoai mot container da chon (giu container cu the hon).
            if any(is_descendant(existing, node) for existing in picked):
                continue
            picked.append(node)

    if picked:
        rich_picked = [node for node in picked if is_rich_container(node)]
        return rich_picked or picked
    return [soup.body or soup]


def safe_span(value: str | None) -> int:
    try:
        parsed = int((value or "1").strip())
    except ValueError:
        return 1
    return max(parsed, 1)


def extract_table_rows(table) -> list[list[str]]:
    active_spans: dict[int, tuple[int, str]] = {}
    rows: list[list[str]] = []

    def consume_spans(row: list[str], start_col: int) -> int:
        col_idx = start_col
        while col_idx in active_spans:
            remaining, value = active_spans[col_idx]
            row.append(value)
            if remaining <= 1:
                del active_spans[col_idx]
            else:
                active_spans[col_idx] = (remaining - 1, value)
            col_idx += 1
        return col_idx

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue

        row: list[str] = []
        col_idx = consume_spans(row, 0)
        for cell in cells:
            col_idx = consume_spans(row, col_idx)
            value = normalize_line(cell.get_text(" ", strip=True))
            colspan = safe_span(cell.get("colspan"))
            rowspan = safe_span(cell.get("rowspan"))
            for span_offset in range(colspan):
                cell_value = value if span_offset == 0 else ""
                row.append(cell_value)
                if rowspan > 1:
                    active_spans[col_idx + span_offset] = (rowspan - 1, cell_value)
            col_idx += colspan
        consume_spans(row, col_idx)
        if any(value for value in row):
            rows.append(row)

    return rows


# Chuyển HTML thành các dòng text markdown-lite, gồm heading/list/table.
def html_to_lines(html: str) -> list[str]:
    from bs4 import BeautifulSoup

    # Parse HTML và loại các node không đóng góp nội dung.
    soup = BeautifulSoup(html, "lxml")
    # DNN/ASP.NET pages often wrap everything in a <form>, so do NOT drop forms.
    for tag in soup.select("script, style, noscript, svg, canvas"):
        tag.decompose()
    for tag in soup.select("input, button, textarea, select, option"):
        tag.decompose()

    containers = pick_main_containers(soup)
    lines: list[str] = []
    table_index = 1
    visited_nodes: set[int] = set()
    # Duyệt các node content theo thứ tự để chuyển thành các dòng markdown-lite.
    for container in containers:
        for node in container.find_all(
            list(CONTENT_TAGS),
            recursive=True,
        ):
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            if node.name != "table" and node.find_parent("table") is not None:
                continue

            if node.name == "table":
                table_rows = extract_table_rows(node)
                if table_rows:
                    if lines and lines[-1].strip():
                        lines.append("")
                    lines.extend(render_table_lines(table_rows, table_index))
                    table_index += 1
                continue

            text = normalize_line(node.get_text(" ", strip=True))
            if not text:
                continue
            if node.name == "li":
                classified = f"- {text.lstrip('-*• ').strip()}"
            else:
                classified = classify_text_line(text)

            if classified and (not lines or lines[-1] != classified):
                lines.append(classified)

    # Nếu không lấy được node có cấu trúc, fallback về plain text toàn vùng.
    if not lines:
        fallback = "\n".join(container.get_text("\n", strip=True) for container in containers)
        cleaned = normalize_text(fallback)
        return [line for line in cleaned.splitlines() if line.strip()]

    # Cân bằng cấp heading cho web: tránh trường hợp # -> ### mà không có ##.
    lines = rebalance_heading_levels(lines)
    cleaned = normalize_text("\n".join(lines))
    return [line for line in cleaned.splitlines() if line.strip()]


# Nếu tài liệu không có heading ##, hạ đồng loạt heading nội dung xuống cho liền mạch.
def rebalance_heading_levels(lines: list[str]) -> list[str]:
    non_table_levels: list[int] = []
    for line in lines:
        match = HEADING_MD_RE.match(line)
        if not match:
            continue
        text = match.group(2).strip()
        if TABLE_HEADING_RE.fullmatch(text):
            continue
        non_table_levels.append(len(match.group(1)))

    if not non_table_levels:
        return lines

    min_level = min(non_table_levels)
    if min_level <= 2:
        return lines
    shift = min_level - 2

    balanced: list[str] = []
    for line in lines:
        match = HEADING_MD_RE.match(line)
        if not match:
            balanced.append(line)
            continue
        marks, text = match.group(1), match.group(2).strip()
        if TABLE_HEADING_RE.fullmatch(text):
            balanced.append(line)
            continue
        new_level = max(2, len(marks) - shift)
        balanced.append(f"{'#' * new_level} {text}")
    return balanced


# Build một file TXT từ một URL web (có cache HTML).
def build_web_txt(
    url: str,
    out_root: Path,
    cache_root: Path,
    allow_fetch: bool,
    timeout: int,
    render_js: bool = False,
) -> Path:
    from bs4 import BeautifulSoup

    # B1: Resolve slug + cache path và nạp HTML (ưu tiên cache).
    slug = safe_slug_from_url(url)
    cache_path = cache_root / "html" / f"{slug}.html"
    html = load_or_fetch_html(
        url,
        cache_path=cache_path,
        allow_fetch=allow_fetch,
        timeout=timeout,
        render_js=render_js,
    )

    # B2: Rút tiêu đề + body lines rồi finalize thành document text.
    soup = BeautifulSoup(html, "lxml")
    title = normalize_line(soup.title.get_text(" ", strip=True) if soup.title else "") or slug
    body_lines = [f"Source: {url}", *html_to_lines(html)]
    text = finalize_document(title=title, body_lines=body_lines)

    # B3: Ghi TXT vào data_txt/web/<slug>.txt.
    out_path = out_root / "web" / f"{slug}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


# Đọc link.md, build TXT cho từng URL và trả về thống kê.
def process_link_md(
    link_md: Path,
    out_root: Path,
    cache_root: Path,
    allow_fetch: bool,
    timeout: int = 20,
    render_js: bool = False,
) -> tuple[int, int]:
    if not link_md.exists():
        print(f"[WARN] link.md not found: {link_md}")
        return 0, 0

    # B1: Đọc và trích danh sách URL duy nhất từ link.md.
    urls = extract_urls_from_markdown(link_md.read_text(encoding="utf-8", errors="ignore"))
    if not urls:
        print(f"[WARN] No URLs found in {link_md}")
        return 0, 0

    # B2: Build TXT cho từng URL; URL lỗi thì bỏ qua nhưng pipeline vẫn chạy tiếp.
    written = 0
    for url in urls:
        try:
                out_path = build_web_txt(
                    url=url,
                    out_root=out_root,
                    cache_root=cache_root,
                    allow_fetch=allow_fetch,
                    timeout=timeout,
                    render_js=render_js,
                )
        except Exception as exc:
            print(f"[WARN] Skip URL: {url} ({exc})")
            continue
        written += 1
        print(f"Built WEB TXT: {url} -> {out_path}")

    # B3: Trả thống kê tổng URL và số URL build thành công.
    return len(urls), written


# Điểm vào CLI cho quy trình link.md -> TXT.
def main() -> None:
    # 1) Khai báo tham số CLI.
    parser = argparse.ArgumentParser(description="Fetch URLs from link.md and build TXT files (no images).")
    parser.add_argument(
        "--link-md",
        default="data_raw/web/link.md",
        help="Markdown file containing web links to fetch.",
    )
    parser.add_argument("--out-root", default="data_txt", help="Output folder for TXT files.")
    parser.add_argument("--cache-root", default="data_raw/web_links", help="Folder for cached HTML files.")
    parser.add_argument("--offline", action="store_true", help="Do not fetch; only use cached HTML.")
    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Use Playwright to render and click tabs/collapses before extracting text.",
    )
    # 2) Parse args.
    args = parser.parse_args()

    # 3) Chạy pipeline link.md -> (cache HTML) -> TXT.
    process_link_md(
        link_md=Path(args.link_md),
        out_root=Path(args.out_root),
        cache_root=Path(args.cache_root),
        allow_fetch=not args.offline,
        render_js=args.render_js,
    )


if __name__ == "__main__":
    main()
