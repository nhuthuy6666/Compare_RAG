from __future__ import annotations

import argparse

from chat_service import answer_question
from utils import configure_console_utf8


## Khai báo CLI cho chế độ query nhanh hoặc interactive mode.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truy vấn GraphRAG lưu trong Qdrant.")
    parser.add_argument("--question", help="Đặt một câu hỏi rồi thoát.")
    parser.add_argument("--top-k", type=int, help="Số graph fact đưa vào ngữ cảnh. Mặc định lấy từ baseline.")
    return parser.parse_args()


## In ra kết quả trả lời duy nhất để giao diện dòng lệnh nhất quán với web UI.
def print_response(result) -> None:
    print("\nTrả lời:")
    print(result.answer)


## Entry point CLI cho GraphRAG.
## 1) Cấu hình console UTF-8.
## 2) Parse tham số dòng lệnh.
## 3) Nếu có `--question` thì trả lời một lần rồi thoát.
## 4) Nếu không có `--question` thì vào vòng lặp interactive cho tới khi người dùng gõ `exit` hoặc `quit`.
def main() -> None:
    configure_console_utf8()
    args = parse_args()

    if args.question:
        print_response(answer_question(args.question, top_k=args.top_k))
        return

    while True:
        question = input("Bạn hỏi: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        print_response(answer_question(question, top_k=args.top_k))
        print("-" * 60)


if __name__ == "__main__":
    main()
