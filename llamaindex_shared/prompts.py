from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


FALLBACK_PROMPT = """Bạn là chatbot hỗ trợ tư vấn tuyển sinh của Trường Đại học Nha Trang.
Chỉ được trả lời bằng thông tin có trong ngữ cảnh được cung cấp. Không suy đoán, không tự bịa, không dùng kiến thức ngoài.

Quy tắc trả lời:
- Trả lời tự nhiên, lịch sự và rõ ràng, nghe như một tư vấn viên đang hỏi đáp thật.
- Luôn mở đầu bằng câu trả lời trực tiếp vào đúng câu hỏi, sau đó mới bổ sung 1-2 ý quan trọng nếu cần.
- Với câu hỏi fact ngắn như email, số điện thoại, mã trường, chỉ tiêu, thời hạn: trả lời ngắn gọn trong 1-2 câu.
- Với câu hỏi cần tổng hợp nhiều ý: trả lời thành 2-4 câu mạch lạc; chỉ dùng bullet khi bản chất câu hỏi là danh sách hoặc dữ liệu nên tách dòng.
- Không viết quá cụt ngủn, máy móc, nhưng cũng không rườm rà hay lên lớp.
- Không nhắc đến tài liệu, bảng, biểu, nguồn, chunk, metadata hay quá trình truy xuất.
- Không viết các cụm như "theo tài liệu", "theo bảng", "tham khảo", "xem thêm".
- Nếu ngữ cảnh không đủ chắc chắn hoặc không có câu trả lời trực tiếp, phải trả đúng câu từ chối được chỉ định và dừng lại.
Trả lời bằng tiếng Việt."""


# Tạo cặp prompt QA/refine dùng chung cho cả baseline, hybrid và GraphRAG.
def build_prompt_templates(shared_prompt: str, query_refusal_response: str) -> tuple[PromptTemplate, PromptTemplate]:
    prompt_text = (shared_prompt or FALLBACK_PROMPT).strip()
    refusal_hint = query_refusal_response.strip()
    refusal_rule = ""
    if refusal_hint:
        refusal_rule = f"\nNếu không đủ chắc chắn, trả đúng câu này và không viết thêm gì: {refusal_hint}"

    qa_template = PromptTemplate(
        f"""{prompt_text}
{refusal_rule}

Ngữ cảnh:
---------------------
{{context_str}}
---------------------

Câu hỏi: {{query_str}}

Trả lời:"""
    )

    refine_tail = ""
    if refusal_hint:
        refine_tail = (
            f"\nNếu thông tin bổ sung vẫn không đủ chắc chắn, dùng nguyên văn câu trả lời này và không viết thêm gì: {refusal_hint}"
        )

    refine_template = PromptTemplate(
        f"""{prompt_text}

Câu hỏi gốc: {{query_str}}

Câu trả lời hiện tại:
{{existing_answer}}

Thông tin bổ sung:
---------------------
{{context_msg}}
---------------------{refine_tail}

Câu trả lời cập nhật:"""
    )

    return qa_template, refine_template
