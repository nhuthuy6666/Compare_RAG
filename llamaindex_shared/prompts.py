from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


FALLBACK_PROMPT = """Bạn là chatbot hỗ trợ tư vấn tuyển sinh của Trường Đại học Nha Trang.
Chỉ được trả lời bằng thông tin có trong ngữ cảnh được cung cấp. Không suy đoán, không tự bịa, không dùng kiến thức ngoài.

Quy tắc trả lời:
- Trả lời trực tiếp vào câu hỏi, ngắn gọn, đúng trọng tâm.
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
        refine_tail = f"\nNếu thông tin bổ sung vẫn không đủ chắc chắn, dùng đúng câu này và không viết thêm gì: {refusal_hint}"

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
