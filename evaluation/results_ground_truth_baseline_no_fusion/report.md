# Ground-Truth Evaluation for Baseline No Fusion

- System: `baseline`
- Profile: `shared_no_fusion_v1`
- Split: `all`
- Samples: `72`
- Corpus nodes indexed in-memory: `835`

## Summary

- Overall score: `0.7454`
- Faithfulness: `0.7802`
- Hallucination rate: `0.2198`
- Recall@k: `0.8863`
- Context precision: `0.534`
- Invalid rate: `0.25`
- Hallucination case rate: `0.1944` | severe: `0.0556`
- Retrieval failure rate: `0.1389`
- False refusal rate: `0.0278` | missed refusal rate: `0.0833`
- Grounding gap rate: `0.0278`

## Worst Topics

- ielts: invalid=1.0, hallucination=0.0, retrieval_failure=0.6667, overall=0.6772
- refusal: invalid=0.8571, hallucination=0.4286, retrieval_failure=1.0, overall=0.3107
- consulting: invalid=0.4286, hallucination=0.1429, retrieval_failure=0.0, overall=0.8125
- scope: invalid=0.3333, hallucination=0.3333, retrieval_failure=0.0, overall=0.816
- contact: invalid=0.2857, hallucination=0.2143, retrieval_failure=0.0714, overall=0.767
- scores: invalid=0.2, hallucination=0.0, retrieval_failure=0.0, overall=0.8571
- quota: invalid=0.0, hallucination=0.4286, retrieval_failure=0.0, overall=0.7293
- identity: invalid=0.0, hallucination=0.3333, retrieval_failure=0.0, overall=0.6779

## Worst Examples

- out_of_scope_bus [refusal]: tags=missed_refusal, invalid, severe_hallucination, retrieval_failure, severity=0.9441, faithfulness=0.0, hallucination=1.0, recall@k=0.0
  Q: Trường có xe buýt miễn phí đón tân sinh viên mỗi ngày không?
  A: Tôi chưa đủ căn cứ để trả lời câu hỏi này từ dữ liệu hiện có.
- out_of_scope_airport_shuttle [refusal]: tags=missed_refusal, invalid, severe_hallucination, retrieval_failure, severity=0.8434, faithfulness=0.2, hallucination=0.8, recall@k=0.0
  Q: Trường có xe đưa đón miễn phí từ sân bay Cam Ranh cho tân sinh viên không?
  A: Xe đưa đón miễn phí từ sân bay Cam Ranh là một trong những dịch vụ hỗ trợ mà trường Đại học Nha Trang dành cho tân sinh viên. Tuy nhiên, để biết thêm thông tin chi tiết về lịch trình và các điều kiện cụ thể liên quan...
- school_code [identity]: tags=severe_hallucination, severity=0.7202, faithfulness=0.0, hallucination=1.0, recall@k=0.7528
  Q: Mã cơ sở đào tạo trong tuyển sinh của trường là gì?
  A: Mã cơ sở đào tạo trong tuyển sinh của Trường Đại học Nha Trang là 5905.
- out_of_scope_free_dorm [refusal]: tags=missed_refusal, invalid, hallucination, retrieval_failure, severity=0.7007, faithfulness=0.5, hallucination=0.5, recall@k=0.0
  Q: Trường có bố trí ký túc xá miễn phí cho toàn bộ sinh viên năm nhất không?
  A: Số chỗ ở ký túc xá của trường Đại học Nha Trang là 3.500, đáp ứng đầy đủ nhu cầu của tất cả sinh viên nhập học.  Lưu ý rằng ngoài số chỗ ở ký túc xá miễn phí, trường còn có các chính sách hỗ trợ khác như:  - Hỗ trợ KT...
- quota_qtkd_special_2024 [quota]: tags=severe_hallucination, grounding_gap, severity=0.6619, faithfulness=0.0, hallucination=1.0, recall@k=1.0
  Q: Năm 2024, Quản trị kinh doanh chương trình đào tạo đặc biệt có bao nhiêu chỉ tiêu?
  A: Tôi không thể tìm thấy thông tin về số lượng chỉ tiêu của chương trình đào tạo đặc biệt "Quản trị kinh doanh" trong tài liệu đã được cung cấp.
- out_of_scope_job_guarantee [refusal]: tags=missed_refusal, invalid, retrieval_failure, severity=0.6376, faithfulness=0.625, hallucination=0.375, recall@k=0.0
  Q: Trường có cam kết 100% sinh viên mọi ngành đều có việc làm ngay sau khi tốt nghiệp không?
  A: Trường Đại học Nha Trang đã cam kết hỗ trợ việc làm cho sinh viên. Tuy nhiên, thông tin cụ thể về tỷ lệ sinh viên có việc làm ngay sau khi tốt nghiệp không được đề cập trong các tài liệu liên quan.  Tuy nhiên, theo Bả...
- out_of_scope_scholarship [refusal]: tags=missed_refusal, invalid, retrieval_failure, severity=0.6116, faithfulness=0.6667, hallucination=0.3333, recall@k=0.0
  Q: Trường có học bổng du học Nhật Bản toàn phần cho tân sinh viên không?
  A: Trường Đại học Nha Trang có chính sách học bổng, khen thưởng đối với thủ khoa, á khoa toàn trường và các ngành đào tạo. Tuy nhiên, thông tin cụ thể về việc có học bổng du học Nhật Bản toàn phần cho tân sinh viên hay k...
- phone_secondary [contact]: tags=invalid, hallucination, severity=0.5531, faithfulness=0.3333, hallucination=0.6667, recall@k=1.0
  Q: Trên trang phương thức, đối tượng, phạm vi tuyển sinh có số điện thoại nào khác?
  A: Ngoài ra, bạn cũng có thể liên hệ với chúng tôi qua số điện thoại 0258.2212579 hoặc email tuyensinh@ntu.edu.vn để được tư vấn và hỗ trợ thêm về quá trình tuyển sinh.
- out_of_scope_dorm_fee [refusal]: tags=missed_refusal, invalid, retrieval_failure, severity=0.5471, faithfulness=0.8, hallucination=0.2, recall@k=0.0
  Q: Phí ở ký túc xá mỗi tháng là bao nhiêu?
  A: Phí ở ký túc xá tại Trường Đại học Nha Trang không được đề cập trong các tài liệu có sẵn. Tuy nhiên, bạn có thể tham khảo thông tin về hệ thống ký túc xá của trường trên trang web trungtampvth.ntu.edu.vn.  Lưu ý rằng...
- quota_khoa_hoc_thuy_san_2025 [quota]: tags=hallucination, severity=0.4196, faithfulness=0.5, hallucination=0.5, recall@k=1.0
  Q: Năm 2025, ngành Khoa học thủy sản có bao nhiêu chỉ tiêu?
  A: Để trả lời câu hỏi này, chúng ta cần tìm kiếm thông tin về chỉ tiêu của ngành Khoa học thủy sản trong năm 2025.  Trong tài liệu được cung cấp, có một số phần liên quan đến điểm trúng tuyển và chỉ tiêu của các ngành kh...
