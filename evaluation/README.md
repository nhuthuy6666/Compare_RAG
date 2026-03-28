# Evaluation

Thư mục này dùng cho flow đánh giá V1 của 3 hệ:

- `baseline`: project `rag`
- `hybrid`: project `hybrid_rag`
- `graphrag`: project `graph_rag`

Cả ba hệ cùng dùng:

- dữ liệu từ `extract_md/rag_baseline.json`
- embedding `bge-m3:latest`
- chat model `llama3.1:8b`
- cùng bộ test V1 cân bằng trong `evaluation/dataset/testset.json` + `evaluation/dataset/testset_additions_v1.json`

## Bộ Test V1 Cân Bằng

Bộ test V1 hiện có tổng `72` câu, chia đều thành `3` nhóm:

- `24` câu `dense_shared`
- `24` câu `graph`
- `24` câu `neutral`

Bucket được gán tự động khi load dataset và được ghi ra báo cáo sau mỗi lần evaluate.

## Công Thức Overall

`Overall = 0.25 * Recall@k + 0.35 * Faithfulness + 0.25 * Answer_Relevancy + 0.15 * Context_Precision`

Với thiết lập hiện tại:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`

Giải thích chi tiết metric nằm trong `evaluation/METRICS_V1.md`.

## Quy Trình Benchmark Mới

- `dev` và `held_out_test` được tách cố định trong `evaluation/dataset/splits_v1.json`
- `controlled same-settings`: `python evaluation/evaluate-v1.py --mode controlled --split held_out_test`
- `best-tuned-per-architecture`: `python evaluation/evaluate-v1.py --mode best_tuned --split held_out_test`
- có thể evaluate candidate profile riêng cho từng hệ:
  `python evaluation/evaluate-v1.py --system hybrid --mode best_tuned --profile-source candidate --profile-name hybrid_no_fusion --split dev`
- tuning protocol thật trên `dev` để chọn candidate tốt nhất và khóa lại vào manifest:
  `python evaluation/tune_v1.py --write-locks`
- repeated study với nhiều lần chạy + CI + significance:
  `python evaluation/study_v1.py --mode controlled --split held_out_test`
- workflow human judgments:
  `python evaluation/judgments.py prepare`
  rồi
  `python evaluation/judgments.py report`

Policy benchmark nằm tại `evaluation/benchmark_policy_v1.json`, candidate profile nằm tại `evaluation/profile_candidates_v1.json`, profile khóa nằm tại `evaluation/locked_profiles_v1.json`.

Mặc định `python evaluation/evaluate-v1.py ...` giờ chỉ ghi file kết quả chính là `evaluation/results_v1/comparison.md`.
Nếu cần giữ thêm CSV/JSON trung gian để debug hoặc chạy study/tuning nâng cao, dùng thêm cờ `--keep-artifacts`.

## Chạy Ra Kết Quả Cuối

Làm đúng thứ tự dưới đây để ra bộ kết quả cuối cùng.

### 1. Mở terminal 1 và vào repo

Repo này:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project_fusion
```

Repo song song:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project
```

Nếu dùng virtualenv, kích hoạt trước khi chạy lệnh Python.

### 2. Bật 3 RAG server

Chạy trong terminal 1 và giữ nguyên cửa sổ này:

```bash
python run_all_rags.py
```

Khi ổn, bạn sẽ thấy 3 địa chỉ:

- `hybrid`: `http://127.0.0.1:8000/`
- `baseline`: `http://127.0.0.1:8001/`
- `graphrag`: `http://127.0.0.1:8502/`

### 3. Mở terminal 2 để tuning trên `dev`

Repo này:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project_fusion
python evaluation/tune_v1.py --write-locks
```

Repo còn lại:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project
python evaluation/tune_v1.py --write-locks
```

Lệnh này sẽ:

- chỉ chạy trên `dev`
- duyệt các candidate profile trong `evaluation/profile_candidates_v1.json`
- chọn candidate tốt nhất theo protocol ngân sách hiện tại
- ghi profile thắng cuộc vào `evaluation/locked_profiles_v1.json`

### 4. Chạy kết quả cuối cho `controlled same-settings`

```bash
python evaluation/evaluate-v1.py --mode controlled --split held_out_test --run-label final_controlled
```

### 5. Chạy kết quả cuối cho `best-tuned-per-architecture`

```bash
python evaluation/evaluate-v1.py --mode best_tuned --split held_out_test --run-label final_best_tuned
```

### 6. Chạy repeated study để lấy mean, std, CI và significance

Controlled:

```bash
python evaluation/study_v1.py --mode controlled --split held_out_test --label final_controlled_study
```

Best tuned:

```bash
python evaluation/study_v1.py --mode best_tuned --split held_out_test --label final_best_tuned_study
```

### 7. Xuất bảng và báo cáo tổng hợp

```bash
python evaluation/compare.py
python evaluation/visualize.py
python evaluation/reliability.py
```

### 8. Các file cần xem sau khi chạy xong

- `evaluation/results_v1/comparison.csv`
- `evaluation/results_v1/comparison.json`
- `evaluation/results_v1/strength_breakdown.csv`
- `evaluation/results_v1/comparison.md`
- `evaluation/studies_v1/final_controlled_study/study.md`
- `evaluation/studies_v1/final_best_tuned_study/study.md`
- `evaluation/locked_profiles_v1.json`

### 9. Lệnh ngắn gọn nếu chỉ muốn chạy full pipeline

Sau khi đã bật `python run_all_rags.py` ở terminal 1, terminal 2 chạy lần lượt:

```bash
python evaluation/tune_v1.py --write-locks
python evaluation/evaluate-v1.py --mode controlled --split held_out_test --run-label final_controlled
python evaluation/evaluate-v1.py --mode best_tuned --split held_out_test --run-label final_best_tuned
python evaluation/study_v1.py --mode controlled --split held_out_test --label final_controlled_study
python evaluation/study_v1.py --mode best_tuned --split held_out_test --label final_best_tuned_study
python evaluation/compare.py
python evaluation/visualize.py
python evaluation/reliability.py
```

### 10. Note quan trọng

- Không sửa prompt, profile hay code sau khi đã nhìn kết quả `held_out_test`, nếu không kết quả đó không còn là kết quả cuối nữa.
- Muốn tune lại thì quay về bước 3, chạy trên `dev`, khóa profile lại, rồi mới chạy lại `held_out_test`.
- `comparison.md` chỉ phản ánh kết quả mới nhất trong `evaluation/results_v1/`.
- Nếu muốn làm cho cả 2 repo, hãy lặp lại toàn bộ quy trình trên trong từng repo riêng.

## Chạy Đủ 72 Câu Trong Một Lần

Nếu mục tiêu của bạn là chấm toàn bộ `72` câu trong một lần, hãy dùng `--split all`.

Phải dùng `2` terminal:

Terminal 1:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project_fusion
python run_all_rags.py
```

Terminal 2:

```bash
cd c:\Users\Admin\OneDrive\Desktop\workspace\llm_project_fusion
python evaluation/evaluate-v1.py --mode controlled --split all --run-label final_controlled_all72
python evaluation/evaluate-v1.py --mode best_tuned --split all --run-label final_best_tuned_all72
python evaluation/compare.py
python evaluation/reliability.py --split all
```

Giải thích ngắn:

- terminal 1 để bật `3` server RAG
- terminal 2 để chạy benchmark
- `--split all` nghĩa là chạy trên toàn bộ `49 + 23 = 72` câu

## Tạo Báo Cáo Từ Kết Quả V1

```bash
python evaluation/compare.py
python evaluation/visualize.py
python evaluation/reliability.py
```

Các script trên mặc định dùng `evaluation/config_v1.yaml`.
`reliability.py` dùng cùng `load_examples()` với `evaluate-v1`, nên cũng nạp `testset_additions_v1.json`, áp dụng `strength_buckets_v1.json` và căn metric rows theo `example_id` trước khi bootstrap.

## File Quan Trọng

- `evaluation/evaluate-v1.py`: entrypoint benchmark V1
- `evaluation/config_v1.yaml`: cấu hình dataset, endpoint và thư mục kết quả
- `evaluation/METRICS_V1.md`: mô tả metric và công thức
- `evaluation/outputs_v1/`: output gốc của từng hệ sau khi chạy
- `evaluation/results_v1/`: file metric và báo cáo tổng hợp sau khi chạy
- `evaluation/results_v1/strength_breakdown.csv`: breakdown theo `dense_shared`, `graph`, `neutral`
- `evaluation/dataset/testset_additions_v1.json`: câu bổ sung để nâng bộ test lên 72 câu
- `evaluation/dataset/strength_buckets_v1.json`: các override bucket cho những câu không thể suy ra chỉ từ `topic`

## Ghi Chú

- Thư mục đã được dọn sạch phần legacy để chỉ còn V1.
- Trước mỗi lần benchmark mới, bạn có thể xóa nội dung trong `outputs_v1/` và `results_v1/` hoặc để `evaluate-v1` ghi đè kết quả cũ.
