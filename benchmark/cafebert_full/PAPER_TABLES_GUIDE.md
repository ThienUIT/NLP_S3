# Lọc và tổng hợp `full_results.csv` cho bảng kết quả chính

File nguồn là `benchmark/cafebert_full/reference/full_results.csv`. Đây là bảng run-level gồm 480 hàng, tương ứng với bốn corpus, sáu mô hình, bốn seed và năm mức số topic. Không sửa trực tiếp file này. Mọi bảng cho bài báo phải được sinh bằng script để vẫn truy lại được nguồn.

## 1. Kiểm tra artifact trước khi trích số liệu

Từ root repository, chạy:

```bash
make cafebert-reference-audit
python -m benchmark.cafebert_full.build_paper_tables
```

Lệnh audit phải trả về `status: pass`, `rows: 480` và không có issue. Script thứ hai dừng ngay nếu thiếu hàng, có `status` khác `ok`, sai model/seed/$k$, có hash cấu hình khác nhau hoặc có nhiều hơn một hàng cho cùng một tổ hợp corpus--model--seed--$k$.

Các bảng được ghi vào `benchmark/cafebert_full/reference/paper_tables/`.

## 2. Quy tắc dùng số liệu trong phần Results

| Câu hỏi trong bài báo | Lọc hoặc tổng hợp | Cột dùng | Bảng sinh ra |
|---|---|---|---|
| Kết quả chính theo từng $k$ | `seed == 42`, giữ toàn bộ 4 corpus × 5 $k$ × 6 model | `wec_in` | `table_main_wec_in_seed42.tex` |
| Chất lượng bổ sung | Cùng lát cắt seed 42 | `topic_diversity` | `table_companion_diversity_seed42.tex` |
| Độ ổn định | Nhóm theo `corpus`, `n_topics`, `model`; mean và sample SD trên 11/29/42/47 | `wec_in` | `table_sensitivity_wec_in_mean_sd.tex` |
| Kiểm tra độ nhạy metric | Cùng nhóm bốn seed | `c_npmi` | `table_appendix_cnmpi_mean_sd.tex` |
| Chi phí phân tích topic khi đã có cache | Cùng nhóm bốn seed | `fit_seconds` | `table_timing_fit_seconds_mean_sd.tex` |

`wec_in` là coherence chính. `topic_diversity` là chỉ số kèm theo. `c_npmi` được báo cáo để minh bạch, nhưng không được dùng riêng để chọn mô hình thắng. Bảng chính không được chỉ giữ một mức $k$ tốt nhất sau khi xem kết quả; five $k$ đã được khóa trước là 10, 20, 30, 40 và 50.

## 3. Bảng nên đưa vào thân bài

Đưa **WEC-in, seed 42** vào bảng chính. Bảng có 20 hàng (bốn corpus × năm $k$) và sáu cột model. Phần kết quả có thể nêu rõ: “Bảng báo cáo một seed chính đã định trước; bảng mean $\pm$ SD bốn seed được đặt ở phần độ nhạy.”

Đưa **topic diversity** ngay sau bảng WEC-in hoặc ở phụ lục ngắn. Nếu số trang hạn chế, giữ WEC-in trong thân bài và chuyển diversity, C_NPMI, cùng timing detailed sang phụ lục.

Các giá trị in đậm trong hai bảng seed-42 là giá trị lớn nhất của đúng một hàng corpus--$k$. In đậm chỉ giúp đọc bảng. Nó không phải kiểm định thống kê và không đủ để kết luận mô hình hơn mô hình khác.

## 4. Câu chữ cho timing

`fit_seconds` chỉ đo thời gian fit/phân tích topic sau khi representation đã có. Nó trả lời câu hỏi chi phí thuật toán topic modeling trong điều kiện cache warm. Không gọi số này là “end-to-end” hoặc “thời gian train toàn bộ”.

`pipeline_seconds` là representation cộng fit trong pipeline cold-reference. `total_cold_seconds` còn cộng ingest/preprocess và nạp encoder. Với LDA/NMF, representation là CountVectorizer; với S³ và BERTopic+UMAP+KMeans, representation là CafeBERT mean pooling. Vì vậy các cột timing phải được nêu đúng stage và không dùng để claim S³ nhanh nhất một cách vô điều kiện.

## 5. Chèn vào LaTeX

Các file `.tex` dùng `booktabs`. Ví dụ:

```latex
\usepackage{booktabs}
\input{benchmark/cafebert_full/reference/paper_tables/table_main_wec_in_seed42.tex}
\input{benchmark/cafebert_full/reference/paper_tables/table_sensitivity_wec_in_mean_sd.tex}
```

Nếu luận văn không dùng đường dẫn repository làm đường dẫn build, copy các file `.tex` đã sinh vào thư mục `tables/` của luận văn. Không nhập số bằng tay.

## 6. Truy vết một ô trong bảng

Mỗi ô seed-42 có thể truy lại bằng filter sau:

```python
import pandas as pd

df = pd.read_csv("benchmark/cafebert_full/reference/full_results.csv")
cell = df.query(
    "corpus == 'visfd' and model == 's3_combined' and seed == 42 and n_topics == 20"
)
print(cell[["wec_in", "topic_diversity", "c_npmi", "fit_seconds", "pipeline_seconds", "total_cold_seconds", "document_ids_sha256", "config_sha256"]])
```

Để kiểm tra cả ma trận, dùng `make cafebert-reference-audit`, không chỉ đếm số dòng CSV.
