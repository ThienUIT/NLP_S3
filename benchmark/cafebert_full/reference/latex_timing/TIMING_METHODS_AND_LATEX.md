# Timing appendix: CafeBERT benchmark

## Đơn vị báo cáo

Mỗi ô timing trước hết là trung bình qua `k = 10, 20, 30, 40, 50` trong cùng một tổ hợp **corpus × mô hình × seed**. Sau đó bảng báo `mean ± sample SD` qua bốn seed đã đăng ký trước: 11, 29, 42 và 47. Vì vậy, `n=4` cho độ lệch chuẩn; mỗi ô tóm tắt 20 phép chạy thô. Các kết quả chỉ áp dụng cho môi trường được ghi trong `environment.json`, corpus manifest và cấu hình hash `4bbba1f8131d9c8ed741219255d2985be219ecc9b9368ad84e025bcac1cd840b`.

| Stage ghi trong CSV | Nghĩa vận hành | Có thể diễn giải |
|---|---|---|
| `ingest_preprocess_seconds_shared` | Đọc corpus và tiền xử lý đã khóa, phân bổ theo corpus trong runner | Chi phí chuẩn bị văn bản, không phải thời gian fit topic |
| `encoder_model_load_seconds_shared` | Nạp checkpoint CafeBERT cục bộ cho hàng dùng CafeBERT; bằng 0 cho LDA/NMF | Chi phí khởi tạo encoder, không bao gồm tải mạng |
| `representation_seconds_cold_reference` | Mã hóa CafeBERT mean pooling cho S³/BERTopic, hoặc `CountVectorizer.fit_transform` cho LDA/NMF | Chi phí tạo biểu diễn lạnh theo lớp mô hình |
| `fit_seconds` | Fit mô hình và trích top-term sau khi representation sẵn sàng | So sánh chi phí phân tích/trích topic *warm cache* |
| `pipeline_seconds` | `representation_seconds_cold_reference + fit_seconds` | Chi phí pipeline sau khi corpus/model đã sẵn sàng, không gồm metric |
| `total_cold_seconds` | `ingest + model load + pipeline` | Cold-start có thể so sánh trong cấu hình hiện tại |
| `metric_seconds` | WEC-in, diversity, C_NPMI và audit alphabetic sau khi trích topic | Chi phí đánh giá, không được cộng vào pipeline hoặc fit |

## Quy tắc viết kết quả

Không gọi `fit_seconds` là “thời gian end-to-end”. Đây là thời gian fit và trích topic khi biểu diễn đã sẵn sàng. Khi dùng `pipeline_seconds`, phải ghi “cold-reference pipeline”, vì representation lạnh được đo một lần cho mỗi corpus/lớp biểu diễn và sao chép vào các hàng tương đương để không thay cache hit bằng 0 giây. `total_cold_seconds` thêm nạp dữ liệu, tiền xử lý và nạp model cục bộ. Network download bị loại khỏi toàn bộ phép đo.

LDA và NMF dùng `CountVectorizer`; S³ và BERTopic + UMAP + KMeans dùng CafeBERT. Vì vậy bảng tổng thời gian so sánh các pipeline mô hình hoàn chỉnh, không phải encoder ablation. Một claim hợp lệ có dạng: “Trên máy và cấu hình đã ghi, S³ combined có median/mean `fit_seconds` thấp hơn [phương pháp] trong [corpus]”. Không được suy ra rằng S³ luôn nhanh hơn mọi baseline hoặc nhanh hơn trên phần cứng khác.

## Tệp đầu ra

* `cafebert_stage_timing_summary.csv`: dữ liệu bảng đã tổng hợp, có mean/SD theo stage.
* `table_cafebert_timing.tex`: bảng LaTeX 24 hàng, đầy đủ corpus × model.
* `table_cafebert_timing_compact.tex`: bốn bảng LaTeX nhỏ theo corpus, dành cho phụ lục hoặc xoay ngang.
* `timing_validation.json`: các kiểm tra tái hợp thức stage, coverage và quy tắc gộp.

## Chèn vào LaTeX

Trong preamble, thêm `\usepackage{booktabs}`, `\usepackage{longtable}`, `\usepackage{graphicx}` và `\usepackage{pdflscape}`. Bảng 24 hàng trong `table_cafebert_timing.tex` đã được đặt trong `landscape`; bảng compact tự co theo `\linewidth`. Không sửa số trực tiếp trong `.tex`; khi chạy lại benchmark hoặc thay CSV, chạy lại `python3 generate_cafebert_timing_appendix.py`.
