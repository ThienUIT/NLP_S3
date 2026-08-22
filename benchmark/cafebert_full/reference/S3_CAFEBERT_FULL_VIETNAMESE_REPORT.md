# Báo cáo benchmark S³ với CafeBERT trên bốn corpus tiếng Việt

## Tóm tắt

Thí nghiệm đánh giá ba cấu hình **S³ — Semantic Signal Separation** (`axial`, `angular`, `combined`) cùng LDA, NMF và **BERTopic + UMAP + KMeans** trên bốn corpus tiếng Việt. Encoder cho S³ và BERTopic là checkpoint `uitnlp/CafeBERT`, pooling trung bình theo attention mask và chuẩn hóa L2. LDA/NMF dùng biểu diễn `CountVectorizer`, vì vậy đây là so sánh giữa các lớp mô hình topic, không phải ablation encoder.[1] [2]

Tập kết quả gồm **480 lần chạy thực**, tương ứng 4 corpus × 6 mô hình × 5 giá trị k × 4 seed. Kiểm toán cấu trúc xác nhận 480/480 hàng duy nhất, trạng thái `ok`, metric hữu hạn, số topic trả về đúng yêu cầu và mỗi topic có ít nhất 10 term. WEC-in là metric coherence chính theo protocol S³; topic diversity được báo song song và C_NPMI được giữ ở phụ lục như phép kiểm tra không đồng thuận.[1]

> **Kết quả ở phạm vi đã đo.** Ở seed 42, một biến thể S³ đạt WEC-in cao nhất trong **15/20** ô *corpus × k*. Trên toàn bộ seed 11, 29, 42 và 47, S³ dẫn đầu WEC-in trong **59/80** ô; phân rã theo corpus là Vietnamese-news 0/20, UIT-ViSFD 19/20, ViMedical 20/20 và VNTC-CNTT 20/20. Điều này là bằng chứng tái lập trên bốn corpus đã chọn, không phải chứng minh tính ưu việt phổ quát của S³.

## 1. Dữ liệu và provenance

| Corpus | Tài liệu | Điều kiện lọc | SHA-256 document ID theo thứ tự | Nguồn snapshot |
|---|---:|---|---|---|
| Vietnamese-news | 858 | ≥ 10 ký tự | `4a69f5df5cc5cee9d68d807eca0f7d5d3c4bd2c42a1637f8e83d3247819a1c7d` | [huggingface.co/datasets/vanhai123/vietnamese-news-dataset](https://huggingface.co/datasets/vanhai123/vietnamese-news-dataset) |
| UIT-ViSFD | 10,000 | ≥ 10 ký tự | `c719fdf08e863348f772bc3900112d2708ed736eb526037ad3dd5b5e8f2bdfb8` | [github.com/LuongPhan/UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD) |
| ViMedical Disease | 12,060 | ≥ 30 ký tự | `15f5601e6cb64d461ded353f61131a7395abd361ca4fce9959f42a19b966d614` | [github.com/PB3002/ViMedical_Disease](https://github.com/PB3002/ViMedical_Disease) |
| VNTC-CNTT | 3,571 | ≥ 150 ký tự | `6fbe5bfbdf1b963816a421668db82d426899d6689adcc353fde88d9d6ff61212` | [github.com/duyvuleo/VNTC](https://github.com/duyvuleo/VNTC) |

Các nhãn sentiment trong UIT-ViSFD và nhãn trong Vietnamese-news không được đưa vào bước fit. Do đó, kết quả topic không được diễn giải như accuracy phân loại sentiment hoặc nhãn chủ đề có sẵn. Các manifest, hash document ID, revision nguồn và giấy phép đang lưu cùng kết quả để tái kiểm tra.

## 2. Protocol tái lập đã khóa

| Thành phần | Thiết lập |
|---|---|
| Mô hình S³ | `SemanticSignalSeparation` của Turftopic; `feature_importance = axial / angular / combined` [2] |
| Encoder ngữ nghĩa | `uitnlp/CafeBERT`; attention-mask mean pooling, L2-normalized, `max_length=256`, `batch_size=32`, xử lý batch theo độ dài tăng dần rồi trả về đúng thứ tự tài liệu [3] |
| Baseline lexical | LDA và NMF, `CountVectorizer(min_df=10)`, token pattern Unicode có hỗ trợ dấu gạch nối |
| Baseline embedding | BERTopic + UMAP (`n_neighbors=15`, `n_components=5`, `min_dist=0`, cosine) + KMeans (`n_init=10`), không phải BERTopic mặc định HDBSCAN [4] |
| Lưới | `k ∈ {10, 20, 30, 40, 50}`; 10 top-term/topic; seed chính 42; kiểm tra độ nhạy 11, 29, 42, 47 |
| WEC-in chính | Trung bình cosine của các cặp top-term trong từng topic, với Word2Vec huấn luyện trên chính corpus tiếng Việt |
| Chỉ số đi kèm | Topic diversity và alphabetic-term rate |
| Robustness phụ lục | Gensim C_NPMI. Metric này không dùng để chọn winner hoặc tạo chỉ số gộp |
| WEC-ex | **N/A**. Google News Word2Vec không phù hợp tiếng Việt |
| Điều kiện BERTopic | Vectorizer tạo topic của BERTopic dùng `min_df=2`, do `min_df=10` gây lỗi c-TF-IDF suy biến trong smoke test corpus 858 tài liệu. Thiết lập này không thay đổi tokenizer chung cho S³/LDA/NMF hay WEC-in/C_NPMI. |

`pipeline_seconds` là thời gian biểu diễn lạnh tham chiếu cộng fit và trích topic; `fit_seconds` là fit/trích topic khi biểu diễn đã sẵn sàng; `total_cold_seconds` cộng thêm ingest/preprocess và thời gian load model. Biểu diễn được cache theo corpus, nhưng thời gian lạnh gốc được sao chép vào mỗi hàng cùng loại mô hình để tránh đánh giá cache hit như 0 giây. Không cộng các cột này qua hàng để suy ra wall-clock tổng.

## 3. Kết quả chính: WEC-in, seed 42

Các ô đậm là WEC-in cao nhất trong cùng *corpus × k*. Đây là bảng seed chính; độ nhạy đa seed nằm ở mục 5.

### 3.1. Vietnamese-news

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.994 | 0.993 | 0.994 | **0.998** | 0.998 | 0.960 |
| 20 | 0.994 | 0.993 | 0.994 | **0.998** | 0.997 | 0.970 |
| 30 | 0.994 | 0.993 | 0.994 | **0.997** | 0.997 | 0.965 |
| 40 | 0.994 | 0.994 | 0.994 | **0.997** | 0.996 | 0.956 |
| 50 | 0.994 | 0.994 | 0.994 | **0.997** | 0.996 | 0.944 |

### 3.2. UIT-ViSFD

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.458 | 0.447 | **0.470** | 0.357 | 0.380 | 0.426 |
| 20 | **0.523** | 0.521 | 0.515 | 0.337 | 0.389 | 0.435 |
| 30 | 0.467 | **0.503** | 0.490 | 0.357 | 0.379 | 0.468 |
| 40 | 0.526 | **0.548** | 0.527 | 0.355 | 0.385 | 0.451 |
| 50 | 0.504 | **0.521** | 0.520 | 0.370 | 0.396 | 0.424 |

### 3.3. ViMedical Disease

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.501 | 0.507 | **0.514** | 0.206 | 0.177 | 0.382 |
| 20 | 0.520 | **0.526** | 0.519 | 0.161 | 0.293 | 0.479 |
| 30 | 0.546 | **0.552** | 0.549 | 0.167 | 0.328 | 0.439 |
| 40 | 0.519 | **0.525** | 0.520 | 0.160 | 0.369 | 0.447 |
| 50 | **0.542** | 0.542 | 0.536 | 0.162 | 0.402 | 0.439 |

### 3.4. VNTC-CNTT

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | **0.262** | 0.259 | 0.260 | 0.136 | 0.123 | 0.243 |
| 20 | 0.305 | **0.314** | 0.307 | 0.144 | 0.119 | 0.218 |
| 30 | 0.276 | **0.284** | 0.276 | 0.150 | 0.124 | 0.221 |
| 40 | 0.305 | **0.318** | 0.308 | 0.147 | 0.140 | 0.208 |
| 50 | 0.269 | 0.269 | **0.273** | 0.140 | 0.136 | 0.234 |

![WEC-in trung bình ± SD qua bốn seed](cafebert-wec-in.png)

## 4. Topic diversity, seed 42

Topic diversity là tỉ lệ top-term khác nhau trên toàn bộ topic; giá trị cao không đồng nghĩa trực tiếp với coherence cao. Các ô đậm chỉ phương pháp cao nhất trong cùng *corpus × k*.

### 4.1. Vietnamese-news

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.620 | 0.610 | 0.630 | 0.740 | **0.820** | 0.800 |
| 20 | 0.590 | 0.610 | 0.590 | 0.640 | **0.735** | 0.680 |
| 30 | 0.527 | 0.530 | 0.533 | 0.597 | **0.640** | 0.587 |
| 40 | 0.430 | 0.463 | 0.440 | 0.510 | **0.542** | 0.507 |
| 50 | 0.396 | 0.426 | 0.404 | 0.452 | **0.476** | 0.444 |

### 4.2. UIT-ViSFD

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.850 | 0.870 | **0.880** | 0.630 | 0.760 | 0.390 |
| 20 | 0.835 | **0.880** | 0.825 | 0.605 | 0.685 | 0.280 |
| 30 | 0.760 | **0.850** | 0.810 | 0.537 | 0.600 | 0.230 |
| 40 | 0.790 | **0.820** | 0.802 | 0.545 | 0.535 | 0.212 |
| 50 | 0.742 | **0.862** | 0.786 | 0.532 | 0.528 | 0.212 |

### 4.3. ViMedical Disease

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | **0.790** | 0.730 | 0.770 | 0.370 | 0.450 | 0.510 |
| 20 | 0.835 | 0.840 | **0.845** | 0.285 | 0.535 | 0.485 |
| 30 | 0.770 | **0.810** | 0.783 | 0.263 | 0.490 | 0.477 |
| 40 | 0.723 | **0.767** | 0.738 | 0.305 | 0.530 | 0.458 |
| 50 | 0.682 | **0.750** | 0.702 | 0.286 | 0.532 | 0.436 |

### 4.4. VNTC-CNTT

| k | S³ axial | S³ angular | S³ combined | LDA | NMF | BERTopic + UMAP + KMeans |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | **0.900** | 0.880 | 0.900 | 0.430 | 0.590 | 0.370 |
| 20 | **0.820** | 0.795 | 0.815 | 0.470 | 0.575 | 0.360 |
| 30 | 0.903 | 0.913 | **0.923** | 0.463 | 0.563 | 0.400 |
| 40 | 0.858 | **0.882** | 0.865 | 0.427 | 0.560 | 0.378 |
| 50 | 0.808 | **0.858** | 0.828 | 0.420 | 0.562 | 0.388 |

![Topic diversity trung bình ± SD qua bốn seed](cafebert-diversity.png)

## 5. Kiểm tra độ nhạy qua bốn seed

Mỗi seed trước hết được trung bình qua năm giá trị k; bảng sau là mean ± sample SD trên seed 11, 29, 42 và 47. WEC-in là cột diễn giải chính. C_NPMI được đưa vào nguyên trạng, kể cả khi thứ hạng không trùng WEC-in.

### 5.1. Vietnamese-news

| Phương pháp | WEC-in | Diversity | C_NPMI | Fit-only (s) | Pipeline cold-ref (s) | Alphabetic rate |
|---|---:|---:|---:|---:|---:|---:|
| S³ axial | 0.994 ± 0.000 | 0.533 ± 0.017 | -0.445 ± 0.005 | 7.09 ± 0.60 | 128.65 ± 0.60 | 1.000 |
| S³ angular | 0.994 ± 0.000 | 0.535 ± 0.019 | -0.448 ± 0.002 | 7.42 ± 1.48 | 128.99 ± 1.48 | 1.000 |
| S³ combined | 0.994 ± 0.000 | 0.536 ± 0.019 | -0.447 ± 0.006 | 6.87 ± 0.53 | 128.43 ± 0.53 | 1.000 |
| LDA | 0.997 ± 0.000 | 0.581 ± 0.012 | 0.093 ± 0.009 | 0.80 ± 0.06 | 0.82 ± 0.06 | 1.000 |
| NMF | 0.997 ± 0.000 | 0.642 ± 0.001 | 0.146 ± 0.006 | 0.28 ± 0.03 | 0.29 ± 0.03 | 1.000 |
| BERTopic + UMAP + KMeans | 0.956 ± 0.003 | 0.595 ± 0.006 | -0.133 ± 0.014 | 3.64 ± 1.39 | 125.21 ± 1.39 | 0.997 |

### 5.2. UIT-ViSFD

| Phương pháp | WEC-in | Diversity | C_NPMI | Fit-only (s) | Pipeline cold-ref (s) | Alphabetic rate |
|---|---:|---:|---:|---:|---:|---:|
| S³ axial | 0.504 ± 0.013 | 0.802 ± 0.009 | -0.361 ± 0.006 | 46.95 ± 0.42 | 3086.23 ± 0.42 | 0.858 |
| S³ angular | 0.513 ± 0.014 | 0.861 ± 0.008 | -0.358 ± 0.009 | 46.56 ± 0.43 | 3085.85 ± 0.43 | 0.860 |
| S³ combined | 0.508 ± 0.013 | 0.826 ± 0.009 | -0.359 ± 0.005 | 46.26 ± 0.62 | 3085.54 ± 0.62 | 0.859 |
| LDA | 0.359 ± 0.008 | 0.592 ± 0.016 | 0.090 ± 0.005 | 25.70 ± 0.63 | 25.94 ± 0.65 | 0.998 |
| NMF | 0.386 ± 0.001 | 0.623 ± 0.002 | 0.087 ± 0.001 | 1.86 ± 0.36 | 2.10 ± 0.35 | 1.000 |
| BERTopic + UMAP + KMeans | 0.440 ± 0.006 | 0.268 ± 0.002 | -0.222 ± 0.002 | 24.33 ± 1.80 | 3063.62 ± 1.80 | 1.000 |

### 5.3. ViMedical Disease

| Phương pháp | WEC-in | Diversity | C_NPMI | Fit-only (s) | Pipeline cold-ref (s) | Alphabetic rate |
|---|---:|---:|---:|---:|---:|---:|
| S³ axial | 0.520 ± 0.005 | 0.744 ± 0.014 | -0.397 ± 0.004 | 28.80 ± 0.36 | 1803.35 ± 0.36 | 1.000 |
| S³ angular | 0.522 ± 0.006 | 0.777 ± 0.017 | -0.394 ± 0.005 | 28.20 ± 0.33 | 1802.75 ± 0.33 | 0.998 |
| S³ combined | 0.520 ± 0.006 | 0.755 ± 0.015 | -0.395 ± 0.005 | 28.22 ± 0.37 | 1802.76 ± 0.37 | 0.999 |
| LDA | 0.156 ± 0.013 | 0.284 ± 0.017 | 0.109 ± 0.005 | 21.39 ± 0.23 | 21.54 ± 0.23 | 1.000 |
| NMF | 0.314 ± 0.002 | 0.510 ± 0.004 | 0.091 ± 0.001 | 3.05 ± 0.02 | 3.20 ± 0.03 | 1.000 |
| BERTopic + UMAP + KMeans | 0.427 ± 0.012 | 0.466 ± 0.005 | -0.271 ± 0.020 | 11.62 ± 0.07 | 1786.17 ± 0.07 | 1.000 |

### 5.4. VNTC-CNTT

| Phương pháp | WEC-in | Diversity | C_NPMI | Fit-only (s) | Pipeline cold-ref (s) | Alphabetic rate |
|---|---:|---:|---:|---:|---:|---:|
| S³ axial | 0.279 ± 0.008 | 0.852 ± 0.008 | -0.316 ± 0.004 | 89.86 ± 0.65 | 4044.89 ± 0.65 | 0.910 |
| S³ angular | 0.292 ± 0.002 | 0.864 ± 0.007 | -0.319 ± 0.004 | 89.83 ± 0.78 | 4044.86 ± 0.78 | 0.889 |
| S³ combined | 0.283 ± 0.003 | 0.858 ± 0.012 | -0.317 ± 0.004 | 90.15 ± 0.49 | 4045.19 ± 0.49 | 0.900 |
| LDA | 0.137 ± 0.005 | 0.456 ± 0.013 | 0.027 ± 0.003 | 27.84 ± 0.34 | 28.54 ± 0.36 | 0.995 |
| NMF | 0.128 ± 0.001 | 0.575 ± 0.004 | 0.049 ± 0.001 | 4.04 ± 0.03 | 4.74 ± 0.04 | 0.997 |
| BERTopic + UMAP + KMeans | 0.227 ± 0.003 | 0.383 ± 0.006 | -0.201 ± 0.006 | 26.01 ± 0.07 | 3981.04 ± 0.07 | 0.998 |

## 6. Timing theo giai đoạn

Bảng này ghi mean ± SD qua seed của các trung bình theo k. Cột **fit-only warm** phù hợp khi so sánh chi phí điều chỉnh mô hình sau khi biểu diễn đã có. Cột **pipeline cold-ref** và **total cold** thêm chi phí biểu diễn; vì các phương pháp ngữ nghĩa dùng CafeBERT còn LDA/NMF dùng CountVectorizer, các cột này phải được gắn đúng định nghĩa, không gọi chung là “speed” mà không nêu stage.

| Corpus | Phương pháp | Fit-only warm (s) | Pipeline cold-ref (s) | Total cold (s) |
|---|---|---:|---:|---:|
| Vietnamese-news | S³ axial | 7.09 ± 0.60 | 128.65 ± 0.60 | 144.54 |
| Vietnamese-news | S³ angular | 7.42 ± 1.48 | 128.99 ± 1.48 | 144.88 |
| Vietnamese-news | S³ combined | 6.87 ± 0.53 | 128.43 ± 0.53 | 144.33 |
| Vietnamese-news | LDA | 0.80 ± 0.06 | 0.82 ± 0.06 | 4.26 |
| Vietnamese-news | NMF | 0.28 ± 0.03 | 0.29 ± 0.03 | 3.74 |
| Vietnamese-news | BERTopic + UMAP + KMeans | 3.64 ± 1.39 | 125.21 ± 1.39 | 141.10 |
| UIT-ViSFD | S³ axial | 46.95 ± 0.42 | 3086.23 ± 0.42 | 3102.13 |
| UIT-ViSFD | S³ angular | 46.56 ± 0.43 | 3085.85 ± 0.43 | 3101.74 |
| UIT-ViSFD | S³ combined | 46.26 ± 0.62 | 3085.54 ± 0.62 | 3101.44 |
| UIT-ViSFD | LDA | 25.70 ± 0.63 | 25.94 ± 0.65 | 29.38 |
| UIT-ViSFD | NMF | 1.86 ± 0.36 | 2.10 ± 0.35 | 5.54 |
| UIT-ViSFD | BERTopic + UMAP + KMeans | 24.33 ± 1.80 | 3063.62 ± 1.80 | 3079.51 |
| ViMedical Disease | S³ axial | 28.80 ± 0.36 | 1803.35 ± 0.36 | 1819.24 |
| ViMedical Disease | S³ angular | 28.20 ± 0.33 | 1802.75 ± 0.33 | 1818.64 |
| ViMedical Disease | S³ combined | 28.22 ± 0.37 | 1802.76 ± 0.37 | 1818.66 |
| ViMedical Disease | LDA | 21.39 ± 0.23 | 21.54 ± 0.23 | 24.98 |
| ViMedical Disease | NMF | 3.05 ± 0.02 | 3.20 ± 0.03 | 6.65 |
| ViMedical Disease | BERTopic + UMAP + KMeans | 11.62 ± 0.07 | 1786.17 ± 0.07 | 1802.06 |
| VNTC-CNTT | S³ axial | 89.86 ± 0.65 | 4044.89 ± 0.65 | 4060.79 |
| VNTC-CNTT | S³ angular | 89.83 ± 0.78 | 4044.86 ± 0.78 | 4060.75 |
| VNTC-CNTT | S³ combined | 90.15 ± 0.49 | 4045.19 ± 0.49 | 4061.08 |
| VNTC-CNTT | LDA | 27.84 ± 0.34 | 28.54 ± 0.36 | 31.98 |
| VNTC-CNTT | NMF | 4.04 ± 0.03 | 4.74 ± 0.04 | 8.18 |
| VNTC-CNTT | BERTopic + UMAP + KMeans | 26.01 ± 0.07 | 3981.04 ± 0.07 | 3996.94 |

| Cột timing | Nội dung đo | Cách dùng đúng trong luận văn |
|---|---|---|
| `ingest_preprocess_seconds_shared` | Đọc corpus và tiền xử lý đã khóa, ghi theo corpus | Chi phí chuẩn bị dữ liệu; không phải fit topic |
| `encoder_model_load_seconds_shared` | Nạp CafeBERT cục bộ; bằng 0 cho LDA/NMF | Chi phí khởi tạo encoder; không gồm tải mạng |
| `representation_seconds_cold_reference` | CafeBERT mean encoding cho S³/BERTopic hoặc CountVectorizer cho LDA/NMF | Chi phí tạo biểu diễn lạnh theo lớp mô hình |
| `fit_seconds` | Fit mô hình và trích top-term sau khi biểu diễn sẵn sàng | So sánh phân tích/trích topic với warm cache |
| `pipeline_seconds` | Representation cold-reference + fit | Pipeline sau khi corpus/model đã sẵn sàng; không gồm metric |
| `total_cold_seconds` | Ingest + nạp model + pipeline | Cold-start trong đúng môi trường đã ghi |
| `metric_seconds` | WEC-in, diversity, C_NPMI và alphabetic audit | Chi phí đánh giá, không cộng vào fit/pipeline |

Phụ lục `latex_timing/` được sinh trực tiếp từ `full_results.csv`: có bảng LaTeX 24 hàng theo *corpus × mô hình*, bảng compact theo từng corpus, CSV summary và kiểm tra hai đẳng thức `pipeline = representation + fit` cùng `total cold = ingest + model load + pipeline`. Mỗi ô trước hết trung bình qua năm giá trị k trong một seed, sau đó báo mean ± sample SD qua bốn seed. Không cộng các hàng này để suy ra wall-clock của toàn bộ benchmark.

![Fit-only trung bình ± SD](cafebert-fit-timing.png)

## 7. Diễn giải và giới hạn có thể dùng trong luận văn

Trong protocol này, kết quả 480 phép chạy hỗ trợ phát biểu hẹp: **S³ có thể vận hành với CafeBERT trên bốn corpus tiếng Việt và đạt WEC-in cao nhất ở 59/80 ô corpus × seed × k đã đánh giá.** Đây là coherence dựa trên embedding nội bộ corpus, không phải đánh giá “đúng nhãn” của chủ đề. Vì vậy không nên suy diễn kết quả thành độ chính xác sentiment của UIT-ViSFD, chất lượng chẩn đoán y khoa của ViMedical hay hiệu quả tìm kiếm ngoài dữ liệu.

WEC-in, diversity và C_NPMI đo những thuộc tính khác nhau. C_NPMI có thể cho thứ hạng khác WEC-in do dựa vào đồng xuất hiện term; báo cáo giữ cả ba chỉ số thay vì chọn riêng metric có lợi. Không dùng `aggregate_proxy` ở bất kỳ bảng hay kết luận nào. WEC-ex vẫn N/A, bởi phép đo Google News Word2Vec không có cùng ngôn ngữ và corpus đích.

Kết quả thời gian chỉ so sánh được trong môi trường đã ghi tại `environment.json`, với đúng phiên bản thư viện, dữ liệu và cấu hình này. Khi muốn đưa claim thời gian vào luận văn, cần nêu rõ **fit-only warm** hoặc **pipeline cold-reference**, đồng thời nhấn mạnh LDA/NMF không dùng CafeBERT. Thí nghiệm không có gold-standard topic labels và chỉ giới hạn ở bốn corpus; đây là hai giới hạn cần giữ trong phần thảo luận.

## 8. Tái lập

Các artifact tham chiếu đã kiểm toán nằm trong `benchmark/cafebert_full/reference/`; một lần chạy mới ghi vào `benchmark/cafebert_full/results/` mặc định. Các file gồm `full_results.csv`, `full_topics.json`, `run_config.json`, corpus manifest, `environment.json`, `experiment_contract.json`, báo cáo audit, bảng summary và biểu đồ. Từ root của clone Git, lệnh kiểm toán và tái sinh báo cáo là:

```bash
export S3_CAFEBERT_RESULTS_DIR="$PWD/benchmark/cafebert_full/reference"
python -m benchmark.cafebert_full.audit_cafebert_full
python -m benchmark.cafebert_full.generate_cafebert_full_report
python -m benchmark.cafebert_full.generate_cafebert_timing_appendix
```

## Tài liệu tham khảo

[1]: https://aclanthology.org/2025.acl-long.32/ "Kardos et al. (2025), Semantic Signal Separation"
[2]: https://github.com/x-tabdeveloping/turftopic "Turftopic repository"
[3]: https://huggingface.co/uitnlp/CafeBERT "uitnlp/CafeBERT model card"
[4]: https://maartengr.github.io/BERTopic/ "BERTopic documentation"
