# Tái lập S³ với CafeBERT và dữ liệu tiếng Việt

Pipeline này tái hiện phần cốt lõi của **S³ — Semantic Signal Separation**
(ACL 2025), đồng thời thay encoder trong bài bằng `uitnlp/CafeBERT` và dùng:

- `dataset/ViSFD/ViSFD.csv`: trường `comment`, 11.122 tài liệu.
- `dataset/Vietnamese-News/all/*.parquet`: trường `text`, 2.421.826 tài liệu.

## Khác biệt so với thí nghiệm gốc

Bài báo dùng các SentenceTransformer/GloVe tiếng Anh. CafeBERT là XLM-R tiếp tục
pretrain bằng masked-language modeling, không kèm pooling sentence embedding.
Pipeline này vì vậy dùng **masked mean pooling** trên hidden state cuối rồi L2
normalize. Đây là adaptation với CafeBERT, không phải cấu hình encoder nguyên bản
của bài báo.

Metric `embedding_coherence` dùng chính CafeBERT thay cho external Google-News
Word2Vec và internal Word2Vec trong bài. Topic diversity giữ đúng định nghĩa của
bài. Kết quả vì vậy dùng để so sánh các số topic trong hai corpus tiếng Việt,
không nên so trực tiếp với các con số tiếng Anh trong paper.

## Cài đặt (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy smoke test trước

Lần chạy đầu sẽ tải model `uitnlp/CafeBERT` (khoảng 0,6B tham số).

```powershell
.\.venv\Scripts\python.exe -m s3_reproduction.cli --backend turftopic --dataset visfd --n-topics 10 --max-documents 500 --max-features 1000 --batch-size 4
```

## Chạy hai dataset

```powershell
.\.venv\Scripts\python.exe -m s3_reproduction.cli --backend turftopic --dataset all --n-topics 10 20 30 40 50 --max-documents 20000 --batch-size 8
```

`turftopic` là backend mặc định và dùng implementation S³ chính thức của tác
giả. Có thể dùng `--backend custom` để chạy bản tự triển khai và kiểm tra parity.
Hai backend được lưu riêng ở `artifacts/turftopic/<dataset>/` và
`artifacts/custom/<dataset>/` để không ghi đè nhau.

`--max-documents 0` sẽ dùng toàn bộ dữ liệu. Không khuyến nghị cho
Vietnamese-News trên một máy cá nhân vì corpus có hơn 2,4 triệu bài và CafeBERT
rất lớn. Kết quả nằm trong `artifacts/<backend>/<dataset>/topics_<N>.json`; embedding được
cache thành `.npy` để phân tích tiếp.

Mỗi JSON chứa danh sách 10 từ/topic, topic diversity và embedding coherence.
Khi lấy mẫu Vietnamese-News, thứ tự shard và dòng đều được chọn xác định bằng
seed (không cần nạp 2,4 triệu dòng vào RAM chỉ để lấy sample). Seed mặc định là
42; vocabulary được center bằng mean học từ document embeddings trước khi chiếu.
FastICA dùng `parallel`, whitening SVD và unit variance như
phần Appendix C.1 của bài báo.

## Trực quan hóa kết quả

Sau khi đã chạy đủ 10, 20, 30, 40 và 50 topics:

```powershell
.\.venv\Scripts\python.exe -m s3_reproduction.visualize
```

Biểu đồ PNG/PDF, bảng topic tốt nhất, `metrics.csv` và `summary.md` được ghi vào
`artifacts/visualizations/`. Điểm aggregate interpretability là trung bình nhân
`sqrt(coherence * diversity)`, theo cách tổng hợp metric của paper.


## Run

### turftopic
.\.venv\Scripts\python.exe -m s3_reproduction.cli --backend turftopic --dataset all --n-topics 10 20 30 40 50 --max-documents 50000 --batch-size 8

### custom 
.\.venv\Scripts\python.exe -m s3_reproduction.cli --backend custom --dataset all --n-topics 10 20 30 40 50 --max-documents 50000 --batch-size 8