<!--
Mục 2 — Related Work / Background (~1.0-1.25 trang khi vào khuôn ACL 2 cột)
Nguồn: main/content.md Act 1 (S1-S4) + Act 2 (S5-S9) + act2_from_pptx.md (S1-S9).
Trích dẫn dùng cú pháp \citep{key}/\citet{key} sẵn cho LaTeX -- key khớp với
report/refs.bib (đã tạo cùng file này). Khi có template ACL, copy thẳng phần
"### Nội dung" vào \section{Related Work} của .tex, không cần viết lại.
-->

# 2. Công trình liên quan

### Nội dung

Mô hình hoá chủ đề (topic modeling) là nhóm phương pháp thống kê không giám
sát nhằm khám phá các chủ đề ẩn trong một kho văn bản lớn mà không cần đọc
thủ công từng tài liệu, mỗi chủ đề thường được biểu diễn bằng một tập từ
khoá đại diện. Các phương pháp cổ điển như Latent Semantic Analysis (LSA)
\citep{deerwester1988lsa} và Latent Dirichlet Allocation (LDA)
\citep{blei2003lda} đều xây dựng trên biểu diễn Bag-of-Words (BoW) -- đếm
tần suất xuất hiện của từ, bỏ qua ngữ pháp và thứ tự từ. LSA nén ma trận
đếm từ xuống không gian có số chiều thấp hơn bằng phân tích giá trị kỳ dị
(SVD), trong khi LDA là một mô hình sinh xác suất, giả định mỗi tài liệu là
một phân phối Dirichlet trên các chủ đề rồi suy luận ngược bằng thống kê
Bayes. Cách tiếp cận dựa trên BoW gặp ba hạn chế cố hữu: (i) nhạy cảm với
các từ dừng (stop words) có tần suất cao nhưng không mang nội dung, khiến
chủ đề sinh ra thường lẫn nhiều từ vô nghĩa; (ii) đòi hỏi một khâu tiền xử
lý (loại bỏ từ dừng, chuẩn hoá từ...) không có chuẩn thống nhất, làm giảm
khả năng tái lập giữa các nghiên cứu; (iii) biểu diễn BoW rất thưa và có số
chiều cao, gây tốn kém tính toán và không nắm bắt được quan hệ ngữ nghĩa
giữa các từ đồng nghĩa hoặc đa nghĩa.

Sự xuất hiện của các mô hình biểu diễn ngữ cảnh dựa trên mạng nơ-ron sâu,
tiêu biểu là BERT \citep{devlin2019bert} và các biến thể tối ưu cho câu như
Sentence-BERT \citep{reimers2019sbert}, mở ra hướng đi mới: biểu diễn tài
liệu bằng vector đặc (dense embedding) mang thông tin ngữ cảnh, bền vững
hơn trước lỗi chính tả, và tận dụng được học chuyển giao (transfer learning)
từ mô hình đã huấn luyện trước trên kho ngữ liệu khổng lồ. Tuy nhiên, bản
thân embedding chỉ là một biểu diễn -- chưa phải là một mô hình chủ đề --
nên các phương pháp xây dựng trên nền tảng này chia thành hai nhánh chính.

**Nhánh mạng nơ-ron (neural).** Contextualized Topic Models (CTM)
\citep{bianchi2021ctm} dùng một bộ tự mã hoá biến phân (VAE) nhận đầu vào là
embedding ngữ cảnh (có thể kết hợp thêm BoW) để tái tạo lại phân phối từ.
ECRTM \citep{wu2023ecrtm} bổ sung một cơ chế điều chuẩn phân cụm embedding
(embedding clustering regularization) để buộc các chủ đề tách biệt nhau rõ
hơn, còn FASTopic \citep{wu2024fastopic} mô hình hoá quan hệ tài liệu - chủ
đề - từ như một bài toán vận chuyển tối ưu (optimal transport). Điểm chung
của các mô hình này là vẫn cần một bộ giải mã hoặc hàm mất mát định nghĩa
trên không gian BoW, nên preprocessing nặng vẫn là một khâu khó tránh; đồng
thời quá trình huấn luyện lặp (nhiều epoch, tối ưu hoá hai giai đoạn) khiến
tốc độ chạy chậm và kết quả có thể không ổn định giữa các lần chạy.

**Nhánh phân cụm (clustering).** Top2Vec \citep{angelov2020top2vec} và
BERTopic \citep{grootendorst2022bertopic} đi theo hướng hình học: dùng UMAP
\citep{mcinnes2018umap} để giảm chiều embedding xuống một không gian vài
chục chiều, sau đó dùng HDBSCAN \citep{campello2013hdbscan} để phân cụm mật
độ (density-based clustering) mà không cần định trước số cụm. Chủ đề của
mỗi cụm được suy ra hậu kỳ: Top2Vec dùng độ tương đồng cosine giữa embedding
từ và tâm cụm, còn BERTopic dùng một biến thể TF-IDF theo lớp (class-based
TF-IDF) -- tức là vẫn quay lại đếm từ ở bước cuối cùng. Pipeline hai giai
đoạn này (UMAP rồi HDBSCAN) có tổng cộng 5-6 siêu tham số không có hướng dẫn
lựa chọn rõ ràng, số cụm sinh ra không kiểm soát được (có thể cần thêm bước
gộp cụm), và một tài liệu bị phân sai cụm ở bước UMAP sẽ không thể sửa được
ở các bước sau.

**Ba thách thức chung.** Nhìn xuyên suốt cả hai nhánh, các mô hình chủ đề
ngữ cảnh hiện có đối mặt với ba vấn đề: (1) nhạy cảm với lựa chọn siêu tham
số, khiến kết quả khó tái lập; (2) vẫn phụ thuộc vào một hình thức đếm từ
nào đó (BoW trong decoder của CTM, hay TF-IDF trong BERTopic), nên chưa
thực sự thoát khỏi hạn chế của cách tiếp cận cổ điển; (3) chưa có bằng
chứng rõ ràng cho thấy các mô hình này tận dụng được thông tin ngữ cảnh
thật sự, vì phần lớn được đánh giá trên dữ liệu đã qua tiền xử lý kỹ lưỡng.
Ba thách thức này chính là động lực trực tiếp dẫn tới Semantic Signal
Separation (S³) \citep{kardos2025s3} -- phương pháp mà nhóm tái lập và mở
rộng cho tiếng Việt trong báo cáo này, được trình bày ở Mục 3.

---
### Ghi chú cho người biên tập (xoá khi ráp vào .tex)
- Độ dài hiện tại ~620 từ tiếng Việt -- ước tính chiếm ~1.0-1.1 trang 2 cột
  ACL ở 11pt, khớp ngân sách đề ra. Có thể cắt đoạn "Nhánh phân cụm" xuống
  1-2 câu nếu Mục 5 cần thêm chỗ.
- Chưa có câu chuyển ý mở đầu nối từ Introduction -- sẽ thêm 1 câu ngắn khi
  viết Mục 1 (Introduction) sau cùng, biết rõ Introduction kết ở đâu.
- `\citep{devlin2019bert}` -- BERT không nằm trong danh sách reference tôi
  trích được từ paper gốc (trang 10-11 chỉ có SBERT/Reimers & Gurevych). Vẫn
  đưa vào vì là kiến thức nền chuẩn, nhưng nếu thầy yêu cầu bám sát 100%
  reference list của paper gốc thì nên bỏ câu nhắc BERT, chỉ giữ SBERT.
