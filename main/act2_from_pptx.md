# Act 2 — trích từ S3_Background_9slides.pptx (bản mới của bạn cùng nhóm)

> Nguồn: `main/S3_Background_9slides.pptx` (9 slide, không có ảnh nhúng). Bản trích để import vào `main.html`.

## S1 — Quá trình phát triển Topic Modeling (timeline 4 giai đoạn)
- GĐ1 Bag-of-Words: LSA (1988) · LDA (2003)
- GĐ2 Contextual Embedding: BERT · SBERT (~2019)
- GĐ3 Neural / Clustering: CTM · BERTopic · Top2Vec
- GĐ4 S³ (2025): ICA trên Embedding

## S2 — Giai đoạn 1: Bag-of-Words
- Biểu diễn văn bản bằng đếm từ — bỏ ngữ pháp & thứ tự.
- Bảng document-term (mèo/đuổi/chuột/chó/sợ trên d₁,d₂,d₃).
- Điểm yếu: mất ngữ cảnh hoàn toàn · không phân biệt nghĩa (đa nghĩa) · ma trận rất thưa · bắt buộc tiền xử lý nặng.
- BoW là nền tảng; LSA & LDA đều xây trên nó, khác cách "tìm chủ đề".

## S3 — LSA (1988) & LDA (2003) — cùng nền BoW
- **LSA (đại số tuyến tính):** nén ma trận đếm từ xuống K chiều; SVD phân tách A = U·Σ·Vᵀ. Yếu: trục trực giao (uncorrelated) — 1 trục lẫn nhiều chủ đề; input vẫn BoW.
- **LDA (thống kê xác suất):** mô hình sinh — chọn phân phối chủ đề (Dirichlet) → sinh từng từ → suy luận ngược (Bayes). Yếu: giả định Dirichlet có thể sai, nhạy α,β, Bayes tốn thời gian; input vẫn BoW.
- S³ kế thừa tinh thần LSA (phân tách ma trận tuyến tính), chỉ đổi input + thuật toán. Cả hai bị giới hạn bởi BoW.

## S4 — Giai đoạn 2: Contextual Embedding
- BERT (2018), SBERT (2019). Bảng BoW vs Sentence Embedding (kích thước, thưa/dày, ngữ cảnh, ngữ pháp, tiền xử lý).
- Transfer learning: embedding pretrain trên hàng tỷ câu → dùng ngay trên corpus riêng.
- Embedding chỉ là biểu diễn — chưa tìm topic. → 2 nhánh: Neural vs Clustering.

## S5 — Giai đoạn 3: Neural vs Clustering
- **Neural** ("học" bằng mạng nơ-ron): CTM (VAE encoder→decoder BoW; topic lỗi "145, ax, 0d..."); ECRTM (VAE+Sinkhorn 1000 lặp, chậm nhất, không contextual; "verbeek, billington..."); FASTopic (optimal transport, gộp nhầm topic, kém khi embedding lớn; "moon, bike, car, orbit...").
- **Clustering** (hình học): UMAP → HDBSCAN → từ khóa; Top2Vec cosine với tâm cụm; BERTopic c-TF-IDF (quay lại đếm từ). Thất bại: BERTopic 1 cụm toàn stopword (stopword tới 100%); Top2Vec chỉ 2 cụm cho cả corpus.
- Điểm yếu chung: ① chậm (50–200 epoch / UMAP+HDBSCAN) · ② nhạy hyperparameter · ③ nhiều model vẫn quay lại đếm từ.

## S6 — Clustering: Pipeline & Điểm yếu hệ thống
- Pipeline: 1. UMAP (768→~5 chiều, giữ gần/xa) → 2. HDBSCAN (tự tìm vùng mật độ = cụm) → 3. Từ khóa (Top2Vec cosine; BERTopic c-TF-IDF).
- Điểm yếu hệ thống: 1. quá nhiều tham số (UMAP 3 + HDBSCAN 2 = 5–6, không hướng dẫn); 2. số topic nổ không kiểm soát → cần bước gộp; 3. phân cụm sai → lỗi chảy xuôi, không cứu được.

## S7 — 3 Thách thức chung (Neural + Clustering)
1. Nhạy hyperparameter, khó tái lập (Neural: lr/epochs/kiến trúc; Clustering: UMAP+HDBSCAN+topic reduction).
2. Vẫn phụ thuộc tiền xử lý (CTM decoder BoW; BERTopic c-TF-IDF).
3. Chưa rõ tận dụng context thật (đánh giá trên corpus đã xử lý; chưa ai so corpus thô vs đã xử lý).

## S8 — S³ — Hậu duệ contextual của LSA
- Nền tảng: trục ngữ nghĩa đã khám phá (Musil & Mareček 2024; Yamagiwa 2023). Bảng: prior work (trục phổ quát, không topic model) vs S³ (trục đặc thù corpus, sinh từ khóa + đánh giá).
- LSA: đếm từ + SVD → trục trực giao; S³: embedding + ICA → trục độc lập.
- Vì sao ICA thay SVD: uncorrelated (yếu) vs independent (mạnh); cocktail party — 3 micro thu hỗn hợp → ICA tách từng giọng; tài liệu = hỗn hợp topic → ICA tách topic.
- Paper: "the contextual successor of Latent Semantic Analysis". Đóng góp: biến kỹ thuật trục ngữ nghĩa thành topic model đầy đủ.

## S9 — S³ giải quyết 3 thách thức
- Bảng: hyperparameter (FastICA default scikit-learn); tiền xử lý (không bước đếm từ nào — ICA & gán từ khóa thuần embedding); context (hưởng lợi nhiều nhất khi bỏ tiền xử lý — corpus thô > baseline có TXL).
- Điểm mấu chốt: S³ loại bỏ hoàn toàn mọi bước đếm từ/TF-IDF khỏi pipeline — khác biệt căn bản nhất so với mọi model trước.
