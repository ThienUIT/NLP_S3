# S³ — Kịch bản thuyết trình (bản "hook-first")

> Thứ tự kể: **mở màn bằng Cocktail Party → cầu nối sang văn bản → ICA vs PCA khái quát → động lực & baseline → thuật toán S³ → thí nghiệm & kết quả.**
> Nguồn nội dung: bài báo gốc *S³ — Semantic Signal Separation*, Kardos et al., ACL 2025 (tr. 633–666). Bản chi tiết từng thuật ngữ: xem `../temp/content.md`.
> Deck đi kèm: `main.html` (mở trực tiếp bằng trình duyệt, không cần deploy).
> Ký hiệu: **[Slide]** tiêu đề · *Ý chính* · 🎤 câu dẫn gợi ý · ⏱ thời lượng.

---

## ACT 1 — HOOK: Tách tín hiệu (dễ, trực quan) — ~8 phút

### [S1] Tiêu đề ⏱1'
*Tiêu đề: **S³ — Semantic Signal Separation**. Phụ đề: **Phân tích chủ đề của văn bản sử dụng thuật toán ICA**. Đặt câu hỏi khơi gợi thay vì định nghĩa ngay.*
🎤 "Trước khi nói về văn bản hay AI, mình hỏi một câu tưởng chừng không liên quan: làm sao tách được từng giọng nói trong một căn phòng ồn ào?"

### [S2] Cocktail Party — bài toán tách nguồn ⏱3'
*Ẩn dụ trung tâm. 3 người nói cùng lúc → mỗi micro thu được hỗn hợp trộn lẫn → thuật toán ICA tách ngược lại thành từng giọng, dù không biết trước giọng ai (blind source separation).*
- Nhấn 3 ý: (1) nhiều nguồn **độc lập**; (2) cái ta quan sát là **bản trộn**; (3) vẫn **gỡ ngược** ra được.
🎤 "Ba người nói cùng lúc, micro chỉ thu được một mớ âm thanh chồng lên nhau. Vậy mà có một thuật toán — ICA — tách lại được từng giọng, chỉ nhờ giả định các giọng vốn độc lập với nhau. Đây gọi là 'tách nguồn mù'."

### [S3] Cầu nối: văn bản cũng là 'bản trộn' ⏱2'  ← SLIDE MỚI (vá lỗ hổng)
*Đưa người nghe từ âm thanh sang văn bản.*
- Một tài liệu hiếm khi nói đúng một chủ đề — nó **trộn nhiều chủ đề** (giống phòng tiệc trộn nhiều giọng).
- Việc **tự động tìm & tách các chủ đề** trong một kho văn bản lớn gọi là **topic modeling**.
- Ý tưởng của bài: **coi mỗi chủ đề như một "giọng nói" → dùng đúng ICA để tách chủ đề ra khỏi văn bản.** Đó là lý do tên bài: *Semantic Signal Separation*.
🎤 "Giờ đổi 'giọng nói' thành 'chủ đề', 'căn phòng' thành 'một bài viết'. Một bài viết thường trộn nhiều chủ đề. Nếu tách được chúng như tách giọng nói thì sao? Đó đúng là điều S³ làm."

### [S4] ICA khác PCA thế nào (mức khái quát) ⏱2'
*Chỉ nói ý chính, chưa vào công thức/whitening.*
- **PCA**: tìm hướng dữ liệu trải rộng nhất, các trục **không tương quan** (điều kiện *dễ*) → hay dùng để nén/giảm chiều.
- **ICA**: tìm các thành phần **độc lập** (điều kiện *khó hơn*) → tách được các nguồn thật sự tách bạch.
- Một câu chốt: *"Không tương quan" chưa chắc "độc lập"* — và chính vì đòi "độc lập" mà ICA cho các chủ đề **ít chồng lấn**.
🎤 "PCA và ICA nghe giống nhau nhưng khác nhau ở một chữ: PCA chỉ cần các trục 'không tương quan', còn ICA đòi chúng 'độc lập' — chặt hơn nhiều. Cái 'độc lập' đó là chìa khoá để các chủ đề không dính vào nhau." (Chi tiết y=x², whitening để dành Act 3.)

---

## ACT 2 — ĐỘNG LỰC & BỐI CẢNH (vì sao cần) — ~12 phút

### [S5] Topic model là gì ⏱2'
*Định nghĩa gọn + ví dụ.* Nhóm phương pháp **thống kê, unsupervised** tự khám phá chủ đề ẩn; mỗi topic = một tập từ khoá đại diện. Ứng dụng: tóm tắt "kho này nói về gì" mà không đọc thủ công.
🎤 "Có 100.000 bài báo, muốn biết chúng nói về gì mà không đọc hết — topic model trả về vài chục nhóm từ khoá đại diện."

### [S6] Cách cổ điển & 3 giới hạn của Bag-of-Words ⏱3'
*LSA/LDA dựa trên BoW (đếm từ, bỏ ngữ cảnh).* 3 giới hạn: (1) nhạy stop words → topic vô nghĩa nếu không làm sạch; (2) preprocessing tạo nhiều "bậc tự do" → khó tái lập; (3) BoW thưa, số chiều cao → tính kém.
🎤 "Đếm từ thì 'the, is, of' luôn nhiều nhất, nên buộc phải làm sạch rất kỹ — và mỗi người làm một kiểu."

### [S7] Embeddings mở cơ hội — nhưng contextual model hiện tại vẫn vướng ⏱3'
*Embedding: biểu diễn dày, có ngữ cảnh, bền lỗi chính tả, cho phép giả định Gaussian + transfer learning.* Nhưng các mô hình contextual hiện tại: (1) chậm, không ổn định; (2) vẫn cần preprocessing nặng; (3) chưa rõ có thật sự dùng ngữ cảnh (vì vẫn test trên dữ liệu đã làm sạch).
🎤 "Embedding tốt hơn đếm từ, nhưng mỉa mai là người ta vẫn test chúng trên dữ liệu đã làm sạch mất ngữ cảnh — nên không chắc chúng có dùng được ngữ cảnh không." (3 vấn đề này chính là thứ S³ giải quyết.)

### [S8] Bối cảnh: các baseline để so sánh ⏱2'
*Bảng 3 nhóm: Classical (LDA, NMF) · Neural/VAE (ZeroShotTM, CombinedTM, ECRTM, FASTopic) · Clustering (Top2Vec, BERTopic = UMAP+HDBSCAN).* S³ không thuộc nhóm nào — nó là **phân rã ma trận trên embedding**.
🎤 "Không cần thuộc bảng — chỉ cần nhớ 3 trường phái: đếm từ, mạng nơ-ron, và gom cụm. S³ đi hướng thứ tư."

### [S9] Cú lật cách nghĩ: topic = TRỤC ⏱2'
*BERTopic/Top2Vec: topic = cụm; LDA/CTM: topic = phân phối xác suất; S³: topic = một TRỤC (hướng) ngữ nghĩa.* Phân rã thành **A** (các topic) và **S** (độ mạnh topic trong mỗi tài liệu). Dùng ICA để các trục độc lập.
🎤 "Thay vì hỏi 'tài liệu này thuộc cụm nào', S³ hỏi 'không gian ý nghĩa của kho có những trục độc lập nào' — mỗi trục là một topic."

---

## ACT 3 — THUẬT TOÁN S³ (đi sâu) — ~14 phút

### [S10] 6 bước — bức tranh tổng ⏱3'
*Toàn bộ phần "khó" chỉ là một phép phân rã ma trận: tài liệu = topic × độ mạnh.*
1. Encode tài liệu → X · 2. FastICA: X = A·S · 3. Encode từ vựng → V · 4. C = A⁺ · 5. W = V·Cᵀ · 6. Word importance.
🎤 "Không có mạng nơ-ron phải train, không vòng lặp tối ưu tốn kém — chỉ là phân rã ma trận. Đó là lý do nó nhanh."

### [S11] Chi tiết bước 1–2: phân rã + whitening ⏱4'
*X = A·S bằng FastICA.* Đây là chỗ trả bài phần ICA/PCA từ Act 1:
- FastICA là mô hình **noiseless** → phải **whitening** trước (xoay + co giãn cho đám mây thành hình cầu).
- Ngay tại whitening, S³ **giảm chiều còn N** (giữ N principal components đầu, N = số topic).
- ⇒ "Bên trong S³ có một bước giống PCA (dọn dẹp), rồi ICA mới xoay trục cho độc lập." (Nhắc lại y=x²: 'không tương quan' ≠ 'độc lập'.)
🎤 "Nhớ Act 1 chứ? Giờ là lúc thấy PCA và ICA phối hợp: PCA dọn dẹp, ICA tách."

### [S12] Chi tiết bước 3–5: chiếu từ lên trục ⏱3'
*V = embedding từ vựng (cùng encoder); C = A⁺; W = V·Cᵀ.* W₍ⱼₜ₎ = vị trí từ j trên trục topic t. Inference tài liệu mới: Ŝ = X̂·Cᵀ (một phép nhân → cực nhanh).
🎤 "Cùng một encoder cho cả tài liệu lẫn từ vựng — điều kiện để chiếu chúng vào cùng không gian trục."

### [S13] Bước 6: 3 cách tính word importance ⏱3'
*Axial β=W_jt (từ nổi bật nhất, coherence cao) · Angular β=W_jt/‖W_j‖ (từ đặc trưng nhất, diversity cao) · Combined β=(W_jt)³/‖W_j‖ (cân bằng; mũ lẻ giữ dấu).* Khuyến nghị: **Combined** mặc định.
🎤 "Ba cách chỉ khác nhau ở đánh đổi mạch lạc ↔ đa dạng; Combined là lựa chọn an toàn."

### [S14] Điểm độc đáo: mô tả topic bằng từ NEGATIVE ⏱1'
*Cả 3 công thức cho phép từ có độ quan trọng âm → liệt kê 'top từ âm' = định nghĩa âm.* Hai topic có từ-dương giống nhau vẫn phân biệt được nhờ từ âm. (So sánh định lượng bỏ qua từ âm để công bằng.)
🎤 "Không mô hình nào khác cho biết một topic KHÔNG nói về cái gì — S³ thì có."

---

## ACT 4 — CHỐT HẠ: Thí nghiệm & Kết quả — ~10 phút

### [S15] Thiết lập ⏱2'
*6 datasets (đặc biệt 20NG raw & preprocessed để đo preprocessing) · 4 embeddings (GloVe→E5) · số topic 10–50 · KHÔNG tune hyperparameter (tránh p-hacking) · chạy 1 lần.* Metrics: Diversity d, Coherence C (external + internal, gộp geometric mean), Interpretability = √(C̄·d).
🎤 "Dùng cả 20 Newsgroups thô lẫn đã làm sạch để tách riêng ảnh hưởng của preprocessing — đây là thí nghiệm then chốt."

### [S16] Kết quả: các con số biết nói ⏱2'
*4.5× nhanh hơn á quân BERTopic · 27.5× nhanh hơn baselines (trung bình) · #1 hiệu năng tổng hợp.* ECRTM/FASTopic đa dạng nhưng kém mạch lạc; Top2Vec mạch lạc nhưng kém đa dạng; S³ cân bằng tối ưu.
🎤 "Các mô hình khác giỏi một mặt; S³ giỏi cả hai — và nhanh hơn hẳn."

### [S17] Kiểm định thống kê ⏱2'
*Linear regression dự đoán interpretability, mốc = S³_com. F=167.4, p<0.001, R²=0.673. Mọi baseline có hệ số âm & có ý nghĩa → S³ vượt trội có ý nghĩa thống kê.*
🎤 "Không chỉ báo cáo trung bình — họ chứng minh khác biệt CÓ Ý NGHĨA thống kê sau khi kiểm soát encoder/dataset/số topic."

### [S18] Phát hiện phản trực giác nhất ⏱2'
*S³ là mô hình DUY NHẤT liên tục chạy TỐT HƠN trên văn bản thô so với dữ liệu đã tiền xử lý* → bằng chứng nó thật sự khai thác được ngữ cảnh. Trên kho thô, S³ cao hơn TẤT CẢ.
🎤 "Mọi mô hình khác cần bạn làm sạch dữ liệu; S³ thì ngược lại — càng để nguyên, nó càng chạy tốt. Đây là câu trả lời cho câu hỏi ở đầu bài: contextual model có thật sự dùng ngữ cảnh không? Có."

### [S19] Định tính + Concept Compass ⏱2'
*So sánh top từ: LDA/BERTopic đầy function words; S³/Top2Vec sạch & đặc trưng.* Concept Compass (ArXiv): đặt 2 trục lên mặt phẳng → 'bản đồ ngữ nghĩa' đặt bất kỳ từ nào lên xem nằm đâu. Từ âm phân biệt Topic 0 vs 4 (đều 'clustering').
🎤 "S³ không chỉ liệt kê topic — nó cho một bản đồ ngữ nghĩa; thứ mà mô hình cụm/xác suất không làm được."

### [S20] Kết luận + Hạn chế ⏱1'
*3 trụ: chất lượng (mạch lạc + đa dạng) · tốc độ (nhanh nhất) · không cần preprocessing.* Hạn chế khách quan: metric có giả định mạnh; cài lại baseline; không tune hyperparameter; 1 seed; preprocessing thử trên 1 kho.
🎤 "Một phương pháp đơn giản, theory-driven, kèm thư viện Turftopic mã nguồn mở."

### [S21] Cảm ơn & Q&A
*Turftopic demo (pip install turftopic).* Câu hỏi dự phòng: "khác gì PCA/LSA?", "chọn số topic thế nào?", "áp dụng tiếng Việt được không?" (được — cần sentence transformer đa ngữ).

---

## Đối chiếu với thứ tự bài báo (để yên tâm không lệch nội dung)
- Cocktail Party/ICA (Act 1) ↔ §2.1 Semantic Axes + §3 (Jutten & Herault 1991, FastICA).
- Động lực & baseline (Act 2) ↔ §1 Introduction + §2.2.
- Thuật toán (Act 3) ↔ §3.1 Model (6 bước, term importance, negative, inference).
- Kết quả (Act 4) ↔ §4 Setup + §5 Results + §6 Qualitative + §7–8.
> Chỉ **đảo thứ tự trình bày** cho dễ hiểu; **không thêm/bớt kết luận** so với bài gốc.
