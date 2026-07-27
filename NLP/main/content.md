# S³ — Kịch bản thuyết trình (đồng bộ với main.html)

> Mạch kể: **Cocktail Party (tách tín hiệu) → động lực & baseline → thuật toán S³ chi tiết (Turftopic, pipeline, 6 bước, word importance, từ negative, inference) → thí nghiệm & kết quả.**
> Nguồn: bài báo *S³ — Semantic Signal Separation*, Kardos et al., ACL 2025 (tr. 633–666). Bản chi tiết thuật ngữ: `../temp/content.md`.
> Deck: `main.html`. Ảnh: `figure3.png`, `figure7.png`, `table2.png` (xem `picture_mapping.csv`).
> Ký hiệu: **[Slide N]** tiêu đề · *Ý chính* · 🎤 câu dẫn · ⏱ thời lượng · 🖼️ hình · 💻 code.

---

## ACT 1 — TÁCH TÍN HIỆU (~8')

### [S1] Tiêu đề ⏱1'
*Tiêu đề: **S³ — Semantic Signal Separation**. Phụ đề: **Phân tích chủ đề của văn bản sử dụng thuật toán ICA**. Mục tiêu: topic modeling trực tiếp trong không gian nhúng neural, tiếp cận hình học tuyến tính.*
🎤 "Hôm nay nhóm mình báo cáo nghiên cứu S³ của ĐH Aarhus (ACL 2025). Mục tiêu cốt lõi: khám phá chủ đề (topic modeling) trực tiếp trong không gian nhúng neural bằng cách tiếp cận hình học tuyến tính. Để hiểu cơ chế, ta bắt đầu từ một bài toán nền tảng trong xử lý tín hiệu."

### [S2] Bài toán Bữa tiệc Cocktail & Thuật toán ICA ⏱3' 🖼️(sơ đồ tách nguồn)
*Nhiều nguồn phát âm thanh độc lập; micro chỉ ghi được bản trộn tuyến tính (mixed signal). Nhiệm vụ: khôi phục nguồn gốc khi không biết trước nguồn phát — Tách nguồn mù (Blind Source Separation). ICA: giả định các nguồn độc lập thống kê → tính ma trận giải trộn tối đa hoá tính độc lập → khôi phục tín hiệu gốc.*
🎤 "Cocktail Party: nhiều nguồn độc lập, micro ghi bản trộn tuyến tính, phải khôi phục nguồn gốc mà không biết trước — Blind Source Separation. ICA giả định nguồn độc lập thống kê, tính ma trận giải trộn để tối đa hoá tính độc lập, từ đó khôi phục tín hiệu gốc."

### [S3] Chuyển dịch: Văn bản là một bản trộn ngữ nghĩa ⏱2'
*Một tài liệu là bản trộn ngữ nghĩa của nhiều chủ đề ẩn. Đóng góp đầu tiên của S³: chuyển logic tách tín hiệu âm thanh sang không gian ngữ nghĩa. Thay vì clustering, coi mỗi chủ đề = một nguồn phát độc lập; áp FastICA lên document embeddings → phân rã tìm các vector cơ sở độc lập = Trục ngữ nghĩa (Semantic Axes), mỗi trục = một chủ đề.*
🎤 "Một bài viết trộn nhiều chủ đề. S³ coi mỗi chủ đề là một nguồn độc lập, dùng FastICA trên document embeddings để tách ra các Trục ngữ nghĩa."

### [S4] Bản chất toán học: ICA vs PCA ⏱2'
*PCA: hướng phương sai lớn nhất, trục không tương quan (uncorrelated) — nhưng không tương quan chưa chắc độc lập → chỉ nén, không tách nguồn. ICA: đòi độc lập thống kê cao nhất → tách nguồn thật. Nhờ đó S³ tự trích xuất trục ngữ nghĩa mà không cần định nghĩa thủ công cặp từ đối lập; chiếu từ vựng lên trục ICA → chủ đề tách biệt, diễn giải được.*
🎤 "PCA chỉ 'không tương quan' → nén; ICA đòi 'độc lập' → tách nguồn. Vậy Topic Modeling hiện hành gặp điểm nghẽn nào? Mời [người 2]."

---

## ACT 2 — ĐỘNG LỰC & BỐI CẢNH (~12')

### [S5] Topic model là gì ⏱2'
*Nhóm phương pháp thống kê, unsupervised, tự khám phá chủ đề ẩn trong kho văn bản. Mỗi topic = một tập từ khoá đại diện.*
🎤 "Có 100.000 bài báo, muốn biết chúng nói về gì mà không đọc hết — topic model trả về vài chục nhóm từ khoá đại diện."

### [S6] Cách cổ điển: Bag-of-Words & 3 giới hạn ⏱3'
*LSA/LDA dựa trên BoW (đếm từ, bỏ ngữ cảnh). 3 giới hạn: (1) nhạy stop words → topic vô nghĩa; (2) preprocessing tạo nhiều bậc tự do → khó tái lập; (3) BoW thưa, số chiều cao → tính kém.*
🎤 "Đếm từ thì 'the, is, of' luôn nhiều nhất, buộc phải làm sạch kỹ, mỗi người một kiểu."

### [S7] Embeddings mở cơ hội — nhưng vẫn vướng ⏱3'
*Embedding: có ngữ cảnh, bền lỗi chính tả, giả định Gaussian, transfer learning. Nhưng contextual model hiện tại: (1) chậm/không ổn định; (2) vẫn cần preprocessing nặng; (3) chưa rõ có thật sự dùng ngữ cảnh (vẫn test trên dữ liệu đã làm sạch). Ba vấn đề này chính là mục tiêu của S³.*

### [S8] Bối cảnh: các baseline để so sánh ⏱2' (bảng)
*3 nhóm: Classical (LDA, NMF); Neural/VAE (ZeroShotTM, CombinedTM, ECRTM, FASTopic); Clustering (Top2Vec, BERTopic = UMAP+HDBSCAN). S³ đi hướng thứ tư — phân rã ma trận trên embedding.*

### [S9] Cú lật: topic = TRỤC ⏱2'
*BERTopic/Top2Vec: topic = cụm; LDA/CTM: phân phối xác suất; S³: topic = một TRỤC ngữ nghĩa. Phân rã thành A (topic) và S (độ mạnh topic trong tài liệu); dùng ICA để các trục độc lập.*
🎤 "Thay vì hỏi 'tài liệu thuộc cụm nào', S³ hỏi 'không gian ý nghĩa có những trục độc lập nào'."

---

## ACT 3 — THUẬT TOÁN S³ (~16', phần của Người 3)

### [S10] Giới thiệu mô hình S³ — Turftopic ⏱2' 💻
*Tác giả đóng gói S³ thành thư viện Python Turftopic (interface kiểu scikit-learn).*
💻 `pip install turftopic` → `SemanticSignalSeparation(n_components=10)` → `fit_transform(documents)` → `print_topics()`.

### [S11] Pipeline S³ tổng thể ⏱2' 🖼️(sơ đồ luồng)
*Hàng trên: Tài liệu → Encoder (sentence transformers) → X (embedding) → Whitening (giảm chiều → N) → FastICA (A = trục topic, S = tỷ lệ topic). Hàng dưới: Từ điển → Encoder → V → C=A⁺, W=V·Cᵀ (chiếu từ lên trục) → Filter β → từ mô tả topic. Công thức: **X = A @ S**, **W = V @ C.T**.*

### [S12] Thuật toán S³ — 6 bước (tổng quan) ⏱1'
*1. Encode tài liệu → X · 2. Áp dụng FastICA: X = A·S · 3. Encode từ điển → V · 4. C = A⁺ · 5. W = V·Cᵀ · 6. Tính word importance bằng β.*

### [S13] Ý tưởng — quan điểm tác giả ⏱1'
*S³ quan niệm topic là các trục ngữ nghĩa giải thích biến thiên đặc thù của kho; phân rã biểu diễn thành A (topic) và S (độ quan trọng topic trong tài liệu); dùng ICA để tách bạch topic.*

### [S14] Bước 1: Biểu diễn tài liệu + Whitening ⏱2' 💻
*Mã hoá tài liệu bằng sentence transformer → ma trận X. Whitening trước FastICA (mô hình noiseless); giảm chiều bằng cách lấy N thành phần chính đầu (N = số topic).*
💻 `from sklearn.decomposition import FastICA` · `ica = FastICA(n_components=N, whiten="unit-variance")`

### [S15] Bước 2: Phân rã X bằng FastICA ⏱2' 💻
*X = A · S. A = mixing matrix; S = source matrix (document-topic importances). FastICA tìm trực tiếp A và S.*
💻 `X = enc.encode(documents)` · `S = ica.fit_transform(X)` · `A = ica.mixing_`

### [S16] Bước 3: Mã hoá từ vựng → V ⏱2' 💻
*Mã hoá từ vựng bằng cùng encoder → V. Từ vựng thường trích ngay từ dữ liệu đầu vào (không cần từ điển ngoài).*
💻 `vocab = CountVectorizer().fit(documents).get_feature_names_out()` · `V = enc.encode(vocab)`

### [S17] Bước 4: Ma trận tách C = A⁺ ⏱2' 💻
*C = pseudo-inverse (nghịch đảo giả) của A. C giữ "quy luật" các trục ngữ nghĩa, dùng 2 việc: (1) chiếu từ vựng lên C để gọi tên topic; (2) nhân văn bản mới với C để suy ra topic mà không train lại.*
💻 `A = ica.mixing_` · `C = np.linalg.pinv(A)`

### [S18] Bước 5: Chiếu từ lên trục ngữ nghĩa ⏱2' 💻
*W = V · Cᵀ. Vì C chứa vector chỉ phương của trục, V chứa vector từ → tích vô hướng = hình chiếu; W₍ⱼₜ₎ = hình chiếu của từ j lên trục topic Tₜ.*
💻 `V = enc.encode(vocab)` · `C = np.linalg.pinv(A)` · `W = V @ C.T`

### [S19] Bước 6: Word importance — hình học ⏱1' 🖼️(figure3.png)
*Figure 3: hình chiếu vₜ càng lớn (càng dài) thì điểm β_ax càng cao. vₜ = hình chiếu · ‖v‖ = độ dài vector từ · Θ = góc → sinh ra 3 công thức.*

### [S20] Bước 6: 3 cách tính word importance ⏱2'
*① Axial β=W_jt (từ nổi bật, coherence cao) · ② Angular β=W_jt/‖W_j‖ (từ đặc trưng, diversity cao) · ③ Combined β=(W_jt)³/‖W_j‖ (cân bằng, mũ lẻ giữ dấu). Khuyến nghị: Combined mặc định.*

### [S21] Từ có độ quan trọng âm (negative importance) ⏱1'
*S³ cho phép từ mang importance âm so với một topic → phát hiện từ "bài xích trục chủ đề", cho biết topic đối lập với cái gì / không phải là gì. LSA cũng có nhưng chưa từng khai thác.*
🎤 "Không mô hình nào khác cho biết một topic KHÔNG nói về cái gì — S³ thì có."

### [S22] Concept Compass — bản đồ ngữ nghĩa 🖼️(figure7.png)
*Chọn 2 trục topic làm 2 chiều, đặt mọi từ lên mặt phẳng. Trục ngang: Vật lý/Sinh học/Thị giác ↔ Ngôn ngữ; trục dọc: Deep Learning ↔ Thuật toán. S³ cho một bản đồ ngữ nghĩa — điều mô hình cụm/xác suất không làm được.*

### [S23] S³ cho ra gì? — Ví dụ trên ArXiv (Table 2) 🖼️(table2.png)
*Mỗi topic có top từ dương (nói về gì) + top từ âm (không phải gì). Topic 0 & 4 đều có "clustering" ở cột dương — chỉ cột âm mới phân biệt (0 ⟂ reinforcement/planning; 4 ⟂ cnn/deepmind).*

### [S24] Model — Inference tài liệu mới ⏱1' 💻
*Suy luận tỷ lệ topic cho tài liệu mới: Ŝ = X̂ · Cᵀ. Không train lại, không tối ưu lặp — chỉ một phép nhân ma trận.*
💻 `X_hat = encoder.encode([new_text])` · `S_hat = X_hat @ C.T`

---

## ACT 4 — THÍ NGHIỆM & KẾT QUẢ (~10')

### [S25] Thiết lập thí nghiệm ⏱2' (bảng datasets)
*6 datasets (đặc biệt 20NG raw & preprocessed để đo preprocessing), 4 embeddings (GloVe→E5), số topic 10–50, KHÔNG tune hyperparameter (tránh p-hacking), chạy 1 lần.*

### [S26] Đo chất lượng thế nào ⏱1'
*Diversity (d): topic khác nhau đến đâu. Coherence (C): topic mạch lạc đến đâu. C̄ = √(C_ex·C_in); Interpretability = √(C̄·d) — dùng geometric mean để phạt lệch, buộc tốt cả hai mặt.*

### [S27] Kết quả: các con số biết nói ⏱2'
*4.5× nhanh hơn á quân BERTopic · 27.5× nhanh hơn baselines (TB) · #1 hiệu năng tổng hợp. ECRTM/FASTopic đa dạng nhưng kém mạch lạc; Top2Vec mạch lạc nhưng kém đa dạng; S³ cân bằng tối ưu.*

### [S28] Kiểm định thống kê ⏱2' (Table 3)
*Linear regression dự đoán interpretability, mốc = S³_com. F=167.4, p<0.001, R²=0.673. Mọi baseline hệ số âm & có ý nghĩa → S³ vượt trội có ý nghĩa thống kê.*

### [S29] Phát hiện phản trực giác nhất ⏱2'
*S³ là mô hình DUY NHẤT liên tục chạy tốt hơn trên văn bản thô so với dữ liệu đã tiền xử lý → bằng chứng nó thật sự khai thác được ngữ cảnh.*

### [S30] So sánh định tính (20NG) ⏱1'
*LDA/BERTopic đầy function words; CTM/ECRTM nhiễu; Top2Vec & S³ sạch, đặc trưng.*

### [S31] Semantic Axes & Concept Compass (định tính) ⏱1'
*Topic 0 & 4 có từ dương giống nhau, chỉ từ âm phân biệt; concept compass = bản đồ ngữ nghĩa 2 trục.*

### [S32] Kết luận & Hạn chế ⏱1'
*3 trụ: chất lượng (mạch lạc + đa dạng), tốc độ (nhanh nhất), không cần preprocessing. Hạn chế: metric giả định mạnh; cài lại baseline; không tune hyperparameter; 1 seed; preprocessing thử 1 kho.*

### [S33] Cảm ơn & Q&A
*Turftopic (pip install turftopic). Câu hỏi dự phòng: "khác gì PCA/LSA?", "chọn số topic thế nào?", "tiếng Việt được không?" (được — cần sentence transformer đa ngữ).*

---

## Đối chiếu với bài báo (không lệch nội dung)
- Cocktail Party / ICA (Act 1) ↔ §2.1 Semantic Axes + §3 (Jutten & Herault 1991; FastICA).
- Động lực & baseline (Act 2) ↔ §1 Introduction + §2.2.
- Thuật toán chi tiết (Act 3) ↔ §3.1 Model (6 bước, term importance, negative, inference) + §6.2 (Table 2, Figure 7).
- Kết quả (Act 4) ↔ §4 Setup + §5 Results + §6 Qualitative + §7–8.
> Chỉ đảo/tinh chỉnh thứ tự trình bày cho dễ hiểu; không thêm/bớt kết luận so với bài gốc.
