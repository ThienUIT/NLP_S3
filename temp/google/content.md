# Tài liệu Học tập: S³ — Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)
## HƯỚNG DẪN HỌC TẬP TOÀN DIỆN: PHẦN 1, 2, 3 & CÁC MÔ HÌNH LIÊN QUAN

Tài liệu này bao gồm bản dịch chi tiết, không cắt giảm kiến thức cho các mục: **1. Introduction (Mở đầu)**, **1.1 Contributions (Đóng góp)**, **2. Related Work (Nghiên cứu liên quan)**, **3. Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)** và **3.1 Model (Mô hình)** của bài báo khoa học *S³ - Semantic Signal Separation* (ACL 2025). 
Đồng thời, tài liệu cung cấp phần phân tích chuyên sâu bổ trợ về các mô hình baseline (BERTopic, Top2Vec, CTM, ECRTM, FASTopic), sự khác biệt giữa hai trường phái Neural và Clustering, chi tiết công thức toán học và hướng dẫn sử dụng thư viện Python `turftopic`.

---

## MỤC LỤC
1. **Bản Dịch Phần 1: Introduction & 1.1 Contributions**
2. **Bản Dịch Phần 2: Related Work, 2.1 Semantic Axes & 2.2 Embedding-based Topic Models**
3. **Phân Tích So Sánh: Neural Topic Models vs. Clustering Topic Models**
4. **Mổ Xẻ Công Thức & Cách Thu Thập Tham Số của Các Mô Hình Baselines**
   - 4.1 BERTopic & Công thức c-TF-IDF
   - 4.2 Top2Vec & Độ tương đồng Cosine (Cosine Similarity)
   - 4.3 CTM (Contextualized Topic Models) & Hàm lỗi ELBO của VAE
   - 4.4 ECRTM & FASTopic
5. **Thư viện Turftopic: Nguồn gốc và Cách tích hợp trong Python**
6. **Bản Dịch Phần 3 & 3.1 (Phương pháp S³)**
7. **Kiến Thức Nền Tảng: ICA, FastICA & Các Khái Niệm Mổ Xẻ Kèm Ví Dụ**
   - 7.1 Không gian Embedding Liên tục (Continuous Space) & Ví dụ
   - 7.2 Dense Embedding (Embedding dày đặc) & Ví dụ
   - 7.3 Bản chất ICA & Ẩn dụ Cocktail Party
   - 7.4 So sánh chi tiết PCA vs ICA
   - 7.5 Ma trận trộn $A$ (Mixing Matrix) là gì? & Ví dụ
   - 7.6 Phép làm trắng (Whitening) & Ví dụ trực quan
   - 7.7 Tại sao FastICA là mô hình không nhiễu (Noiseless)?
8. **Phân Tích Chi Tiết 3 Công Thức Tính Điểm Từ Vựng của S³**
9. **Cẩm Nang Đọc Ký Hiệu & Thuật Ngữ Toán Học Mở Rộng**
10. **Quy Trình Huấn Luyện & Triển Khai Thực Tế của S³**

---

## 1. BẢN DỊCH PHẦN 1: INTRODUCTION & 1.1 CONTRIBUTIONS

### 1. Introduction (Mở đầu)
"Mô hình chủ đề" (**Topic models**) là một thuật ngữ bao trùm (umbrella term) chỉ các phương pháp tiếp cận thống kê cho phép khám phá chủ đề không giám sát (**unsupervised topic discovery**) trong các kho ngữ liệu văn bản lớn (Blei, 2012). Chúng thường được áp dụng trong phân tích dữ liệu khám phá (**exploratory data analysis**) đối với dữ liệu văn bản, bởi vì chúng cho phép các nhà thực hành khai quật và cô đọng thông tin về nội dung ngữ nghĩa của một kho ngữ liệu mà không cần phải đọc kỹ từng văn bản và tốn nhiều công sức thủ công. Theo cách truyền thống, các chủ đề được trình bày cho người dùng dưới dạng một tập hợp các thuật ngữ quan trọng (từ khóa - **keywords**) nhằm cung cấp các hiểu biết sâu sắc về các cách diễn giải có thể có của chủ đề đó.

Các phương pháp tiếp cận cổ điển đối với việc mô hình hóa chủ đề, chẳng hạn như Phân tích Ngữ nghĩa Ẩn (LSI / LSA - **Latent Semantic Indexing / Latent Semantic Analysis**) (Deerwester và cộng sự, 1988; Dumais, 2004) and Phân bổ Dirichlet Ẩn (LDA - **Latent Dirichlet Allocation**) (Blei và cộng sự, 2003; Blei, 2012), đã dựa trên các biểu diễn tài liệu dạng túi từ (BoW - **bag-of-words**) dựa trên tần suất. Mặc dù các mô hình này đã được sử dụng thành công trong nhiều thập kỷ nghiên cứu xử lý ngôn ngữ tự nhiên (NLP) (Jelodar và cộng sự, 2018), tất cả chúng đều chia sẻ một số hạn chế thực tế và lý thuyết. 

Ví dụ, các mô hình BoW nhạy cảm với các từ có tính chất thống kê không điển hình (chẳng hạn như các từ chức năng - **function words/stop words**), những từ này có thể làm ô nhiễm các mô tả chủ đề dựa trên từ khóa trừ khi các đường ống tiền xử lý nặng (heavy preprocessing pipelines) được áp dụng. Các đường ống như vậy tạo ra nhiều bậc tự do cho nhà nghiên cứu (researcher degrees of freedom). Hơn nữa, tính thưa thớt (sparsity) và số chiều cao của biểu diễn BoW thường dẫn đến hiệu quả tính toán thấp hơn và độ khớp mô hình kém hơn (poorer model fit).

Với sự ra đời của các biểu diễn ngôn ngữ dạng nơ-ron dày đặc (dense neural language representations) (Vaswani và cộng sự, 2017; Le và Mikolov, 2014; Pennington và cộng sự, 2014; Mikolov và cộng sự, 2013), những cơ hội mới đã mở ra cho nghiên cứu mô hình hóa chủ đề. Embedding câu (**Sentence embeddings**) (Reimers và Gurevych, 2019) hứa hẹn rất lớn cho mô hình hóa chủ đề, vì chúng cung cấp các biểu diễn ngôn ngữ mang tính ngữ cảnh, nhạy bén với ngữ pháp (contextual, grammar-sensitive), và mạnh mẽ hơn trước các lỗi chính tả và các từ ngoài từ điển (out-of-vocabulary terms), giảm bớt nhu cầu tiền xử lý vốn loại bỏ đi các thông tin ngôn ngữ có giá trị. 

Ngoài ra, chúng tạo ra các biểu diễn dày đặc trong một không gian liên tục (dense representations in a continuous space), cho phép thực hiện các giả định về phân phối chuẩn (Gaussianity). Chúng cũng cho phép học chuyển giao (**transfer learning**) (Ruder và cộng sự, 2019) trong mô hình hóa chủ đề, tận dụng thông tin đã học được từ các kho ngữ liệu bên ngoài lớn hơn để trích xuất chủ đề.

Một số phương pháp tiếp cận do đó đã được đề xuất bằng cách sử dụng các biểu diễn nơ-ron dày đặc cho việc mô hình hóa chủ đề ngữ cảnh, và các phương pháp này đã được chứng minh là vượt trội so với các đối thủ phi ngữ cảnh của chúng (Bianchi và cộng sự, 2021a; Bianchi và cộng sự, 2021b; Grootendorst, 2022; Angelov, 2020; Wu và cộng sự, 2024b). Tuy nhiên, nhiều phương pháp tiếp cận trong số này vẫn yêu cầu tiền xử lý để đạt hiệu năng tối ưu. Đây là một hạn chế đáng kể của lĩnh vực này, vì các đường ống tiền xử lý không được chuẩn hóa và có thể loại bỏ thông tin có giá trị, điều này đặc biệt có tác động lớn đối với các văn bản ngắn (Wu và cộng sự, 2020).

### 1.1 Contributions (Đóng góp)
Chúng tôi giới thiệu **Semantic Signal Separation** (hoặc **S³**), một kỹ thuật mô hình hóa chủ đề ngữ cảnh hóa mới quan niệm việc mô hình hóa chủ đề như là sự khám phá ra các trục ngữ nghĩa ẩn (latent semantic axes) trong một kho ngữ liệu. Các trục này được phát hiện bằng cách phân rã ma trận embedding của tài liệu bằng cách sử dụng thuật toán FastICA (Hyvärinen và Oja, 2000).

Phương pháp tiếp cận được đề xuất có các đặc điểm:
- **(a)** Đơn giản về mặt khái niệm và được dẫn dắt bởi lý thuyết (theory-driven).
- **(b)** Hoạt động ngang ngửa với các phương pháp tiếp cận hiện tại về độ mạch lạc của word-embedding (word-embedding coherence) và tạo ra độ đa dạng gần như hoàn hảo (near-perfect diversity).
- **(c)** Hiệu quả hơn về mặt tính toán so với các phương pháp tiếp cận hiện có.
- **(d)** Có thể khai thác hiệu quả thông tin ngữ cảnh.

Bên cạnh việc giới thiệu một phương pháp mới, chúng tôi cung cấp một giao diện thống nhất đơn giản dựa trên thư viện scikit-learn cho cả S³ và các phương pháp tiếp cận mô hình hóa chủ đề ngữ cảnh khác trong gói thư viện Python mang tên **Turftopic**.

---

## 2. BẢN DỊCH PHẦN 2: RELATED WORK, 2.1 SEMANTIC AXES & 2.2 EMBEDDING-BASED TOPIC MODELS

### 2. Related Work (Nghiên cứu liên quan)

#### 2.1 Semantic Axes (Các trục ngữ nghĩa)
Phân tích Thành phần Độc lập (ICA - **Independent Component Analysis**) (Jutten và Herault, 1991) trước đây đã từng được áp dụng vào các không gian embedding để khám phá các trục ngữ nghĩa (Musil và Mareček, 2024; Yamagiwa và cộng sự, 2023). Tuy nhiên, các cuộc điều tra này chủ yếu định hướng vào việc tìm kiếm các chiều ngữ nghĩa phổ quát (universal dimensions of semantics) trong embedding từ và ảnh. 

Họ đã chứng minh rằng các trục được phát hiện bởi ICA có thể diễn giải được và thường trùng khớp giữa các không gian embedding và các phương thức (modalities) khác nhau. Ngược lại, nghiên cứu của chúng tôi định hướng vào việc sử dụng các trục ngữ nghĩa để khám phá các chủ đề có tính diễn giải cao trong một kho ngữ liệu cụ thể được quan tâm, chứ không phải nhằm khám phá các chiều ngữ nghĩa phổ quát. Thêm vào đó, không có mô tả chủ đề nào được tính toán hoặc đánh giá trong các nghiên cứu trước đó đó.

#### 2.2 Embedding-based Topic Models (Các mô hình chủ đề dựa trên Embedding)
Nhiều phương pháp tiếp cận mô hình hóa chủ đề sử dụng biểu diễn ngôn ngữ nơ-ron đã được đề xuất trong vài năm qua.

**Neural Topic Models** (Các mô hình chủ đề dạng nơ-ron) (Wu và cộng sự, 2024a) dựa vào mạng nơ-ron sâu để ước lượng tham số. Các mô hình chủ đề ngữ cảnh hóa (CTMs - **Contextualized Topic Models**) (Bianchi và cộng sự, 2021a) là các mô hình sinh (generative models) của biểu diễn BoW, nhưng sử dụng hệ tiên đề mạng tự mã hóa biến phân (VAE - **variational autoencoding**) để suy luận (inference) (Srivastava và Sutton, 2017). Embedding ngữ cảnh được sử dụng làm đầu vào cho mạng mã hóa (encoder) trong **ZeroShotTM**, và đôi khi được nối (concatenate) với các vector BoW trong **CombinedTM**. CTMs thông thường yêu cầu tiền xử lý nặng, và hiệu quả tính toán cũng như chất lượng khớp mô hình giảm mạnh với các từ vựng lớn hơn (Bianchi và cộng sự, 2020).

**ECRTM** (Wu và cộng sự, 2023) là một mô hình nơ-ron dựa vào phép điều hòa gom cụm embedding (embedding clustering regularization) để tạo ra các chủ đề đủ khác biệt và ngăn các mô tả chủ đề hội tụ về phía nhau. Tuy nhiên, phương pháp tiếp cận này không sử dụng các biểu diễn ngữ cảnh và chậm hơn đáng kể so với hầu hết các mô hình chủ đề khác (Wu và cộng sự, 2024b).

**FASTopic** (Wu và cộng sự, 2024b) giới thiệu một hệ tiên đề quan hệ ngữ nghĩa kép (dual-semantic-relation paradigm), nơi các mối quan hệ giữa tài liệu, chủ đề và từ được khái niệm hóa dưới dạng các kế hoạch vận chuyển tối ưu (optimal transport plans). Như họ chứng minh, phương pháp tiếp cận của họ hiệu quả hơn và tạo ra các chủ đề chất lượng cao hơn so với các phương pháp tiếp cận nơ-ron trước đó.

**Clustering Topic Models** (Các mô hình chủ đề gom cụm) khám phá các chủ đề trong kho ngữ liệu bằng cách gom cụm các biểu diễn tài liệu trong không gian embedding. Trọng số độ quan trọng của từ cho một chủ đề cho trước được ước lượng *post hoc* (sau khi gom cụm).

**Top2Vec** (Angelov, 2020) ước lượng các trọng số này bằng cách tính toán độ tương đồng cosine (cosine similarity) giữa mã hóa từ và các tâm cụm (cluster centroids). Điều này giả định các cụm có hình cầu và lồi (spherical, convex), và các mô tả chủ đề có thể bị đại diện sai lệch tùy thuộc vào hình dạng của cụm.

**BERTopic** (Grootendorst, 2022) ước lượng độ quan trọng của từ đối với các cụm bằng cách sử dụng lược đồ gán trọng số tf-idf dựa trên lớp (c-TF-IDF - **class-based tf-idf**). Cả hai phương pháp tiếp cận đều sử dụng **UMAP** (McInnes và Healy, 2018) để giảm chiều dữ liệu và **HDBSCAN** (Campello và cộng sự, 2013) để gom cụm. Cả BERTopic và Top2Vec đều đi kèm với một phương pháp giảm số lượng chủ đề (topic reduction). Điều này là cần thiết, vì HDBSCAN tự động học số lượng cụm từ dữ liệu và số lượng cụm có thể tăng lên rất lớn, điều này có thể chứng minh là phi thực tế.

**Các thách thức** của các mô hình chủ đề ngữ cảnh hiện có sẵn, tuy nhiên, vẫn còn rất nhiều. Nhiều mô hình trong số đó nhạy cảm với các lựa chọn siêu tham số (hyperparameters), tạo ra các chủ đề có khả năng diễn giải đáng ngờ, và phụ thuộc vào các đường ống tiền xử lý có cấu trúc không được chuẩn hóa (Wu và cộng sự, 2024a). Thêm vào đó, hiện vẫn chưa rõ liệu các mô hình này có thực sự hiệu quả trong việc sử dụng thông tin ngữ cảnh và cú pháp hay không, vì chúng thường được đánh giá trên các kho ngữ liệu đã được tiền xử lý.

---

## 3. PHÂN TÍCH SO SÁNH: NEURAL TOPIC MODELS VS. CLUSTERING TOPIC MODELS

Trong mục 2.2 của bài báo, các tác giả đã phân loại rõ các mô hình baseline dùng embedding thành hai trường phái chính. Dưới đây là bảng phân tích so sánh chi tiết:

| Tiêu chí | Neural Topic Models (Mô hình nơ-ron) | Clustering Topic Models (Mô hình gom cụm) |
| :--- | :--- | :--- |
| **Đại diện tiêu biểu** | Contextualized Topic Models (CTM), ECRTM, FASTopic | BERTopic, Top2Vec |
| **Bản chất cốt lõi** | Xem việc sinh văn bản/chủ đề là một **mô hình xác suất**. Dùng mạng nơ-ron tối ưu hóa hàm lỗi để tìm tham số. | Xem chủ đề là các **cụm mật độ** (dense clusters) của tài liệu trong không gian hình học. |
| **Cơ chế ước lượng** | Học các phân phối xác suất ẩn của chủ đề thông qua mạng VAE (Variational Autoencoder) hoặc Kế hoạch Vận chuyển Tối ưu. | Sử dụng thuật toán gom cụm mật độ (thường là HDBSCAN) trên không gian embedding đã giảm chiều bằng UMAP. |
| **Cách tính trọng số từ** | Được học đồng thời (end-to-end) trong quá trình huấn luyện mạng nơ-ron (qua ma trận trọng số decoder). | Được tính toán **post hoc** (sau khi đã gom cụm xong tài liệu) bằng cách tính tương đồng hoặc thống kê tần suất từ theo cụm. |
| **Độ phức tạp tính toán** | **Rất cao**: Cần huấn luyện mạng nơ-ron qua nhiều epochs bằng GPU/CPU; hội tụ chậm và nhạy cảm với hyperparameters. | **Trung bình**: UMAP và HDBSCAN chạy tương đối nhanh, nhưng bước gom cụm có thể trở thành nút thắt cổ chai khi dữ liệu lớn. |
| **Ưu điểm** | Có nền tảng thống kê vững chắc; có thể sinh văn bản (generative model); biểu diễn toán học tường minh. | Rất trực quan về mặt hình học; dễ sử dụng; hoạt động hiệu quả trên nhiều loại kho văn bản mà không cần huấn luyện phức tạp. |
| **Nhược điểm** | Rất chậm; dễ bị sụp đổ chế độ (mode collapse) nơi các chủ đề trùng lặp nhau; nhạy cảm với stop words. | Phụ thuộc vào hình dạng cụm (Top2Vec đòi cụm hình cầu lồi); số lượng cụm do HDBSCAN sinh ra có thể quá lớn, khó kiểm soát. |

---

## 4. MỔ XẺ CÔNG THỨC & CÁCH THU THẬP THAM SỐ CỦA CÁC MÔ HÌNH BASELINES

### 4.1 BERTopic & Công thức c-TF-IDF (Class-based TF-IDF)
Trong BERTopic, các tài liệu được gom cụm thành các lớp (classes), mỗi lớp đại diện cho một chủ đề (topic). Để tìm các từ khóa đại diện cho mỗi chủ đề, tác giả coi tất cả các tài liệu trong một cụm là một "siêu tài liệu" và áp dụng công thức c-TF-IDF.

#### Công thức:
$$W_{x, c} = \text{TF}_{x, c} \cdot \log\left(1 + \frac{A}{f_x}\right)$$

#### Giải thích tham số:
*   **$W_{x, c}$**: Điểm số độ quan trọng của từ $x$ đối với lớp/chủ đề $c$ (điểm càng cao, từ càng đại diện tốt).
*   **$\text{TF}_{x, c}$**: Tần suất của từ $x$ xuất hiện trong lớp/chủ đề $c$ (tổng số lần từ xuất hiện trong tất cả tài liệu thuộc cụm đó).
*   **$f_x$**: Tần suất của từ $x$ trên toàn bộ các lớp (tổng số lần từ xuất hiện trong toàn bộ kho văn bản).
*   **$A$**: Số lượng từ trung bình trên mỗi lớp, được tính bằng:
    $$A = \frac{\sum_c N_c}{C_{total}}$$
    Trong đó $\sum_c N_c$ là tổng số từ của tất cả các lớp, và $C_{total}$ là tổng số lớp (số chủ đề).
*   **Hàm $\log(1 + \dots)$**: Hàm logarit cơ số tự nhiên (hoặc cơ số 10) cộng 1 nhằm mục đích làm mịn (smoothing), tránh trường hợp giá trị bên trong bằng 0 và giảm tốc độ tăng trưởng của trọng số khi giá trị quá lớn.

#### Cách collect dữ liệu cho các tham số:
1.  **Bước gom cụm:** Bạn cần chạy thuật toán gom cụm (như HDBSCAN) trên embedding của tài liệu để gán mỗi tài liệu vào một nhãn lớp $c$.
2.  **Đếm tần suất:** Dùng một bộ đếm từ (như `CountVectorizer`) để thống kê tần suất từ $x$ cho từng tài liệu, sau đó nhóm theo nhãn lớp $c$ để tính $\text{TF}_{x, c}$ và tổng hợp lại để có $f_x$.

---

### 4.2 Top2Vec & Độ tương đồng Cosine (Cosine Similarity)
Top2Vec không sử dụng c-TF-IDF mà dùng chung không gian vector cho cả tài liệu, từ vựng và chủ đề. Sau khi gom cụm tài liệu, tâm (centroid) của các tài liệu trong cụm được coi là Vector Chủ đề (Topic Vector). Độ quan trọng của từ được tính bằng khoảng cách hình học từ vector từ đến vector chủ đề này.

#### Công thức:
$$\text{Similarity}(v_j, c_t) = \cos(\Theta) = \frac{v_j \cdot c_t}{\lVert v_j \rVert \cdot \lVert c_t \rVert}$$

#### Giải thích tham số:
*   **$v_j$**: Vector embedding của từ $j$ trong từ vựng (kích thước $d \times 1$).
*   **$c_t$**: Vector tâm cụm (centroid) của chủ đề $t$ (kích thước $d \times 1$), được tính bằng trung bình cộng các vector embedding của các tài liệu thuộc cụm $t$.
*   **$v_j \cdot c_t$**: Phép nhân vô hướng (dot product) của hai vector:
    $$v_j \cdot c_t = \sum_{k=1}^d v_{j,k} \cdot c_{t,k}$$
*   **$\lVert v_j \rVert$ và $\lVert c_t \rVert$**: Chuẩn $L_2$ (độ dài Euclidean) của từng vector.
*   **$\cos(\Theta)$**: Giá trị Cosine của góc giữa hai vector. Giá trị nằm trong khoảng $[-1, 1]$. Càng gần 1, từ càng đồng hướng và càng quan trọng với chủ đề.

#### Cách collect dữ liệu cho các tham số:
1.  **Vector từ $v_j$:** Dùng một thuật toán học vector từ (như Word2Vec, Doc2Vec) để học đồng thời biểu diễn của từ và tài liệu trong cùng một không gian.
2.  **Vector chủ đề $c_t$:** Tính trung bình cộng của toàn bộ vector tài liệu nằm trong cụm $t$.

---

### 4.3 CTM (Contextualized Topic Models) & Hàm lỗi ELBO của VAE
CTM dựa trên mạng tự mã hóa biến phân (Variational Autoencoder - VAE). Nó nhận vào embedding ngữ cảnh của tài liệu để làm điều kiện mã hóa, nhưng mục tiêu là tái cấu trúc lại vector túi từ (BoW) của tài liệu đó. Quá trình huấn luyện tối ưu hóa hàm lỗi ELBO (Evidence Lower Bound).

#### Công thức hàm tối ưu (ELBO Loss):
$$\mathcal{L}_{ELBO}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))$$

#### Giải thích tham số:
*   **$x$**: Vector biểu diễn túi từ (Bag-of-Words) của tài liệu đầu vào (kích thước $V_{vocab} \times 1$).
*   **$z$**: Vector biến ẩn biểu thị tỷ lệ chủ đề trong tài liệu (kích thước $N \times 1$).
*   **$q_\phi(z|x)$**: Mạng mã hóa (Encoder) với các tham số mạng là $\phi$. Nhận vào embedding ngữ cảnh của tài liệu và ước lượng phân phối xác suất của biến ẩn $z$.
*   **$p_\theta(x|z)$**: Mạng giải mã (Decoder) với các tham số mạng là $\theta$. Nhận vào vector chủ đề $z$ và cố gắng tái cấu trúc lại vector BoW gốc $x$.
*   **$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$**: Kỳ vọng toán học của log xác suất tái cấu trúc. Tham số này đo lường khả năng tái tạo dữ liệu đầu vào của mô hình (decoder hoạt động tốt đến đâu).
*   **$D_{KL}(q_\phi(z|x) \parallel p(z))$**: Khoảng cách Kullback-Leibler (KL Divergence) giữa phân phối do encoder dự đoán và phân phối tiên nghiệm $p(z)$ (thường giả định là Logistic-Normal). Đóng vai trò là số hạng điều hòa (regularizer) giúp làm mượt không gian ẩn $z$.

#### Cách collect dữ liệu cho các tham số:
1.  **Dữ liệu đầu vào $x$ (BoW):** Bạn cần xây dựng ma trận đếm tần suất từ của kho văn bản thô.
2.  **Embedding ngữ cảnh (Contextual Embeddings):** Chạy mô hình Sentence Transformer trên các tài liệu để tạo embedding ngữ cảnh làm input cho Encoder $q_\phi$.

---

### 4.4 ECRTM & FASTopic
*   **ECRTM (Embedding Clustering Regularization Topic Model):**
    *   **Bản chất:** Đưa ra một số hạng điều hòa gom cụm (regularization) vào hàm tối ưu của Neural Topic Model. Mục tiêu là buộc các vector biểu diễn chủ đề phải nằm xa nhau trong không gian vector, tránh việc mô hình sinh ra các chủ đề có từ khóa giống hệt nhau (sụp đổ chế độ).
    *   **Hạn chế:** Bản gốc của ECRTM không sử dụng embedding ngữ cảnh trực tiếp làm đầu vào dẫn dắt ngữ nghĩa, mà chỉ dùng để điều hòa phân cụm tĩnh, khiến tốc độ chạy rất chậm.
*   **FASTopic (Fast Arbitrary-Semantic-Space Topic Model):**
    *   **Bản chất:** Mô hình hóa mối quan hệ giữa tài liệu - chủ đề và chủ đề - từ như một bài toán Vận chuyển tối ưu (Optimal Transport - OT). Nó sử dụng khoảng cách Wasserstein để căn chỉnh phân phối của tài liệu và từ vựng trong không gian ngữ nghĩa mà không cần cấu trúc VAE phức tạp, mang lại tốc độ chạy vượt trội và chất lượng chủ đề cao hơn CTM.

---

## 5. THƯ VIỆN TURFTOPIC: NGUỒN GỐC VÀ CÁCH TÍCH HỢP TRONG PYTHON

### 5.1 Nguồn gốc thư viện
*   **Không phải thư viện có sẵn mặc định** trong Python (như `math`, `os` hay `json`).
*   **Do chính nhóm tác giả bài báo S³ phát triển** và phát hành dưới dạng mã nguồn mở (giấy phép MIT).
*   Thư viện được thiết kế theo chuẩn giao diện của **scikit-learn** (sử dụng các hàm quen thuộc như `.fit()`, `.transform()`, `.fit_transform()`).
*   **Mục đích:** Gói gọn thuật toán S³ đề xuất cùng với tất cả các mô hình baseline ngữ cảnh phổ biến (BERTopic, Top2Vec, CTM, v.v.) vào một API duy nhất để các nhà nghiên cứu dễ so sánh đối chiếu.

### 5.2 Cách cài đặt và sử dụng
Bạn có thể dễ dàng cài đặt thư viện này thông qua trình quản lý gói `pip`:
```bash
pip install turftopic
```

#### Ví dụ mã nguồn chạy S³ qua thư viện Turftopic:
```python
from turftopic import KeyNMF, SemanticSignalSeparation
from datasets import load_dataset

# 1. Chuẩn bị dữ liệu tài liệu
dataset = ["Học máy là một ngành của trí tuệ nhân tạo.",
           "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng người.",
           "Mô hình chủ đề phân tích các bài viết tự động."]

# 2. Khởi tạo mô hình S3 với số lượng chủ đề mong muốn (ví dụ N = 3)
model = SemanticSignalSeparation(n_components=3, feature_importance="combined")

# 3. Huấn luyện mô hình và trích xuất chủ đề
model.fit(dataset)

# 4. In ra các từ khóa hàng đầu của từng chủ đề
model.print_topics(top_n=5)
```

---

## 6. BẢN DỊCH PHẦN 3 & 3.1 (PHƯƠNG PHÁP S³)

### 3. Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)
Trong bài báo này, chúng tôi giới thiệu **Semantic Signal Separation** (hoặc **S³**), một phương pháp tiếp cận mới cho bài toán mô hình hoá chủ đề (**topic modeling**) trong không gian embedding liên tục (**continuous embedding spaces**), nhằm vượt qua các thách thức đã đề cập phía trên của các phương pháp topic modeling ngữ cảnh (**contextual topic modeling**) hiện tại.

Thay vì diễn giải các chủ đề dưới dạng các cụm (**clusters**) hoặc phân phối xác suất của từ (**word probabilities**), chúng tôi quan niệm các chủ đề là các **trục ngữ nghĩa (semantic axes)** giải thích sự biến thiên (**variation**) đặc thù của một kho ngữ liệu (**corpus**). Điều này đạt được bằng cách phân rã (**decomposing**) các biểu diễn ngữ nghĩa thành:
- Các **thành phần ẩn (latent components) $A$**: được giả định chính là các chủ đề (topics).
- **Độ mạnh của các thành phần trong từng tài liệu $S$**: chính là mức độ quan trọng giữa tài liệu và chủ đề (document-topic importances).

Để các chủ đề có tính chất tách biệt rõ ràng về mặt khái niệm (**conceptually distinct**), chúng tôi sử dụng phương pháp Phân tích Thành phần Độc lập (**Independent Component Analysis - ICA**) (Jutten và Herault, 1991) để tìm ra chúng. Độ quan trọng của từ đối với chủ đề (**term importances**) được ước lượng từ độ mạnh của các thành phần chủ đề trong embedding của từ (**word embeddings $V$**).

Xét trên một số khía cạnh, phương pháp của chúng tôi có thể được coi là hậu duệ ngữ cảnh (**contextual successor**) của phương pháp Phân tích Ngữ nghĩa Ẩn (**Latent Semantic Analysis - LSA**) (Dumais, 2004; Deerwester và cộng sự, 1988) - phương pháp vốn khám phá ra các nhân tố (factors) dựa trên sự đồng xuất hiện của từ (word-occurrences).

### 3.1 Model (Mô hình)
Các biểu diễn tài liệu (**document representations**) thu được bằng cách mã hoá các tài liệu bằng một mô hình **Sentence Transformer**.

1. Gọi $X$ là ma trận mã hoá của các tài liệu (**document encodings matrix**).
   
Việc phân rã các biểu diễn tài liệu thành các trục ngữ nghĩa độc lập được thực hiện bằng phương pháp Phân tích Thành phần Độc lập (**Independent Component Analysis - ICA**) (Jutten và Herault, 1991). Trong nghiên cứu này, chúng tôi sử dụng thuật toán **FastICA** (Hyvärinen và Oja, 2000) để xác định các thành phần ngữ nghĩa ẩn. 

Là một bước tiền xử lý, phép làm trắng (**whitening**) được áp dụng lên ma trận embedding, vì FastICA là một mô hình không nhiễu (**noiseless model**). Do mặc định ICA sẽ tìm ra số lượng thành phần bằng đúng số chiều của embedding, chúng tôi giảm chiều dữ liệu của embedding trong quá trình làm trắng bằng cách chỉ lấy $N$ thành phần chính đầu tiên (**principal components**), với $N$ là số lượng chủ đề (topics) mong muốn.

2. Phân rã $X$ sử dụng FastICA:
   $$X = A \cdot S$$
   Trong đó, $A$ là **ma trận trộn (mixing matrix)**, và $S$ là **ma trận nguồn (source matrix)** chứa độ quan trọng giữa tài liệu và chủ đề (document-topic importances).

Độ quan trọng của từ đối với chủ đề (**term importances**), dùng để lựa chọn các từ ngữ mô tả chủ đề, được tính toán bằng cách chiếu (**projecting**) các từ lên các trục ngữ nghĩa đã tìm được.

3. Mã hoá từ vựng của kho ngữ liệu bằng cùng một mô hình encoder đó. Gọi ma trận mã hoá từ vựng là $V$.
4. Gọi ma trận tách (hay ma trận mở trộn - **unmixing matrix**) là $C$, được tính bằng nghịch đảo giả (**pseudo-inverse**) của ma trận trộn $A$:
   $$C = A^+$$
5. Chiếu các từ lên các trục ngữ nghĩa đã phát hiện bằng cách nhân ma trận embedding của từ với ma trận unmixing:
   $$W = V \cdot C^T$$
6. Tính toán điểm quan trọng của từ (word importance scores) cho từng chủ đề.

Chúng tôi xem xét ba phương pháp để tính toán độ quan trọng của từ:
1. **Độ quan trọng của từ theo Trục (Axial word importances)**: được định nghĩa là vị trí của từ trên các trục ngữ nghĩa. Độ quan trọng của từ $j$ đối với chủ đề $t$ là:
   $$\beta_{tj} = W_{jt}$$
2. **Chủ đề theo Góc (Angular topics)**: có thể được tính toán bằng cách lấy cosine của góc giữa vector từ đã chiếu và các trục ngữ nghĩa:
   $$\beta_{tj} = \cos(\Theta) = \frac{W_{jt}}{\lVert W_j \rVert}$$
3. **Độ quan trọng kết hợp (Combined word importance)**: là sự kết hợp của cả hai cách tiếp cận trên:
   $$\beta_{tj} = \frac{(W_{jt})^3}{\lVert W_j \rVert}$$
   *(Chúng tôi lấy luỹ thừa bậc lẻ của vị trí từ để giữ nguyên dấu của nó).*

Độ quan trọng theo trục (**axial word importance**) giúp tạo ra các mô tả chủ đề chứa các từ **nổi bật nhất (most salient)** của chủ đề đó, trong khi độ quan trọng theo góc (**angular importance**) sẽ gán trọng số cao nhất cho những từ **đặc trưng riêng biệt nhất (most specific)**. Độ quan trọng kết hợp (**combined importance**) hướng tới việc cân bằng cả hai khía cạnh này.

Lưu ý rằng tất cả các công thức trên đều cho phép các từ có **độ quan trọng âm (negative importance)** đối với một chủ đề nhất định. Mặc dù điều này cũng xuất hiện trong phương pháp LSA, các nghiên cứu trước đây chưa từng khai thác khái niệm này. Việc diễn giải mô hình có thể được mở rộng bằng cách kiểm tra các từ có điểm số thấp nhất trên một chủ đề cho trước, cung cấp một **định nghĩa phủ định (negative definition)** cho chủ đề đó.

Để đảm bảo tính so sánh tương đương với các phương pháp không cho phép định nghĩa phủ định, các thử nghiệm đối sánh mô hình của chúng tôi sẽ bỏ qua các từ có giá trị âm, tuy nhiên một ví dụ minh hoạ thực tế sẽ được trình bày ở Mục 6.2.

**Suy luận (Inference)** tỷ lệ chủ đề trong các tài liệu mới (chưa từng thấy trước đây) có thể đạt được bằng cách nhân embedding của tài liệu mới đó với ma trận unmixing $C$:
1. Gọi mã hoá của các tài liệu mới chưa từng thấy là $\hat{X}$.
2. Tính toán ma trận tài liệu - chủ đề mới:
   $$\hat{S} = \hat{X} \cdot C^T$$

---

## 7. KIẾN THỨC NỀN TẢNG: ICA, FASTICA & CÁC KHÁI NIỆM MỔ XẺ KÈM VÍ DỤ

### 7.1 Không gian Embedding Liên tục (Continuous Space) & Ví dụ
*   **Bản chất:** Trong biểu diễn từ điển rời rạc (cũ), mỗi từ là một chỉ mục (index) cô lập, ví dụ: $mèo = 1, chó = 2, bàn = 3$. Không có từ nào có mã số $1.5$ (nằm giữa chó và mèo) và ta không thể thực hiện các phép tính khoảng cách hay hướng. Ngược lại, **Không gian embedding liên tục** biểu diễn các từ dưới dạng các vector số thực trong một không gian hình học đa chiều. 
*   **Ví dụ trực quan:** Giả sử ta nhúng các loài động vật vào không gian 2 chiều: Chiều 1 là "Độ lớn cơ thể" (0.0 đến 1.0) và Chiều 2 là "Độ thuần hóa/thân thiện" (0.0 đến 1.0).
    *   $Chó nhà = [0.4, 0.9]$
    *   $Mèo nhà = [0.2, 0.8]$
    *   $Hổ rừng = [0.9, 0.1]$
    *   *Khoảng cách:* Khoảng cách Euclid giữa vector Chó nhà và Mèo nhà rất ngắn (biểu thị nghĩa gần nhau), trong khi Hổ rừng nằm rất xa cả hai.
    *   *Tính liên tục:* Ta có thể di chuyển mịn màng giữa vector Chó và Mèo để tìm một vector trung gian như $[0.3, 0.85]$ biểu diễn một loài thú cưng nhỏ khác (ví dụ: Hamster).

### 7.2 Dense Embedding (Embedding dày đặc) & Ví dụ
*   **Bản chất:** 
    *   **Sparse Vector (Vector thưa thớt):** Điển hình là ma trận Bag-of-Words hoặc One-Hot. Vector có số chiều bằng kích thước từ vựng (ví dụ: 50,000 chiều). Mỗi từ chỉ chứa duy nhất một số 1, còn lại 49,999 phần tử đều là số 0. Rất thưa thớt và tốn bộ nhớ.
    *   **Dense Vector (Vector dày đặc):** Có số chiều nhỏ và cố định (ví dụ: 384 hoặc 768 chiều). **Tất cả các phần tử đều là số thực khác 0** (dày đặc). Mỗi tọa độ đại diện cho một nét nghĩa ẩn được nén lại.
*   **Ví dụ trực quan:**
    *   Với từ "mèo" trong từ điển 10,000 từ:
        *   *Sparse Vector:* $[0, 0, 0, 0, 1, 0, 0, \dots, 0]$ (dài 10,000, 99.9% là số 0).
        *   *Dense Vector:* $[0.12, -0.45, 0.78, 0.05, \dots, -0.09]$ (chỉ dài 384 chiều nhưng chứa đầy đủ thông tin ngữ cảnh nén).

### 7.3 Bản chất ICA & Ẩn dụ Cocktail Party
*   *(Xem chi tiết tại Mục 2.1 của phần nội dung trước. Thuật toán S³ áp dụng bài toán Cocktail Party này bằng cách coi "giọng nói gốc" là các chủ đề ẩn, còn "bản ghi âm hỗn hợp của micro" chính là ma trận embedding của tài liệu).*

### 7.4 So sánh chi tiết PCA vs ICA
*   *(Xem chi tiết bảng so sánh tại Mục 2.2 của phần nội dung trước. PCA chỉ tìm các trục không tương quan tuyến tính để nén dữ liệu, còn ICA tìm các trục thực sự độc lập thống kê để tách nguồn).*

### 7.5 Ma trận trộn $A$ (Mixing Matrix) là gì? & Ví dụ
*   **Bản chất:** Ma trận trộn $A$ (kích thước $d \times N$) là "công thức trộn" chỉ ra cách các nguồn chủ đề độc lập ($S$) kết hợp lại theo tỷ lệ nào để tạo ra vector embedding tài liệu quan sát được ($X$). Mỗi cột của $A$ đại diện cho hướng đi của một trục chủ đề trong không gian embedding $d$ chiều.
*   **Ví dụ cụ thể:** Giả sử kho văn bản có 2 chủ đề: $T_1$ (Tech) và $T_2$ (Sports), và embedding dài 3 chiều ($d=3$).
    *   Ma trận trộn $A$ học được là:
        $$A = \begin{bmatrix} 0.8 & 0.1 \\ -0.2 & 0.9 \\ 0.5 & -0.3 \end{bmatrix}$$
        (Cột 1 đại diện cho Tech, Cột 2 đại diện cho Sports).
    *   Một tài liệu chứa 70% nội dung Tech và 30% nội dung Sports, tương ứng vector nguồn $S = \begin{bmatrix} 0.7 \\ 0.3 \end{bmatrix}$.
    *   Embedding tài liệu quan sát được $X$ sẽ được tạo ra bằng cách nhân $A \cdot S$:
        $$X = A \cdot S = \begin{bmatrix} 0.8 \cdot 0.7 + 0.1 \cdot 0.3 \\ -0.2 \cdot 0.7 + 0.9 \cdot 0.3 \\ 0.5 \cdot 0.7 + (-0.3) \cdot 0.3 \end{bmatrix} = \begin{bmatrix} 0.59 \\ 0.13 \\ 0.26 \end{bmatrix}$$

### 7.6 Phép làm trắng (Whitening) & Ví dụ trực quan
*   **Bản chất:** Làm trắng là phép biến đổi tuyến tính giúp dọn dẹp mối tương quan tuyến tính giữa các thuộc tính (decorrelate) và co giãn để phương sai các chiều đều bằng 1. Về mặt hình học, nó biến đổi một "đám mây dữ liệu hình elip lệch xiên" thành một "đám mây hình cầu tròn trịa".
*   **Ví dụ trực quan:** Giả sử ta đo Chiều cao và Cân nặng của một nhóm người.
    *   Vì chiều cao và cân nặng tương quan rất cao (người cao thường nặng), biểu đồ phân tán sẽ là một elip nghiêng chéo.
    *   Sự tương quan này làm nhiễu thuật toán FastICA vì nó che mờ các hướng độc lập thực sự.
    *   *Whitening sẽ:* (1) Trừ giá trị trung bình để đưa tâm elip về gốc tọa độ 0; (2) Xoay trục dữ liệu để triệt tiêu tương quan; (3) Co giãn các trục sao cho phương sai cả hai chiều đều bằng 1.
    *   *Kết quả:* Đám mây elip nghiêng biến thành một hình tròn hoàn hảo. Lúc này, FastICA có thể dễ dàng xoay các trục để tìm hướng độc lập tối ưu nhất.

### 7.7 Tại sao FastICA là mô hình không nhiễu (Noiseless)?
*   **Bản chất:** Trong xử lý tín hiệu, mô hình có nhiễu (Noisy ICA) được viết là $X = A \cdot S + \eta$ (với $\eta$ là ma trận sai số). Mô hình này rất khó giải và tốn thời gian tính toán vì phải ước lượng phân phối nhiễu.
*   **Lý do FastICA chọn Noiseless:** FastICA giả định dữ liệu quan sát được tạo ra hoàn toàn tuyến tính từ nguồn mà không có nhiễu: $X = A \cdot S$. 
    *   Giả định này giúp việc gỡ trộn (tính toán ma trận unmixing $C$) trở nên cực kỳ đơn giản bằng đại số tuyến tính: $S = C \cdot X = A^+ \cdot X$.
    *   Nó mang lại tốc độ chạy tức thời (chạy giải tích thay vì lặp gradient tối ưu phân phối nhiễu), nhưng đòi hỏi bắt buộc dữ liệu phải được làm trắng trước để hạn chế tối đa sai lệch của giả định.

---

## 8. PHÂN TÍCH CHI TIẾT 3 CÔNG THỨC WORD IMPORTANCE CỦA S³
*(Nội dung giữ nguyên phân tích hình học, ưu nhược điểm của Axial, Angular và Combined với giải thích về luỹ thừa bậc 3 đã thực hiện ở phần trước).*

---

## 9. CẨM NANG ĐỌC KÝ HIỆU & THUẬT NGỮ TOÁN HỌC MỞ RỘNG
*(Nội dung giữ nguyên bảng tra cứu ký hiệu toán học Việt - Anh đã thực hiện ở phần trước).*

---

## 10. QUY TRÌNH HUẤN LUYỆN & TRIỂN KHAI THỰC TẾ CỦA S³

Dưới đây là sơ đồ quy trình chi tiết mô tả 2 pha hoạt động của mô hình S³:

### A. Quy trình Huấn luyện (Training Workflow)
Quy trình huấn luyện diễn ra hoàn toàn trên ma trận tài liệu thô để học được các trục chủ đề ẩn:
```
[D tài liệu văn bản thô] 
        │
        ▼ (Mã hóa qua Sentence Transformer)
[Ma trận Document Embeddings X] (Kích thước D x d)
        │
        ▼ (Tính ma trận hiệp phương sai & dọn dẹp tương quan bằng SVD)
[Làm trắng (Whitening) & Giảm chiều về N chủ đề]
        │
        ▼
[Ma trận đã làm trắng X_white] (Kích thước D x N)
        │
        ▼ (Cập nhật lặp fixed-point cực đại hóa Negentropy)
[Tối ưu hóa FastICA để xoay trục độc lập]
        │
        ▼
[Ma Trận Trộn A] (d x N) ───► (Tọa độ của N trục chủ đề trong không gian d chiều)
[Ma Trận Nguồn S] (N x D) ───► (Tỷ lệ phân phối N chủ đề trong D tài liệu)
```

### B. Quy trình Triển khai & Suy luận (Implementation & Inference Workflow)
Quy trình triển khai chiếu từ vựng lên trục để trích xuất từ mô tả và suy luận tài liệu mới:
```
          [Ma Trận Trộn A học được] (d x N)
                    │
                    ▼ (Tính nghịch đảo giả Moore-Penrose: C = A⁺)
          [Ma Trận Unmixing C] (N x d)
                    │
         ┌──────────┴──────────┐
         ▼ (Trích xuất từ khóa)  ▼ (Suy luận tài liệu mới)
  [Mã hóa từ vựng V]    [Mã hóa tài liệu mới X_hat]
   (V_vocab x d)          (D_new x d)
         │                     │
         ▼ (Nhân W = V Cᵀ)     ▼ (Nhân S_hat = X_hat Cᵀ)
  [Ma trận chiếu từ W]  [Phân phối chủ đề mới S_hat]
   (V_vocab x N)          (D_new x N) (Tính tức thời!)
         │
         ▼ (Tính Combined word importance)
  [Top từ khóa mô tả từng chủ đề]
```
