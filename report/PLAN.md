# Kế hoạch báo cáo ACL — S³ tiếng Việt

Mục tiêu: 6-8 trang nội dung (không tính references), khuôn ACL (2 cột), LaTeX.
File này là bản nháp cấu trúc để hai bên chốt trước khi viết `.tex` thật —
sửa trực tiếp vào đây khi thống nhất.

## Nguồn tư liệu đã kiểm tra

- **Paper gốc**: `2025.acl-long.32.pdf` (34 trang, đã đọc §1-§8 + Appendix
  A-C, đọc trang phụ lục I để biết còn gì). Không có thư mục `docs/` trong
  repo — nếu ý bạn là "docs" khác, nói rõ tên/đường dẫn.
- **Kịch bản tiếng Việt đã có** (`main/content.md`, `main/act2_from_pptx.md`):
  33 slide, chia 4 Act, đã tự đối chiếu với đúng mục nào trong paper gốc (ghi
  ở cuối `content.md`). Đây gần như là **outline sẵn cho phần Background +
  Method** của báo cáo — chỉ cần văn xuôi hoá, không cần nghiên cứu lại.
- **Hình sẵn có trong `main/`**: `figure3.png`, `figure7.png`, `table2.png`
  (chụp/tái tạo từ paper gốc — dùng lại cần ghi "reproduced from Kardos et
  al., 2025"), `topic_coherence.png`, `topic_diversity.png`, `runtime.png`,
  `robustness_wecex.png`, `robustness_wecin.png` (không chắc nguồn — cần hỏi,
  xem mục Câu hỏi bên dưới).
- **Thực nghiệm của nhóm** (toàn bộ phiên làm việc này, có số liệu thật, đã
  chạy chứ không suy đoán): xem tổng hợp ở mục 6 bên dưới.

## Cấu trúc đề xuất (ước lượng trang, 2 cột ACL)

| # | Mục | ~Trang | Nguồn nội dung |
|---|---|---|---|
| — | Title + Abstract | 0.25 | Viết mới |
| 1 | Introduction | 0.75 | Viết mới (động lực + đóng góp nhóm) |
| 2 | Related Work / Background | 1.0-1.25 | `content.md` Act 1+2 (BoW→LSA/LDA→embedding→Neural/Clustering baselines) |
| 3 | Method: S³ | 1.25-1.5 | `content.md` Act 3 (pipeline 6 bước, 3 công thức word importance, negative importance, concept compass, inference) — hình `figure3.png`/`figure7.png` |
| 4 | Kết quả gốc (tóm tắt) | 0.5-0.75 | `content.md` Act 4 (metric, tốc độ, Table 3, phát hiện preprocessing) — hình `topic_coherence.png`/`topic_diversity.png`/`runtime.png` |
| 5 | **Thực nghiệm tiếng Việt (trọng tâm, của nhóm)** | 2.0-2.5 | Xem mục 6 |
| 6 | Ứng dụng đề xuất | 0.5 | Dashboard cảnh báo + routing câu hỏi (đã demo thật) |
| 7 | Conclusion & Limitations | 0.5 | Viết mới |

Tổng ~7-7.75 trang — vừa khung 6-8, còn biên độ co giãn khi viết thật.

## Mục 5 — nội dung thực nghiệm đã có sẵn số liệu thật (không cần chạy lại)

1. **Dataset**: ViSFD (11.122 review điện thoại, 11 nhãn khía cạnh thật:
   BATTERY, CAMERA, DESIGN...) + VTSNLP curated (12,169,131 tài liệu, lấy mẫu
   20.000, 25 domain thật: Health, Sports, Real_Estate...).
2. **Tách từ tiếng Việt (ablation)**: CountVectorizer trên văn bản chưa tách
   từ tách vỡ từ ghép ("trầy xước" → 2 âm tiết rời); dùng `pyvi` sửa — có ví
   dụ trước/sau cụ thể + số liệu diversity/coherence.
3. **So sánh encoder CafeBERT vs E5** (đúng yêu cầu đề tài): tốc độ mã hoá
   (34.9s vs 21.9s / 11k review), AUC trung bình 1-trục (0.689 vs 0.722), AUC
   gộp trục (0.867 vs 0.887) — đo trên ViSFD bằng phương pháp ở mục 4. Có cả
   `bge-m3` làm đối chứng thứ 3 nếu muốn thêm (0.719 / 0.898).
4. **Phương pháp đánh giá độ chính xác bằng nhãn thật** — đóng góp phương
   pháp luận riêng của nhóm, paper gốc không làm (§8.6 paper nói rõ họ CHỦ
   ĐÍCH không đánh giá document-topic proportions cho downstream task): dùng
   AUC (point-biserial r + ROC-AUC) so trục ICA với nhãn khía cạnh/domain
   thật. Kết quả:
   - ViSFD (n=40): 3/11 nhãn AUC≥0.75 khi dùng 1 trục, 11/11 khi gộp trục
     (logistic regression 5-fold CV).
   - VTSNLP (n=30): 21/25 domain AUC≥0.75 chỉ với 1 trục (Real_Estate=0.993,
     Sports=0.985) — mạnh hơn ViSFD vì domain là nhãn đơn, tách bạch hơn
     khía cạnh đa nhãn của ViSFD (cần nói rõ đây không phải S³ "tốt hơn" mà
     bài toán dễ hơn).
5. **Quét n_topics** (10/15/20/25/30/35/40/45/50 tuỳ dataset): bảng
   diversity/coherence/aggregate, thảo luận đánh đổi (n cao hơn → tách được
   khía cạnh hiếm nhưng cũng sinh trục nhiễu/trùng chính tả).

## Mục 6 — ứng dụng (ngắn theo đúng yêu cầu, có số liệu)

Hai demo đã dựng và **chạy thật** (không phải mock):
- Dashboard giám sát bất thường (z-score so baseline) — kịch bản giả định
  khủng hoảng pin, bắt đúng tín hiệu (z=-10.01 cho BATTERY, gấp 2x tín hiệu
  mạnh nhì).
- Server định tuyến câu hỏi (FastAPI + WebSocket + Docker riêng, dữ liệu
  `UTS2017_Bank`, 14 nhãn ngân hàng thật) — verify bằng WebSocket client thật.

Có thể chỉ cần 1 đoạn ngắn nêu ý tưởng + 1 con số minh hoạ (ví dụ AUC gộp
trục nhảy từ 3/11 → 11/11 khi làm tagging) theo đúng yêu cầu "không cần nếu
không có số liệu" — ở đây có số liệu thật nên nên giữ.

## Đã chốt (2026-08-22)

- **Ngôn ngữ**: tiếng Việt, trong khuôn định dạng ACL (2 cột, style).
- **Template ACL**: đã tải trực tiếp từ GitHub `acl-org/acl-style-files`
  (chính là nguồn của link Overleaf bạn gửi) — `report/acl.sty`,
  `report/acl_natbib.bst`, `report/acl_lualatex.tex` (mẫu gốc),
  `report/paper.tex` (bài của nhóm, đã ráp Mục 2). **Biên dịch bằng
  XeLaTeX hoặc LuaLaTeX trên Overleaf, KHÔNG dùng pdfLaTeX** (mặc định của
  Overleaf) — vào Menu > Compiler > đổi sang XeLaTeX, nếu không dấu tiếng
  Việt sẽ lỗi/không hiện. Máy tôi không có LaTeX cài sẵn nên chưa tự biên
  dịch thử được — bạn compile trên Overleaf và báo lại nếu lỗi font/dấu.
- **Cấu trúc 7 mục**: giữ nguyên như bảng trên.
- **`robustness_wecex.png`/`robustness_wecin.png`**: sẽ dùng, chờ bạn nói rõ
  nguồn/ý nghĩa (chưa xác nhận — nhắc lại nếu bạn chưa nói).

## Tiến độ viết nội dung (từng mục = 1 file trong report/)

- [x] Mục 2 — Related Work/Background: `report/section2_related_work.md`
- [x] Mục 3 — Method (S³): viết thẳng vào `report/paper.tex` (pipeline 6
      bước, 3 công thức importance + Hình 3, bảng ý nghĩa âm (Table 2 gốc,
      dựng lại bằng LaTeX table thay vì nhúng ảnh), Concept Compass + Hình 7,
      công thức suy luận tài liệu mới Ŝ=X̂Cᵀ)
- [x] Mục 4 — Kết quả gốc (tóm tắt): viết thẳng vào `report/paper.tex`
      (thiết lập, metric, Hình runtime.png, kiểm định thống kê, phát hiện
      robustness tiền xử lý + Hình robustness_wecex.png, định tính + hạn
      chế do chính tác giả nêu)
- [x] Mục 5 — Thực nghiệm tiếng Việt (trọng tâm): viết vào `report/paper.tex`.
      Số liệu AUC (encoder comparison + validate ViSFD/VTSNLP) đã CHẠY LẠI
      trực tiếp trong phiên này (`python -m s3_reproduction.validate_visfd`
      / `validate_curated` --combined) để xác nhận khớp số PLAN.md ghi
      trước đó — khớp chính xác. Bảng n_topics sweep lấy từ
      `artifacts/visualizations/metrics.csv`. Điểm "tách từ tiếng Việt"
      viết theo hướng định tính/mô tả kỹ thuật thật (trích từ
      `s3_reproduction/cli.py`), KHÔNG có số liệu ablation trước/sau định
      lượng (không tìm thấy file kết quả nào lưu lại phép so sánh có/không
      pyvi) — nếu bạn có số liệu đó từ trước, gửi lại để tôi bổ sung.
- [x] Mục 6 — Ứng dụng: viết vào `report/paper.tex`. Số liệu bảng routing
      (Bảng~\ref{tab:bank-routing}) và z-score kịch bản battery-crisis
      CHẠY LẠI trực tiếp trong phiên này trên checkpoint thật đang dùng
      cho server (`uts-bank-e5` n\_topics=30, khớp mô tả trong
      `server/README.md` về việc CUSTOMER_SUPPORT/INTEREST_RATE dùng
      chung trục). Lưu ý: z-score thật đo được là z=-12,79 trên trục có từ
      khoá "nóng/nhiệt" nhưng nhãn hiệu chỉnh gắn cho trục đó là STORAGE
      (không phải BATTERY như bản nháp PLAN cũ ghi) — đã viết lại đúng số
      mới đo được kèm giải thích đây là một hạn chế thật (calibration
      không hoàn hảo), không che giấu.
- [x] Mục 1 — Introduction: viết vào `report/paper.tex` (4 đóng góp: tái
      lập + xử lý tách từ tiếng Việt, so sánh encoder, phương pháp AUC,
      2 ứng dụng; kèm dàn ý các mục còn lại)
- [ ] Mục 7 — Conclusion (viết sau cùng)
- [ ] `report/refs.bib` — build dần theo từng mục cần trích dẫn
