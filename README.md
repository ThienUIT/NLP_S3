# NLP_S3

Slide thuyết trình & tài liệu học về bài báo **S³ — Semantic Signal Separation** (ACL 2025):
https://aclanthology.org/2025.acl-long.32/

Slide viết bằng [reveal.js](https://revealjs.com/) (tải qua CDN) trong file HTML đơn — **không cần build, không cần cài dependency**. Chỉ cần trình duyệt (và mạng để tải CDN).

---

## 👉 Anh em làm việc ở đâu?

> **Làm việc chính trong thư mục [`main/`](main/).** Đây là bản thuyết trình chung của cả nhóm.

- **[`main/main.html`](main/main.html)** — SLIDE CHÍNH (bản thuyết trình, mạch "hook-first" theo 4 Act).
- **[`main/content.md`](main/content.md)** — kịch bản lời nói đi kèm (câu dẫn, thời lượng từng slide).

Mọi chỉnh sửa cho buổi báo cáo → sửa 2 file trong `main/`. Các file trong `temp/` chỉ là bản nháp/tài liệu học tham khảo, **không** dùng để trình bày.

---

## Cấu trúc thư mục

```
main/                  ← LÀM VIỆC Ở ĐÂY
  main.html            slide chính (thuyết trình)
  content.md           kịch bản/ghi chú cho main.html

temp/                  ← bản nháp & tài liệu học (tham khảo)
  slides.html          bản HỌC đầy đủ (có mục "Kiến thức nền")
  slides_3.html        bản HỌC SÂU Phần 3 (thuật toán, cho Người 3)
  content.md           bản chắt lọc nội dung chi tiết từng thuật ngữ
  google/              các bản slide phụ (google_slides*.html)

2025.acl-long.32.pdf   bài báo gốc
Makefile               lệnh chạy/mở/deploy nhanh
vercel.json            cấu hình đường dẫn khi deploy Vercel
```

---

## Lệnh Make (chạy & mở nhanh)

Gõ `make help` để xem danh sách. Đổi cổng bằng `PORT=xxxx` (mặc định 8000).

| Lệnh | Việc |
|---|---|
| `make main` | Chạy server + mở **main/main.html** — *slide chính, dùng cái này để trình bày* |
| `make run2` | Alias của `make main` (giữ tương thích cũ) |
| `make run` | Chạy server + mở **temp/slides.html** (bản HỌC đầy đủ) |
| `make run3` | Chạy server + mở **temp/slides_3.html** (HỌC SÂU Phần 3) |
| `make rungoogle` | Chạy server + mở **temp/google/google_slides.html** |
| `make rungoogle1` | Chạy server + mở **temp/google/google_slides_1.html** |
| `make open` | Mở **temp/slides.html** trực tiếp (không qua server) |
| `make open2` | Mở **main/main.html** trực tiếp (không qua server) |
| `make opengoogle` | Mở **temp/google/google_slides.html** trực tiếp |
| `make opengoogle1` | Mở **temp/google/google_slides_1.html** trực tiếp |
| `make serve` | Chỉ chạy local server tại `http://localhost:8000/` (tự vào đường dẫn) |
| `make deploy` | Deploy lên **Vercel production** (`vercel deploy --prod`) |
| `make clean` | Xoá file tạm (`.vercel/`) |

Ví dụ:
```bash
make main            # mở slide chính để tập thuyết trình
make run3 PORT=9000  # mở bản học sâu Phần 3 ở cổng 9000
make deploy          # đẩy bản mới nhất lên web
```

Ghi chú:
- `make main / run / run3 / ...` tự khởi động một local server rồi mở trình duyệt; nhấn `Ctrl+C` để dừng server.
- `make open* ` mở thẳng file (`file://`) — nhanh hơn nhưng vài trình duyệt có thể chặn tính năng do CORS; khi đó dùng `make main`/`make run` (qua server) cho chắc.
- Lệnh dùng `python3` và `open` (macOS). Máy khác chỉnh `open` thành `xdg-open` (Linux) hoặc `start` (Windows).

---

## Xem trên web (Vercel) — mở được trên điện thoại

Dự án deploy qua **Vercel** (miễn phí). Một domain, nhiều đường dẫn (cấu hình trong `vercel.json`):

| URL | Nội dung |
|---|---|
| `minh-internal-nlp-slides.vercel.app/` | **main/main.html** — slide chính |
| `…/2` hoặc `…/main` | main/main.html (cùng slide chính) |
| `…/1` | temp/slides.html (bản HỌC đầy đủ) |
| `…/3` | temp/slides_3.html (HỌC SÂU Phần 3) |
| `…/g1` `…/g2` `…/g3` | temp/google/... |

Repo đã kết nối Git với Vercel → mỗi `git push` lên `main` sẽ **tự deploy**. Muốn đẩy ngay bản local (chưa commit) thì dùng `make deploy`.

---

## Điều hướng slide

- `→ / ←` hoặc `Space`: chuyển slide kế tiếp/trước.
- `↓ / ↑`: xuống/lên slide con (nếu có).
- `Esc`: xem tổng quan tất cả slide (overview).
- `F`: toàn màn hình · `S`: speaker notes · `M`: menu nhảy nhanh theo mục.
- Thanh tiến trình + số thứ tự slide hiển thị ở góc dưới.

