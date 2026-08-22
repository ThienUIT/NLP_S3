# S3 Department Router — server Docker riêng cho định tuyến + thông báo

Container FastAPI/WebSocket tách biệt khỏi Streamlit demo, nạp 1 checkpoint
S3 (đã train trên `undertheseanlp/UTS2017_Bank`, 14 nhãn khía cạnh ngân hàng
thật) và bảng hiệu chỉnh trục↔phòng ban (`axis_labels.json`, tính offline
bằng `validate_uts_bank.py` — xem module đó để biết AUC từng nhãn). Nhận câu
hỏi mới qua HTTP, định tuyến bằng đúng công thức suy luận tài liệu của paper
(`ica.transform()`, §3.1), rồi phát (broadcast) qua WebSocket cho client nào
đang theo dõi đúng phòng ban đó.

## Chạy

```powershell
docker compose up -d --build   # lần đầu / sau khi sửa code
docker compose up -d           # các lần sau (đã build sẵn)
docker compose logs -f         # xem log (lần đầu mất ~20-30s tải model E5)
docker compose down            # dừng + xoá container
```

Container mount sẵn checkpoint từ `artifacts/turftopic/uts-bank-e5/models/`
(đọc, không sửa) — đổi model bằng cách sửa `CHECKPOINT_PATH` trong
`docker-compose.yml`, không cần build lại image.

## Kiểm tra nhanh

```bash
curl http://localhost:8000/health
curl http://localhost:8000/departments
curl -X POST http://localhost:8000/questions -H "Content-Type: application/json" \
     -d '{"text": "the ATM cua em bi nuot mat roi"}'
```

## Xem thông báo trực tiếp

Mở `server/client.html` thẳng bằng trình duyệt (không cần server web, mở file
trực tiếp) — chọn phòng ban, bấm "Theo dõi", cho phép trình duyệt gửi thông
báo (popup xin quyền lúc mở trang). Mỗi câu hỏi mới khớp phòng ban đang theo
dõi sẽ hiện trong trang **và** bật thông báo desktop thật.

## Bắn câu hỏi mẫu để demo luồng sống

```powershell
.\.venv\Scripts\python.exe server\simulate.py --n 15 --delay 3
```
Lấy câu hỏi thật từ chính UTS2017_Bank (không phải bịa), bắn từng câu cách
nhau vài giây, so luôn định tuyến với nhãn thật để biết đúng/sai ngay trên
console.

## Giới hạn thật (đọc trước khi coi là "chuẩn")

- **Độ chính xác định tuyến 1-trục** chỉ ở mức "khá" cho phần lớn nhãn (xem
  `validate_uts_bank.py --combined` — AUC 1-trục thường 0.68-0.85, thấp hơn
  nhiều so với AUC gộp trục 0.75-0.95). Server hiện dùng **1-trục** (đơn giản,
  nhanh, không cần train thêm classifier) — chấp nhận sai một phần, không
  phải rule sản xuất đáng tin tuyệt đối. Muốn chính xác hơn, thay
  `route_question()` bằng logistic regression gộp trục (xem
  `combined_axes_auc` trong `validate_visfd.py`) — cần lưu thêm classifier
  đã train, không chỉ mapping trục↔nhãn.
- **ACCOUNT, SECURITY** không có trong `/departments` — quá ít mẫu (5, 4 dòng)
  để hiệu chỉnh, không phải bug.
- **CUSTOMER_SUPPORT và INTEREST_RATE dùng chung 1 trục** (Topic 13), tương tự
  **MONEY_TRANSFER và SAVING** (Topic 10) — 2 phòng ban trong mỗi cặp sẽ cùng
  nhận thông báo khi trục đó kích hoạt (xem `shared_with` trong
  `axis_labels.json`). Đây là hạn chế thật của model ở n_topics=30, không
  phải lỗi định tuyến.
- **Thông báo là Browser Notification API thật** (không phải mock) nhưng chỉ
  hoạt động khi trang `client.html` đang mở — không phải push service kiểu
  FCM/APNs hoạt động cả khi tắt trình duyệt (cần tài khoản dịch vụ đẩy ngoài,
  ngoài phạm vi demo local này).
