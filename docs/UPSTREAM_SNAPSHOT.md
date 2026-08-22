# Snapshot repository trước khi nâng cấp benchmark

Nguồn kiểm kê: `https://github.com/ThienUIT/NLP_S3`, nhánh `main`, commit `d373a7db071a262973e4e7e7283f3ffdcff704b1` (truy cập ngày 22-08-2026).

Repository hiện có `s3_reproduction/`, test, demo/server, bản thảo LaTeX và Git LFS cho dữ liệu cùng artifact lớn. CLI hiện có chạy CafeBERT masked-mean với Turftopic hoặc custom S³ trên các corpus hiện hữu; timing hiện được ghi theo segment, embedding, model và tổng. Bản nâng cấp sẽ bổ sung riêng một gói benchmark không thay đổi hành vi các lệnh cũ: bốn corpus, sáu mô hình, seed đa giá trị, provenance, WEC-in, C_NPMI, diversity, timing stage-wise, audit và report LaTeX.

Checkpoint `uitnlp/CafeBERT` là dependency pretrained công khai được tham chiếu theo model identifier và revision/checksum khi khả dụng; S³ không được mô tả là có checkpoint pretrained riêng trong gói benchmark này.

## Nguồn dữ liệu đã xác minh khi đóng gói

Các revision nguồn được ghi trong `benchmark/cafebert_full/fetch_sources.py`. UIT–ViSFD dùng commit `4b11ec2b518e8566e58ac0622a4a94976b45b695` của `LuongPhan/UIT-ViSFD`, nơi dữ liệu gồm `Train.csv`, `Dev.csv` và `Test.csv` bên trong archive. ViMedical Disease dùng commit `2c2cb3909754a05346625d3b1aed609c1f5e0312` của `PB3002/ViMedical_Disease`. VNTC dùng commit `533a3d6e1a78d73cde5dcfaf867cf8fe62c1fca8` của `duyvuleo/VNTC`; archive cần giải nén là `Data/27Topics/Ver1.1/Train.rar` và `Test.rar`, vì loader dùng nội dung 27 chủ đề để lọc `Khoa hoc - Cong nghe`.
