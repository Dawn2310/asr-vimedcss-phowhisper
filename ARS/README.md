# ARS — notebook trình bày ASR Task 2

Bộ notebook để xem lại toàn bộ quá trình mà **không phải huấn luyện lại**. Mọi bảng, hình và log được đọc từ `saved/`
(bản sao kết quả thật lấy từ thư mục `ASR/`). Output đã được lưu sẵn trong từng notebook, mở ra là xem được.

| Notebook | Nội dung |
|---|---|
| `00_Tong_quan` | Tên phiên bản, bảng kết quả DEV chung, TEST/HARD của V-Final |
| `01_Du_lieu` | Nguồn dữ liệu, làm sạch, chia tập, mất cân bằng, oversampling trước/sau, rò rỉ video |
| `02_Baseline` | 2 baseline zero-shot, log đánh giá, so với paper, gap analysis |
| `03_V1` | Cấu hình, log huấn luyện, đường cong loss/WER, so với baseline, phân tích lỗi, ma trận nhầm lẫn, thuật ngữ |
| `04_V2` | Lý do từ lỗi V1, log huấn luyện, so với V1 |
| `05_V_Final` | Thành phần và lý do, lệnh Colab, nhật ký chạy, DEV/TEST/HARD, so với paper, ví dụ đầu ra |
| `06_Thanh_phan_Kha_thi` | Thí nghiệm từng thành phần (bộ chia theo video), tốc độ, kích thước, kết luận khả thi |

Tên phiên bản: Baseline 1 = `B0`, Baseline 2 = `B1`, V1 = `V1`, V2 = `V2`, V-Final = `O3W` (thư mục trong `ASR/experiments`).

- `saved/`: bảng CSV/JSON, `trainer_state.json`, metrics, cấu hình, dự đoán đã chuẩn hóa, log rút gọn (khoảng 5 MB).
- `tools/build.py`: dựng lại toàn bộ thư mục từ `ASR/` (chỉ đọc ASR, chạy trên CPU). ARS đặt bên cạnh `ASR/` hoặc bên trong
  repo `ASR` đều được: `.venv-asr\Scripts\python.exe ARS\tools\build.py`. Cần máy gốc vì đọc `ASR/local/runs` và `ASR/local/logs`
  (không đưa lên GitHub); chỉ xem notebook thì không cần.
- Log đầy đủ không rút gọn vẫn ở `ASR/notebooks/local` và `ASR/local/logs` (log thô chỉ có trên máy gốc).
