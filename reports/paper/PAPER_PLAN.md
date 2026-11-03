# Kế hoạch thí nghiệm cho paper (ghi trước khi có kết quả)

Ngày ghi: 30/10/2026, trước khi chạy bất kỳ thí nghiệm nào dưới đây. Mục đích: cố định câu hỏi, chỉ số và tiêu chí đọc kết quả, để không chọn số sau khi đã thấy dữ liệu. Nếu đổi kế hoạch, ghi lại lý do ở cuối file thay vì sửa nội dung cũ.

Mọi công thức nằm trong `experiments/<ID>/config.yaml`. Notebook: `colab_paper.ipynb`. Bảng: `scripts/build_paper_tables.py` ghi vào `reports/paper/tables/`.

## A. Rò rỉ video

**Câu hỏi:** cho clip của cùng video vào Train làm WER đẹp lên bao nhiêu?

**Thiết kế** (`scripts/build_leak_data.py`, `data/ViMedCSS_leak/split_report.json`):
- Hai bộ Train cùng 29.913 clip (53,23 và 53,22 giờ). Khác nhau ở 722 clip: bản sạch lấy từ video khác, bản rò rỉ lấy từ chính các video trong tập chấm.
- Tập chấm dùng chung: `probe_seen` (809 clip, 192 video, có clip cùng video trong Train rò rỉ) và `probe_unseen` (1.409 clip, 179 video, đối chứng, không rò rỉ ở cả hai bản).
- Thí nghiệm: small 3 seed; medium và large-v3 mỗi loại 1 seed.

**Chỉ số chính:** difference-in-differences (DiD) = (WER sạch − WER rò rỉ trên seen) − (WER sạch − WER rò rỉ trên unseen). Khoảng tin cậy 95% bằng bootstrap ghép cặp theo video.

**Đọc kết quả:**
- Rò rỉ có tác dụng thật nếu khoảng tin cậy của DiD nằm hoàn toàn trên 0 ở cả 3 seed của small.
- Khoảng tin cậy chứa 0: rò rỉ không có tác dụng đo được. Kết quả âm tính này vẫn được báo cáo, và trọng tâm bài chuyển sang B và C.

**Giới hạn đã biết:**
- Tỷ lệ rò rỉ ở đây khoảng 47% clip của video, thấp hơn mức 64% trong split chính thức, nên kết quả có xu hướng đo thấp hơn thực tế.
- Rò rỉ được mô phỏng trên split của nhóm, không đo trực tiếp số của paper ViMedCSS.

## B. Dữ liệu, backbone và oversampling (split chính thức)

**Câu hỏi:** kết quả O3W đến từ đâu?

| Đối chiếu | Đo tác dụng của |
|---|---|
| D1 → D3 | thêm nguồn dữ liệu, small |
| D4 → O2 | thêm nguồn dữ liệu, large-v3 |
| D3 → O2 | backbone small → large-v3 |
| O2 → O3W | oversampling thuật ngữ hiếm |
| O3W → C1 | train có prompt (không dùng danh sách khi chấm) |

D1 là thiết lập gần paper nhất (small + LoRA, chỉ ViMedCSS), so với 27,13 của paper. **Khác paper:** công thức của nhóm (LoRA r32, SpecAugment, 3 epoch) và cách chấm (lệch 5–10 điểm, mục 3.4 báo cáo Task 2).

**Chỉ số:** WER, CER, CS-WER, recall thuật ngữ trên TEST và HARD; khoảng tin cậy ghép cặp theo video. Đối chiếu nào có khoảng tin cậy chứa 0 thì kết luận là không đo được tác dụng.

O3W greedy được giải mã thêm chỉ để so cùng cách giải mã với O2. Hệ cuối đã chốt vẫn là O3W beam 5.

## C. Gợi ý thuật ngữ

**Câu hỏi:** danh sách thuật ngữ có tăng recall thuật ngữ chưa thấy mà không làm hại phần còn lại không? Trọng tâm là HARD, vì 279/338 thuật ngữ của HARD không có trong Train chính thức (TEST chỉ có 1).

| Danh sách | Nguồn | Vai trò |
|---|---|---|
| `oracle30` | thuật ngữ của clip + 30 gây nhiễu | giao thức chuẩn, **mức trần**, không phải kết quả triển khai |
| `video` | thuật ngữ của video trong split | gần oracle vì trung bình chỉ 1,6 clip mỗi video trên TEST |
| `retrieve30` | giải mã 2 lượt, 30 thuật ngữ gần nhất trong từ điển | **kết quả thực tế chính**, không dùng nhãn của clip |
| `distract30` | chỉ 30 thuật ngữ gây nhiễu | đo tác hại (chèn nhầm thuật ngữ) |

**Chỉ số chính:** recall thuật ngữ chưa thấy trên HARD với `retrieve30`, so với không danh sách. **Chỉ số an toàn:** WER và tỷ lệ chèn nhầm trên `distract30`. Cột `gold_in_list` cho biết danh sách có chứa thuật ngữ đúng không; recall không thể vượt quá giới hạn này nhiều.

**Đọc kết quả:** phương pháp có ích nếu recall chưa thấy tăng rõ, trong khi WER với `distract30` không xấu hơn đáng kể so với không danh sách.

**Giới hạn đã biết:**
- Từ điển của `retrieve30` gồm cả thuật ngữ có trong tập chấm, đóng vai từ điển y khoa ngoài. Vì thế phủ 100% thuật ngữ, cao hơn từ điển thật.
- Không dùng thuật ngữ từ TEST/HARD khi train: `train_terms` chỉ đọc Train.

## Thứ tự và điểm dừng

1. **A1 (small, khoảng 3 giờ).** Xem DiD trước khi chạy tiếp. Đây là điểm quyết định hướng bài.
2. **B1, C1** (khoảng 9,5 giờ).
3. **A2, B2** nếu còn đơn vị (khoảng 11,5 giờ).

Thời gian là ước lượng, chưa đo.

## Việc không cần GPU

- 2 người nghe lại khoảng 300–500 clip, tính Cohen's Kappa (kiểm chứng nhãn lệch audio).
- Hỏi giảng viên về quy định môn học và thứ tự tác giả.
- Đọc giấy phép ViMedCSS trước khi công bố danh sách ID của split.

## Thay đổi so với kế hoạch

### 31/10/2026, trước khi chạy bất kỳ thí nghiệm A, B, C nào

Lý do: rà soát phản biện độc lập. Chưa có mô hình `L_*`, `D*`, `O2`, `O2R`, `C1` nào được train; các số dưới đây chỉ đo trên dữ liệu, trên DEV hoặc trên kết quả O3W đã có từ trước. Mục nào dưới đây khác nội dung phía trên thì theo mục này.

**Câu hỏi chính của bài.** Định lượng ảnh hưởng của việc để clip cùng video vào Train lên điểm số benchmark ASR y khoa Việt–Anh, kèm quy trình đánh giá tái lập được (A). B giải thích kết quả O3W đến từ đâu. C là thử nghiệm phụ. Bài không tuyên bố "phương pháp mới vượt các paper". Hệ AG của paper không chấm lại được bằng scorer của nhóm, nên số của hệ đó chỉ được trích, không xếp hạng với O3W. Mục tiêu thực tế: journal Q3 hoặc hội nghị tương đương, chỉ nâng lên nếu kết quả cho phép.

**A. Bổ sung cách đọc:**
- Giữ tiêu chí cũ theo từng seed. Thêm dòng gộp 3 seed: DiD trung bình, khoảng tin cậy bootstrap chung theo video, và độ lệch chuẩn giữa các seed. Bootstrap chỉ đo bất định do mẫu video, không thay thế độ lệch giữa seed.
- Khoảng tin cậy chứa 0 nghĩa là chưa có bằng chứng về tác dụng, không có nghĩa là không có tác dụng.
- Kết quả được phát biểu là "tác dụng của việc đưa clip cùng video vào Train". Thí nghiệm không tách được đóng góp của người nói, âm học, chủ đề hay thuật ngữ, không suy ra được quan hệ tuyến tính với tỷ lệ rò rỉ, và mới chỉ trên một corpus. Tách video chưa bảo đảm tách người nói hay kênh.
- Cân bằng đã đo (`A_balance.csv`): 722 clip thêm và 722 clip bỏ gần nhau về giờ (1,41 / 1,42), số từ mỗi clip (26,1 / 26,8), thuật ngữ mỗi clip (1,05 / 1,04), nhưng khác về số thuật ngữ phân biệt (210 / 306). Trên `probe_seen`, số lượt thuật ngữ có trong từ vựng Train tăng từ 802 lên 828 trên 842 lượt; `probe_unseen` giữ nguyên 1.390 trên 1.489 lượt. Trùng nguyên văn bản với Train: 25 → 31/809 clip ở seen, 23 → 27/1.409 clip ở unseen. Đây là một phần cơ chế rò rỉ và phải được nêu khi diễn giải.

**B. Bổ sung đối chứng:** O2 → O3W có thêm 2,8% số bước (29.050 → 29.864 clip mỗi epoch), nên đo cả công thức. Thêm **O2R**: lặp số clip bằng O3W nhưng chọn ngẫu nhiên, cùng số bước. O2R → O3W mới đo riêng việc nhắm vào thuật ngữ hiếm. Mỗi đối chiếu B chỉ có 1 seed, nên khoảng tin cậy không bao gồm biến thiên huấn luyện.

**C. Thay đổi lớn:**
- `retrieve30` ban đầu dùng từ điển chứa sẵn mọi thuật ngữ của tập chấm, nên chỉ còn là **mức trần**. Kết quả triển khai chính đổi sang `retrieve30-mesh2026`: từ vựng Train + MeSH 2026 (`data/lexicons/mesh2026.txt`, 144.862 mục, sha256 `0cb9da98…`, `scripts/build_lexicon.py`). Từ điển này được cố định trước mọi giải mã có danh sách trên TEST/HARD.
- Đo trên DEV chia theo video với giả thuyết V5B, tỉ lệ thuật ngữ chưa thấy lọt vào danh sách 30 mục là **17,7%** (96 lượt), so với 47,9% khi từ điển chứa sẵn đáp án. Với mọi thuật ngữ: 72,5% so với 80,5%. MeSH chứa 21/31 thuật ngữ chưa thấy của DEV này. Số này giới hạn mức cải thiện kỳ vọng: nhóm không kỳ vọng C thành đóng góp chính.
- Định nghĩa **chưa thấy** chặt hơn: `unseen_any` là thuật ngữ không xuất hiện trong transcript Train của cả 4 nguồn. HARD có 249/338 thuật ngữ (475/758 lượt) thuộc nhóm này, thay vì 279/338 (670/758) khi chỉ xét nhãn ViMedCSS. Dữ liệu tiền huấn luyện của backbone không kiểm tra được, nên không gọi là "mô hình chưa từng thấy".
- **Chỉ số chính:** recall khớp nguyên từ trên nhóm chưa thấy (`d_recall_word_unseen`) so với cùng mô hình không danh sách. Có ích nếu khoảng tin cậy ghép cặp theo video nằm trên 0.
- **Chỉ số an toàn, ngưỡng cố định trước:** cận trên khoảng tin cậy của thay đổi WER không quá **+0,5 điểm** với `distract30` và `retrieve30-mesh2026`. Kiểm định không có ý nghĩa không được dùng để kết luận "không gây hại".
- Thêm `false_terms_per_100_clips`: thuật ngữ sinh ra vượt số lần trong tham chiếu, tính trên toàn bộ thuật ngữ đã gán nhãn của ViMedCSS và danh sách của clip, áp dụng cho mọi kiểu danh sách.
- Thêm `recall_word_listed/unlisted` để đo tác động khi danh sách bỏ sót thuật ngữ đúng, và `oracle10`, `oracle50` trên HARD để đo độ dài danh sách.

**Recall thuật ngữ báo cả hai cận.** Khớp chuỗi con nhận cả biến thể chính tả lẫn khớp nhầm; khớp nguyên từ loại cả hai. O3W beam 5: TEST 79,87 / 69,89, HARD 50,26 / 45,78. Các lượt chỉ khớp chuỗi con chủ yếu là biến thể chính tả (`vaccin` trong `vaccine`, `testosteron` trong `testosterone`), nên giá trị thật nằm giữa hai cận.

**So với paper:** chỉ so trực tiếp khi cùng tập, cùng transcript, cùng scorer, tức là các baseline nhóm tự chạy lại và D1. Không dùng phép cộng chênh lệch cách chấm, mức giảm tương đối, hay tập con 100 clip làm bằng chứng vượt paper.

**Ước lượng mới:** khoảng 26,5 giờ A100 (khoảng 265 đơn vị), chưa đo.

**Thứ tự mới:** A1, rồi B1, rồi A2, rồi B2, cuối cùng C1. Lý do: A là đóng góp chính, còn mức 17,7% ở trên cho thấy C khó thành đóng góp chính nếu không có từ điển tốt hơn. Điểm dừng sau A1 giữ nguyên.

**Tăng tốc trên A100 80 GB, không đổi công thức.** Tắt gradient checkpointing ở các mô hình medium/large (gradient giữ nguyên về toán học), giữ worker dữ liệu sống qua các epoch, chạy nhiều job song song trên một GPU. Giữ batch train 32 vì đó là một phần công thức so với O3W, và giữ batch giải mã 16 vì giới hạn độ dài của decode guard lấy theo clip dài nhất trong batch.

**A1 chạy trên laptop (RTX 4050 6 GB), 01/11/2026.** Công thức giữ nguyên (batch hiệu dụng 32, lr, số epoch, seed), khác Colab ở fp16, micro-batch 4 × tích lũy 8, gradient checkpointing, giải mã batch 1. Cả 6 lần chạy A1 phải cùng một máy: không trộn lần chạy laptop và Colab trong A1. Thứ tự: seed 13 (cặp sạch và rò rỉ), xem DiD, rồi seed 14, 15. Lần xem giữa chừng chỉ dùng để quyết định có tốn thêm thời gian máy không: nếu DiD của seed 13 không dương thì được dừng và báo là kết quả thử một seed, không gọi là kết luận âm tính; nếu chạy tiếp thì báo đủ cả 3 seed theo tiêu chí ban đầu.

**Việc không cần GPU, thêm:** khi 2 người nghe lại audio, phân loại thành nhãn thiếu, mô hình chèn sai, cắt đoạn lệch, không rõ; chọn clip ngẫu nhiên, không chọn theo giả thuyết.

### 01/11/2026, sau khi xem seed 13, trước khi giải mã probe của seed 14 và 15

Seed 13 trên laptop: DiD WER +0,30, khoảng tin cậy [−0,63; +1,14], chứa 0. Vì khoảng này chứa 0, tiêu chí "nằm hoàn toàn trên 0 ở cả 3 seed" cho WER không còn đạt được, bất kể hai seed còn lại. Chỉ số chính giữ nguyên và vẫn báo đủ 3 seed kèm dòng gộp.

**Chỉ số phụ mới: DiD recall thuật ngữ khớp nguyên từ**, báo kèm khớp chuỗi con, bootstrap ghép cặp theo video như WER. Chỉ số này được thêm **sau khi nhìn dữ liệu seed 13**: nguyên từ +9,53 [+4,39; +14,69], chuỗi con +10,91 [+6,25; +15,83]. Vì thế seed 13 chỉ là kết quả thăm dò. Phần kiểm chứng là seed 14 và 15, ghi trước khi giải mã probe của hai seed này: lặp lại được nếu khoảng tin cậy của DiD nguyên từ nằm hoàn toàn trên 0 ở cả hai seed. Khoảng tin cậy chỉ đo bất định do mẫu video, không gồm biến thiên huấn luyện. Bài viết phải nêu rõ chỉ số này được thêm sau khi thấy seed 13, và không chọn thêm chỉ số nào khác từ kết quả A1 để báo như phát hiện chính.

### 01/11/2026, chuyển A1 sang Colab A100

Lý do: cần lấy lại laptop và rút thời gian chờ (còn khoảng 20 giờ), không phải vì kết quả; quyết định được đưa ra sau khi đã thấy seed 13 nên phải nêu trong bài. Lần chạy laptop dừng ở bước 1.742/2.802 của bản sạch seed 14. Seed 14 và 15 chưa giải mã probe trên bất kỳ máy nào.

- A1 chạy lại đủ 6 lần trên Colab A100, cùng công thức trong `config.yaml`. Quy tắc một máy giữ nguyên: bảng chính chỉ gồm 6 lần chạy Colab.
- Cặp seed 13 trên laptop là thử nghiệm sơ bộ. Số của cặp này được báo riêng, không gộp vào bảng chính và không dùng thay lần chạy Colab nào.
- Tiêu chí WER giữ nguyên, nay áp dụng cho 3 seed trên Colab. Câu "không còn đạt được" ở mục trên chỉ nói về loạt laptop.
- Kiểm chứng recall giữ nguyên ngưỡng: khoảng tin cậy DiD khớp nguyên từ nằm hoàn toàn trên 0 ở seed 14 và 15 trên Colab. Seed 13 trên Colab được báo cùng bảng nhưng không tính vào phép kiểm chứng này, vì chỉ số được chọn sau khi xem seed 13.

### 01/11/2026, dừng thí nghiệm để viết báo cáo Task 2

Nhóm dừng ở đây để viết báo cáo môn. Chưa có kết quả Colab nào của A, B, C. Seed 14 và 15 chưa được giải mã probe trên máy nào, nên quy tắc ở hai mục trên vẫn áp dụng nếu sau này chạy tiếp. Báo cáo Task 2 chỉ ghi cặp seed 13 là thử nghiệm sơ bộ, không kết luận về rò rỉ.
