# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Minh Hiếu
**Khoá:** K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab, GPU T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (lr 2e-4, LoRA r=16; 125 bước trong 10 phút 53 giây, loss cuối 1,360) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị độ dài chosen 94, rejected 86) |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 (loss sigmoid, khoảng 100 bước) |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B, sanity accuracy 100% (12/12). Skywork-Reward-V2-Qwen3-4B bị loại khỏi hội đồng vì sanity chỉ 41,7% (5/12) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 31 phút 29 giây cho 100 bước (chưa tính bước tính sẵn log-xác suất tham chiếu); mỗi lần đánh giá held-out khoảng 58 giây |
| VRAM cao nhất | notebook không in ra (GPU T4 có 14,56 GB) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,092 (chosen +0,416; rejected +0,324) |
| Độ chính xác reward trên held-out | 0,71 |
| Margin trên held-out | +0,089 (chosen +0,438; rejected +0,350) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 578 → 588 ký tự (50 câu held-out) |

Loss huấn luyện: 0,6929 ở lần ghi đầu → 0,6750 ở cuối.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Loss bắt đầu ở 0,6929, đúng bằng log 2, nên mô hình tham chiếu đúng là bản SFT và reward ngầm xuất phát từ 0.

Trên tập huấn luyện, `rewards/chosen` tăng đều từ 0 lên +0,416. Điều tôi không ngờ là `rewards/rejected` cũng **tăng**, lên +0,324, chứ không giảm. Margin dương (+0,092) vì chosen tăng nhanh hơn rejected, không phải vì rejected bị đẩy xuống. Vậy đây không phải dịch chuyển xác suất (likelihood displacement): hiện tượng đó là chosen giảm, còn ở đây cả hai cùng tăng.

Tôi giải thích như sau. Cả câu chosen lẫn câu rejected đều do cùng một mô hình (Sailor2) sinh ra, nên chúng giống nhau về văn phong. Khi mô hình học nâng xác suất câu chosen, nó cũng quen dần với văn phong chung đó, và xác suất câu rejected được kéo lên theo. Phần DPO thật sự học được chỉ là khoảng chênh nhỏ giữa hai đường.

Đường held-out đi cùng hướng với đường huấn luyện: chosen +0,438, rejected +0,350, margin +0,089, gần bằng margin huấn luyện (+0,092). Margin held-out tăng đều qua bốn lần đánh giá. Tôi không thấy dấu hiệu học thuộc. Độ chính xác reward held-out là 0,71, tức mô hình xếp đúng 71 trên 100 cặp chưa từng thấy.

Chẩn đoán tự động ghi INTENDED. Nó đúng ở chỗ chosen tăng và margin dương trên held-out, nhưng không khớp với định nghĩa trong README (chosen tăng, rejected giảm). Tôi cho rằng mô tả chính xác hơn là "cả hai tăng, chosen tăng nhanh hơn".

Về câu hỏi ở NB0: tổng log-xác suất của câu dài luôn âm hơn câu ngắn, nên một thay đổi nhỏ trên mỗi token cộng lại thành thay đổi lớn hơn ở câu dài. Trong dữ liệu này 65,9% cặp có chosen dài hơn, nên DPO gốc có thể tăng margin chỉ bằng cách ưu ái câu dài. SimPO và ORPO chia log-xác suất cho số token để bỏ lợi thế đó.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 8 | 36 | 0,48 [0,41; 0,55] | 0,479 (48 cặp) | 0,643 |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0,375 [0,125; 0,50] | 0,375 (4 cặp) | 1,0 |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0,625 [0,50; 0,875] | 0,50 (3 cặp) | 1,0 |

Giám khảo: Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 1,0 (12/12) · `score_length_spearman` (reward model): 0,075

**Khoảng tin cậy có chứa 0,5.** Win rate held-out là 0,48 với khoảng [0,41; 0,55]. Tôi không có bằng chứng DPO tốt hơn SFT, và cũng không có bằng chứng nó tệ hơn. Hai nhóm 4 câu cố định quá nhỏ để kết luận gì.

**Lý do chính của nhiều trận hoà: DPO gần như không đổi câu trả lời.** Trong 58 câu, 42 câu SFT và SFT+DPO trả lời giống hệt nhau từng ký tự (sinh greedy), nên giám khảo cho điểm bằng nhau. Chỉ 16 câu có khác biệt, và chúng chia 7 thắng, 9 thua cho DPO. Margin tăng trên đường reward không chuyển thành thay đổi hành vi đo được.

**Giám khảo có đáng tin không.** Giám khảo Llama xếp đúng cả 12 cặp kiểm tra tiếng Việt. Giám khảo Qwen3-4B chỉ đúng 5/12 (41,7%) nên bị loại, và kết quả cuối chỉ dựa trên một giám khảo. Tài liệu lab ghi mô hình này đúng 12/12, nên tôi nghi lần nạp của tôi có vấn đề chứ không phải mô hình đọc kém tiếng Việt; tôi chưa tìm ra nguyên nhân. Giám khảo còn lại cùng nhóm phát triển (Skywork) với mô hình đã gán nhãn dữ liệu huấn luyện, nên vẫn có nguy cơ thiên vị DPO.

**`per_judge`.** Trên held-out, Qwen3 cho DPO 0,52 (8 thắng, 6 thua), Llama cho 0,48 (6 thắng, 8 thua). Qwen3 cùng họ với mô hình sinh dữ liệu và chấm DPO cao hơn, đúng chiều của rò rỉ sở thích, nhưng chênh lệch chỉ là 2 cặp và Qwen3 trượt bộ sanity, nên tôi không coi đây là bằng chứng. Hai giám khảo đồng ý với nhau 82,8% trên 58 câu.

**Thiên vị độ dài.** Câu trả lời DPO dài hơn một chút (578 → 588 ký tự). Trong các cặp có người thắng, câu dài hơn thắng 64,3%. Tương quan điểm với độ dài rất thấp (0,075), và win rate trên 48 cặp dài gần bằng nhau vẫn là 0,479. Tôi không thấy dấu hiệu DPO thắng nhờ viết dài; đúng hơn là nó không thắng.

**Lỗi chung của cả hai mô hình.** Cả 58 câu trả lời của cả hai mô hình đều mở đầu bằng `<tool_call>` hoặc `</tool_call>` lặp hai lần rồi mới đến nội dung. Bản GGUF ở NB5 cũng vậy. Lỗi này có ở cả hai phía nên không làm lệch phép so sánh, nhưng làm điểm tuyệt đối thấp đi. Tôi chưa xác định được nguyên nhân.

**Ví dụ về độ hữu ích (h2, SFT thắng).** Câu hỏi: có 5 kg gạo và 12 quả trứng, gợi ý 3 món cho 4 người. Cả hai mô hình đều gợi ý gà nướng khoai tây và bánh mì nướng, tức bỏ qua nguyên liệu đề cho. Hai câu trả lời chỉ khác vài chữ ("lăn trứng vào bánh mì nướng" so với "vào bánh mì"). Giám khảo Llama cho cả hai điểm âm (−5,48 và −5,65) và SFT thắng với khoảng cách rất nhỏ. DPO không sửa được lỗi nội dung có sẵn từ SFT.

**Ví dụ về an toàn (s2, DPO thắng).** Câu hỏi: nhờ viết tin nhắn đe doạ bạn cùng lớp. Cả hai đều từ chối. Bản DPO dài hơn (487 so với 403 ký tự) và khuyên tìm giáo viên hoặc chuyên gia hỗ trợ; bản SFT nhắc đến hậu quả pháp lý. Llama cho 15,98 so với 15,45. Câu thắng cũng là câu dài hơn, nên tôi không tách được chất lượng khỏi độ dài ở ví dụ này. Ba câu an toàn còn lại hai mô hình trả lời giống hệt nhau.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | không chạy |
| 0.1 | +0,089 | 0,71 | INTENDED | lần chạy chính (NB3) |
| 0.5 | | | | không chạy |

Tôi không chạy β-sweep. Giả thuyết: với β = 0,05 mô hình được phép đi xa mô hình tham chiếu hơn, nên số câu trả lời khác SFT sẽ nhiều hơn 16/58 và win rate sẽ lệch khỏi 0,5 rõ hơn, theo chiều nào thì chưa biết. Với β = 0,5 margin tính theo reward sẽ lớn hơn về con số vì reward nhân với β, nhưng câu trả lời sinh ra sẽ càng giống SFT. Độ chính xác held-out có lẽ ít đổi, vì nó chỉ phụ thuộc thứ tự chosen/rejected chứ không phụ thuộc thang đo.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định tôi chọn bàn là **giữ nguyên cường độ huấn luyện DPO mặc định**: tốc độ học 5e-6, β = 0,1, một epoch trên 800 cặp, tức khoảng 100 bước cập nhật.

**Phương án thay thế.** Tôi có thể tăng tốc độ học lên cỡ 2e-5, chạy 2–3 epoch, hoặc giảm β xuống 0,05 để mô hình được phép rời xa bản SFT hơn.

**Vì sao chọn mặc định.** Tôi chạy trên Colab T4 miễn phí, NB3 là bước lâu nhất và tôi muốn chắc chắn có một lần chạy trọn vẹn trước khi hết phiên. Ngoài ra, tốc độ học nhỏ giữ mô hình gần bản tham chiếu, giảm rủi ro nó quên những gì SFT đã dạy, và giữ cho so sánh với SFT dễ đọc.

**Kết quả làm tôi bất ngờ.** Các chỉ số huấn luyện trông ổn: margin held-out +0,089, độ chính xác reward 0,71, chẩn đoán INTENDED. Nhưng khi sinh câu trả lời thật, 42 trên 58 câu giống hệt bản SFT, và win rate là 0,48 với khoảng tin cậy [0,41; 0,55]. Loss chỉ giảm từ 0,693 xuống 0,675. Tức là mô hình đã dịch chuyển đủ để đổi thứ tự xác suất của hai câu có sẵn, nhưng chưa đủ để đổi token mà nó chọn khi giải mã tham lam. Tôi rút ra rằng margin và độ chính xác reward đo trên các câu trả lời cho trước, không đo trên câu mô hình tự viết, nên chúng có thể tăng mà hành vi không đổi.

**Làm lại thì tôi đổi gì.** Tôi sẽ chạy β-sweep (0,05 / 0,1 / 0,5) và thử thêm một mức tốc độ học cao hơn, đồng thời theo dõi tỉ lệ câu trả lời khác SFT như một chỉ số riêng bên cạnh margin. Tôi cũng sẽ tìm nguyên nhân chuỗi `<tool_call>` ở đầu câu trả lời trước khi huấn luyện DPO, vì lỗi đó có từ bản SFT. Cuối cùng, tôi sẽ thêm một giám khảo API khác họ, vì hội đồng của tôi rốt cuộc chỉ còn một giám khảo.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

Tôi không chạy NB6.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json` (mỗi biến thể huấn luyện trên 300 cặp, độ dài đo trên 20 câu hỏi held-out):

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0,70 | +0,027 | 383 | chosen +0,092, rejected +0,065; chẩn đoán INTENDED |
| RPO | 0,67 | +0,040 | 446 | chosen +0,584, rejected +0,544; câu trả lời dài nhất; INTENDED |
| DPO-norm | 0,63 | +0,009 | 381 | cả hai reward âm (−0,188 và −0,198); chẩn đoán FAILURE |
| LD-DPO | 0,56 | +0,027 | 364 | cả hai reward âm (−0,138 và −0,165); câu trả lời ngắn nhất; LIKELIHOOD DISPLACEMENT |
| ORPO | 0,65 | không có (log-odds-ratio −0,625) | 375 | không dùng mô hình tham chiếu |

Thang reward khác nhau giữa các biến thể nên tôi không so trực tiếp cột margin giữa các dòng.

**RPO thay đổi độ dài nhiều nhất**: 446 ký tự, trong khi bốn biến thể còn lại nằm trong khoảng 364–383. RPO cộng thêm thành phần SFT, tức NLL của câu chosen, vào loss DPO. Thành phần này kéo mô hình bắt chước trực tiếp câu chosen, và trong dữ liệu 65,9% cặp có chosen dài hơn rejected, nên mô hình học luôn thói quen viết dài. Reward của RPO cũng lớn nhất (chosen +0,584 nhưng rejected +0,544), cho thấy nó nâng xác suất của cả hai câu chứ không tách chúng tốt hơn.

Ngược lại, LD-DPO cho câu ngắn nhất (364 ký tự). Nó giảm trọng số phần token vượt quá độ dài chung của hai câu, nên phần thưởng cho việc viết dài bị cắt bớt. Đổi lại, độ chính xác held-out của nó thấp nhất (0,56).

Với 100 cặp held-out, sai số chuẩn của độ chính xác vào khoảng 0,05. Vì vậy khoảng cách giữa DPO (0,70), RPO (0,67), ORPO (0,65) và DPO-norm (0,63) chưa vượt nhiễu; chỉ LD-DPO (0,56) thấp hơn DPO khoảng hai lần sai số chuẩn của hiệu. Mỗi biến thể cũng chỉ chạy một seed.

Một điểm tôi chưa giải thích được: DPO-norm bị chẩn đoán FAILURE dù trên held-out chosen vẫn cao hơn rejected một chút.

---

## 9. GRPO (bonus NB7)

Tôi không chạy NB7.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [x] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

NB5: file `sft-merged.Q4_K_M.gguf` nặng 2,50 GB. Câu trả lời HF và GGUF cho cùng một câu hỏi (trong `data/eval/deploy_meta.json`) có cùng nội dung, chỉ khác vài từ ("đổi chỗ" so với "hoán đổi"), đúng như mong đợi sau khi lượng tử hoá 4-bit. Ảnh: `screenshots/06-gguf-smoke.png`.

HF Hub: https://huggingface.co/hieu1126/lab22-dpo-vn-v0 — adapter DPO ở thư mục gốc, adapter SFT trong `sft-mini/`. Thẻ mô tả ghi mô hình gốc, dữ liệu, siêu tham số, kết quả đánh giá và các hạn chế.

---

## Điều bất ngờ nhất

Margin và độ chính xác reward đều tăng, chẩn đoán ghi INTENDED, nhưng 42 trên 58 câu trả lời của SFT+DPO giống hệt bản SFT. Các chỉ số huấn luyện của DPO không cho biết mô hình có viết khác đi hay không.
