# Documentation: Representation Steering Pipeline

Tài liệu này mô tả toàn bộ hệ thống "Representation Steering" — một trong 3 phương pháp được so sánh trong đề tài *"Steering, Unlearning, or Monitoring? Characterizing the Scope of Package Hallucination Mitigation in Code LLMs"*.

---

## 1. Mục đích

Steering là phương pháp **can thiệp vào hidden state** của một LLM đã đóng băng (frozen) trong lúc sinh code, nhằm giảm khả năng model sinh ra tên package không tồn tại (hallucinated package) — mà **không cần huấn luyện lại toàn bộ model**.

Ý tưởng cốt lõi: học một **vector duy nhất** cho mỗi layer, sao cho khi cộng vector đó vào hidden state ngay trước thời điểm model sắp sinh ra tên package, nó sẽ đẩy hidden state theo hướng "giống với các trường hợp sinh package thật" hơn là "giống các trường hợp sinh package ảo giác".

---

## 2. Kiến trúc & các file

Hệ thống gồm 2 giai đoạn tách biệt, 3 file Python chính:

| File | Vai trò | Giai đoạn |
|---|---|---|
| `tsv_main.py` | Huấn luyện steering vector từ dữ liệu đã gán nhãn | Training |
| `llm_layers.py` | Cơ chế tiêm (inject) vector vào 1 layer cụ thể của model lúc sinh | Cả 2 (được `tsv_main.py`/`eval_variant_v2.py` import) |
| `eval_variant_v2.py` | Dùng vector đã huấn luyện để sinh code + đo tỷ lệ hallucination | Evaluation |

### Sơ đồ luồng dữ liệu

```
                    ┌─────────────────────────┐
                    │  _results.csv           │  (Prompts, Answers, pip_valid, pip_hallucinated, ...)
                    │   (dữ liệu có sẵn)      │   
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      tsv_main.py        │  train steering vector
                    │  --data_path A.csv B.csv│  (gộp nhiều file được)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    tsv_output/steering_vector.pt
                    tsv_output/metadata.json
                    tsv_output/layer_scores.json
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   eval_variant_v2.py     │  sinh code VỚI steering,
                    │  --method steering       │  đo lại PHR/RHR
                    └────────────┬────────────┘
                                 │
                                 ▼
                    eval_runs/<tag>/<tag>_results.csv   (cùng schema, để so sánh)
                    
```

---

## 3. Cơ chế steering hoạt động thế nào

### 3.1. Training (`tsv_main.py`)

1. Với mỗi ví dụ trong dữ liệu, lấy hidden state của model tại **vị trí boundary** — token ngay trước khi package được sinh ra.
2. Tính 2 **centroid** cố định: trung bình hidden state của các ví dụ `label=1` (real) và `label=0` (hallucinated).
3. Khởi tạo 1 vector `tsv` (ban đầu = 0), rồi dùng gradient descent để học vector này sao cho: `representation + tsv` càng giống centroid đúng nhãn càng tốt (đo bằng cosine similarity + cross-entropy).
4. Lặp lại cho từng layer trong khoảng `--layer_start`–`--layer_end`, chọn layer có AUROC cao nhất trên tập eval giữ lại.

**Lưu ý:** đây là phương pháp học bằng gradient descent (contrastive, so với centroid cố định), **khác** với "vector hiệu số trung bình" (mean-difference vector) mô tả trong đề xuất ban đầu — cùng tinh thần nhưng cơ chế khác, chi phí huấn luyện (setup cost, cho RQ3) cũng khác.

### 3.2. Injection (`llm_layers.py` + `eval_variant_v2.py`)

1. `add_tsv_layers` thay thế **đúng 1 decoder layer** (layer tốt nhất từ bước training) bằng `LlamaDecoderLayerWrapper` — layer này chạy attention + MLP bình thường, rồi cộng thêm vector đã học vào cuối, **chỉ tại các vị trí được chỉ định**.
2. Ở lượt forward **đầu tiên** (xử lý toàn bộ prompt), vector được cộng vào đúng 1 vị trí — token cuối cùng của prompt (boundary).
3. Ở các lượt forward sau (mỗi lượt sinh 1 token mới, dùng KV cache), **không** cộng thêm gì nữa — `steering_positions=[]` báo cho `TSVLayer` bỏ qua injection.
4. Hiệu ứng của lần cộng đầu tiên vẫn lan truyền tới các token sinh sau đó — vì hidden state đã bị thay đổi nằm trong KV cache, các token sau vẫn "nhìn thấy" nó qua cơ chế attention.

**Điểm cần lưu ý cho RQ4:** vì chỉ tiêm 1 lần duy nhất ở đầu, hiệu ứng có thể yếu dần nếu tên package xuất hiện rất xa so với đầu prompt (ví dụ giữa 1 hàm dài).

---

## 4. Định dạng dữ liệu

### 4.1. Input cho `tsv_main.py`

Chấp nhận 2 định dạng, tự nhận diện qua đuôi file:

**a) JSON/JSONL** (định dạng gốc):
```json
{"prompt": "...", "label": 1, "package_generation_position": 42}
```
- `label`: 0 = hallucinated, 1 = real
- Vị trí boundary: `package_generation_position` (ưu tiên), hoặc `package_position`, `package_positions`

**b) CSV kết quả thật** (`LLM_<tag>_results.csv` — được `eval_variant_v2.py` sinh ra, hoặc dữ liệu có sẵn từ trước): tự động chuyển đổi qua `_records_from_results_csv`.
- **Chỉ dùng cột `pip_valid`/`pip_hallucinated`** (tên package trích trực tiếp từ code sinh ra) — **không dùng được** `Test_1`/`Test_2` vì đó là danh sách tên đã trích sẵn, không có văn bản gốc để định vị lại boundary position.
- Với mỗi tên trong `pip_valid`/`pip_hallucinated`, code tìm lại vị trí ký tự của tên đó trong cột `Answers` (khớp `import X`, `from X import`, hoặc `pip install X`), cắt văn bản đến ngay trước tên đó, ghép với `Prompts` làm thành 1 ví dụ huấn luyện.
- **Gộp nhiều file cùng lúc:** `--data_path file1.csv file2.csv ...` — tự động merge, kiểm tra cả 2 nhãn (0 và 1) phải có mặt trong tập **gộp** (không bắt buộc mỗi file riêng lẻ phải đủ 2 nhãn).

### 4.2. Output của `tsv_main.py`

Trong `--output_dir`:
- `steering_vector.pt`: `{"best_layer": int, "vectors": {layer: vector, ...}}`
- `metadata.json`: model, hidden_size, num_layers, best_layer, best_auroc
- `layer_scores.json`: AUROC từng layer đã thử

### 4.3. Input/Output của `eval_variant_v2.py`

- Input prompts: file `.jsonl` (mỗi dòng `{"prompt": ...}`), hoặc trực tiếp file CSV kết quả (lấy cột `Prompts`).
- Output: `eval_runs/<tag>/LLM_<tag>_results.csv` — **đúng 11 cột** giống hệt dữ liệu gốc (`Prompts, Answers, Test_1, Test_2, valid_1, hallucinated_1, valid_2, hallucinated_2, pip, pip_valid, pip_hallucinated`), cột dạng list ghi theo `str(list)` (`"['numpy']"`) để tương thích trực tiếp với dữ liệu gốc.
- Output tóm tắt: `LLM_<tag>_summary.json` — PHR, RHR, số tên hallucinated duy nhất, tính riêng cho 3 "surface": `test_1` (hỏi "packages nào cần để chạy code này"), `test_2` (hỏi "packages nào hữu ích cho bài toán này"), `pip` (quét trực tiếp `import` trong code).

**Định nghĩa 2 chỉ số:**
- **PHR** (Package Hallucination Rate) = số tên hallucinated / tổng số tên được đề cập
- **RHR** (Response Hallucination Rate) = % số dòng/response có ≥1 tên hallucinated

---

## 5. Cách chạy

### 5.1. Local (Windows, qua `run_tsv_main.sh`/`.ps1` và `run_eval_variant.sh`)

```bash
# Bước 1: huấn luyện vector (gộp nhiều file nếu có)
python tsv_main.py --model_path <model> --data_path LLM_LY_results.csv LLM_AT_results.csv \
  --output_dir tsv_output --batch_size 4 --num_epochs 5 --layer_start 4 --layer_end 8

# Bước 2: đánh giá bằng steering
python eval_variant_v2.py --method steering --model_path <model> --tag <ten_lan_chay> \
  --n_prompts <so_luong> --steering_vector_path tsv_output/steering_vector.pt \
  --prompts_file LLM_LY_results.csv
```

Hoặc dùng script tiện lợi (tự activate `.venv`, tự kiểm tra file tồn tại): `run_tsv_main.sh`, `run_eval_variant.sh`.

### 5.2. Google Colab

Notebook `run_tsv_main_colab.ipynb` — chạy trọn cả 2 bước, có mount Google Drive, hiển thị AUROC và PHR/RHR ngay trong notebook.


