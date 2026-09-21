# Dataset & DataLoader

Thư mục `package_loader` chứa pipeline data preprocessing, chia tách tập dữ liệu, chuẩn hóa file, và mã hóa token-level **Tri-Mask** phục vụ các phương pháp Unlearning.


## 1. Xử lý & Phân loại Dữ liệu

Dữ liệu unlearning được trích xuất từ 4 benchmark raw CSVs (`LLM_AT_results.csv`, `LLM_LY_results.csv`, `SO_AT_results.csv`, `SO_LY_results.csv`) tương ứng với 2 ngữ cảnh truy vấn:

### **Mode 1 (Code $\rightarrow$ Required Packages)**

**Phân loại nhãn:** Cột `valid_1` (thư viện hợp lệ) và `hallucinated_1` (thư viện ảo giác do model tự bịa).

### **Mode 2 (Problem Description $\rightarrow$ Helpful Packages)**

**Phân loại nhãn:** Cột `valid_2` (thư viện hợp lệ) và `hallucinated_2` (thư viện ảo giác).

### **Phân loại Bản ghi  **
- **`forget`** chứa ít nhất 1 package ảo giác (`len(hallucinated) > 0`).
  - `chosen`: các package hợp lệ.
  - `rejected`: các package mà model đã sinh ra chứa ảo giác.
- **`retain`** chỉ chứa package hợp lệ để huấn luyện giữ lại kiến thức gốc của mô hình.

Sau đó tập hợp lại và chỉ lấy tập forget chứa ít nhất 1 package ảo và retain có chứa ít nhất 1 package hợp lệ, loại bỏ mẫu vượt quá chiều dài max_length = 2048 cho tri_mask.

## 2. Phân chia Dữ liệu: Train - Val - Test

### 2.1. Phân chia Train - Test (`train_test_hallu_split.py`)
  1. Trích xuất tất cả unique prompt gây ra ảo giác từ 4 file CSV.
  2. Lấy mẫu cố định mặc định 100 prompt/file $\times$ 4 file = 400 prompts.
  3. Chia theo tỷ lệ 0.9: 90% cho Train (360 prompts) và 10% cho Test (40 prompts).
  4. Lọc lại các dòng trong raw CSV để tạo ra:
     - `train_csvs/`: `LLM_AT_results_train.csv`, `LLM_LY_results_train.csv`, ...
     - `test_csvs/`: `LLM_AT_results_test.csv`, `LLM_LY_results_test.csv`, ...
     - `train_prompts.jsonl` và `test_prompts.jsonl`.

### 2.2. Phân chia Train - Val (Hàm `split_records_by_prompt`)

  - **Chỉ trích xuất tập Val từ các mẫu Retain** (`val_ratio = 0.1` $\rightarrow$ 10% retain làm val).
  - Giữ lại **100% tập Forget cho quá trình Train** (không chia nhỏ tập Forget vì mẫu ảo giác cần để tối ưu hàm mất mát unlearning).
  - Gom nhóm theo tiền tố Prompt ID để đảm bảo các truy vấn Mode 1 và Mode 2 cùng thuộc Train hoặc cùng thuộc Val.

## 3. Cấu trúc và Cách tạo File Master

File Master (`master_train.json`, `master_val.json`, `master_test.json`) tổng hợp và chuẩn hóa các file CSV.

### 3.1. Cấu trúc một bản ghi (`UnlearningRecord`)
```json
{
  "sample_id": "LLM_AT_results_train.csv_m2_idx12",
  "split_type": "forget",
  "mode": 2,
  "system_prompt": "You are a coding assistant that recommends Python packages...",
  "user_prompt": "What Python packages would be useful in solving the following coding problem: Generate Python code that connects to an AWS cloud-based pandas cluster...",
  "completion": "boto3, pandas, aws-cluster-manager, s3fs",
  "chosen": "boto3, pandas, s3fs",
  "rejected": "boto3, pandas, aws-cluster-manager, s3fs",
  "valid_packages": ["boto3", "pandas", "s3fs"],
  "hallucinated_packages": ["aws-cluster-manager"]
}
```

Khi file đã được tạo, `PackageUnlearningDataset` sẽ tự động nhận diện thông qua hàm `is_preprocessed(file_path)` và đọc trực tiếp trong vòng vài mili-giây mà không cần parse lại CSV.


## 4. Cơ chế và Cách tạo Token-level Tri-Mask

### 4.1. Quy ước các giá trị Tri-Mask

| Giá trị `tri_mask` | Loại Token | Nhãn `labels` | Hàm Loss tác động |
| :---: | :--- | :---: | :--- |
| **`0`** | Prompt tokens, System template, Padding, Ký tự phân cách (dấu phẩy, khoảng trắng) | `-100` | **Bị bỏ qua**: Không tính gradient |
| **`1`** | **Retain tokens**: Các token cấu thành nên `valid_packages` + token kết thúc chuỗi (`<\|eot_id\|>` / `EOS`) | Token ID gốc | **Cross-Entropy Loss** ($\mathcal{L}_{retain}$): Củng cố kiến thức đúng |
| **`2`** | **Forget tokens**: Các token cấu thành nên `hallucinated_packages` | Token ID gốc | **GA / NPO Loss** ($\mathcal{L}_{forget}$): Hạ xác suất sinh ảo giác |

### 4.2. Thuật toán gán nhãn Tri-Mask (`generate_tri_mask.py`)
1. **Tokenize Prompt:** Dùng Chat Template của mô hình (`apply_chat_template`) với `add_generation_prompt=True`. Toàn bộ token này nhận nhãn `tri_mask = 0`.
2. **Tokenize Response:** Tokenize đoạn response với `return_offsets_mapping=True` để biết chính xác vị trí ký tự `(char_start, char_end)` của từng sub-token trong chuỗi gốc.
3. **Ánh xạ Token Offset:**
   - Nếu khoảng bù ký tự của token giao nhau với vị trí package ảo giác $\rightarrow$ gán `tri_mask = 2`.
   - Nếu giao nhau với vị trí package hợp lệ $\rightarrow$ gán `tri_mask = 1`.
   - Các token còn lại (dấu phẩy, khoảng trắng) $\rightarrow$ gán `tri_mask = 0`.
4. **Gán nhãn EOS Token:** Token kết thúc (`eos_token_id`) được bổ sung vào cuối và nhận `tri_mask = 1` để khuyến khích mô hình học cách dừng sớm thay vì tiếp tục bịa package ảo giác.



Kết quả sinh ra 3 file JSONL đã được tokenized sẵn:
- `npo_retain_tok_*.jsonl`: Chứa các mẫu retain (chỉ gồm token `0` và `1`).
- `npo_forget_tok_*.jsonl`: Chứa các mẫu forget (gồm cả token `0`, `1`, và `2`).
- `npo_val_retain_tok_*.jsonl`: Chứa các mẫu validation retain dùng cho early stopping.


---

## 6. Cấu trúc File

```
data/<Model_Name>/
├── FINAL_RESULTS.csv                 # Kết quả gốc từ benchmark sinh mã ban đầu
├── LLM_AT_results.csv                # Raw CSV benchmark: LLM All Time
├── LLM_LY_results.csv                # Raw CSV benchmark: LLM Last Year
├── SO_AT_results.csv                 # Raw CSV benchmark: Stack Overflow All Time
├── SO_LY_results.csv                 # Raw CSV benchmark: Stack Overflow Last Year
├── train_test_split/                 # [Bước 1] Sinh ra sau train_test_hallu_split.py
│   ├── split_metadata.json           # Thông tin cấu hình và thống kê phân chia
│   ├── train_prompts.jsonl           # 360 unique prompt train
│   ├── test_prompts.jsonl            # 40 unique prompt test cho benchmark eval
│   ├── train_csvs/                   # 4 CSVs chứa dòng train tương ứng
│   ├── test_csvs/                    # 4 CSVs chứa dòng test tương ứng
│   ├── master_train.json             # [Bước 2] Toàn bộ bản ghi train chuẩn hóa
│   ├── master_val.json               # [Bước 2] Bản ghi held-out retain val
│   └── master_test.json              # [Bước 2] Bản ghi test dùng đánh giá unlearning
└── tri_mask/                         # [Bước 3] Sinh ra sau generate_tri_mask.py
    ├── npo_retain_tok_*.jsonl        # Tokenized retain records
    ├── npo_forget_tok_*.jsonl        # Tokenized forget records
    └── npo_val_retain_tok_*.jsonl    # Tokenized validation retain records
```