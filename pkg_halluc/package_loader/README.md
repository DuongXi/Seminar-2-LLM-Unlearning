# Package Hallucination Unlearning Dataset & DataLoader

Module cung cấp:
1. **Llama 3.2**
2. **Qwen 2.5 Coder**
3. **DeepSeek-Coder**

---

## 1. Cấu trúc Prompt & Phân loại Unlearning

### **Mode 1 (Code $\rightarrow$ Required Packages)**
- **System Prompt:**
  ```text
  You are a coding assistant that determines Python packages necessary to execute code. Respond with only a list of Python packages, separated by commas and no additional text or formatting. Your response must begin with the name of a Python package.
  ```
- **User Prompt:**
  ```text
  Which Python packages are required to run this code: <code_snippet>
  ```
- **Dữ liệu phân loại:** `valid_1` và `hallucinated_1`.

### **Mode 2 (Problem Description $\rightarrow$ Helpful Packages)**
- **System Prompt:**
  ```text
  You are a coding assistant that recommends Python packages that would be helpful to solve given problems. Respond with only a list of Python packages, separated by commas and no additional text or formatting. Your response must begin with the name of a Python package.
  ```
- **User Prompt:**
  ```text
  What Python packages would be useful in solving the following coding problem: <problem_prompt>
  ```
- **Dữ liệu phân loại:** `valid_2` và `hallucinated_2`.

---

## 2. Đặc thù Chat Template & Tokenizer của 3 dòng mô hình

| Dòng mô hình | Chat Template Format | Pad Token | EOS Token |
| :--- | :--- | :--- | :--- |
| **Llama 3 / 3.3** | `<\|start_header_id\|>...<\|eot_id\|>` | `<\|eot_id\|>` | `<\|eot_id\|>` |
| **Qwen 2.5 Coder** | ChatML: `<\|im_start\|>...<\|im_end\|>` | `<\|endoftext\|>` | `<\|im_end\|>` |
| **DeepSeekCoder** | `### Instruction:\n...### Response:` | `<\|EOT\|>` | `<\|EOT\|>` |

Hàm `setup_tokenizer(...)` tự động:
- Nhận diện model family từ tên hoặc đường dẫn (`"auto"`).
- Cấu hình đúng `pad_token_id` và `eos_token_id` để tránh lỗi khi batching / padding.
- Tự động bổ sung fallback chat template nếu load các bản Base model.

---

## 3. Hướng dẫn sử dụng

### 3.1. Khởi tạo nhanh cho Qwen 2.5 Coder 3B

```python
from unlearning import get_unlearning_dataloaders

forget_loader, retain_loader, combined_loader = get_unlearning_dataloaders(
    data_source="Llama3_3_Python",
    tokenizer="Qwen/Qwen2.5-Coder-3B-Instruct", # Tự động load và setup
    model_family="qwen",                       # Hoặc 'auto'
    batch_size=4,
    query_modes=[1, 2],
    return_format="pointwise",                 # Prompt mask -100, loss chỉ tính trên completion
    shuffle=True
)
```

### 3.2. Khởi tạo nhanh cho DeepSeekCoder 1.3B

```python
from unlearning import get_unlearning_dataloaders

forget_loader, retain_loader, combined_loader = get_unlearning_dataloaders(
    data_source="Llama3_3_Python",
    tokenizer="deepseek-ai/deepseek-coder-1.3b-instruct",
    model_family="deepseek",
    batch_size=4,
    query_modes=[1, 2],
    return_format="pointwise",
    shuffle=True
)
```

### 3.3. Sử dụng cho DPO Unlearning (Llama-3, Qwen, DeepSeek)

```python
from unlearning import PackageUnlearningDataset, setup_tokenizer

tokenizer = setup_tokenizer("Qwen/Qwen2.5-Coder-3B-Instruct")

dpo_dataset = PackageUnlearningDataset(
    data_source="Llama3_3_Python",
    split_type="forget",
    query_modes=[1, 2],
    tokenizer=tokenizer,
    return_format="dpo"
)

sample = dpo_dataset[0]
print("Prompt:", sample["prompt"])
print("Chosen:", sample["chosen"])
print("Rejected:", sample["rejected"])
```

## 4. Sử dụng DataLoader tương thích TSV

`unlearning/tsv_dataloader.py` cung cấp DataLoader chuẩn hóa theo phương pháp **TSV (Task-Specific Vectors / Truth Steering Vectors)**:
- **Nhãn nhị phân:** `0 = Hallucination` (chứa ảo giác), `1 = Truthful / Valid` (hợp lệ).
- **Chia tập chuẩn TSV:**
  - `exemplars`: Tập mẫu ít shot cân bằng (ví dụ: 32 mẫu gồm 16 ảo giác và 16 đúng).
  - `wild` (`train`): Tập dữ liệu chưa gán nhãn ngoài tự nhiên để gom cụm Sinkhorn-Knopp.
  - `test`: Tập kiểm thử đánh giá AUROC phát hiện ảo giác.
- **Tương thích hoàn toàn:** Trả về cấu trúc `prompts = [test_prompts, train_prompts, exemplar_prompts]` và `labels = [gt_label_test, gt_label_wild, gt_label_exemplar]` để cắm thẳng vào hàm `train_model(model, optimizer, device, prompts, labels, args)` trong `tsv-main/tsv_main.py`.

```python
from unlearning import get_tsv_data_and_loaders

# Load dữ liệu chuẩn hóa cho TSV
tsv_data = get_tsv_data_and_loaders(
    data_source="Llama3_3_Python",
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",  # hoặc Qwen, DeepSeek
    model_family="llama3",
    num_exemplars=32,      # 16 Hallucination (0) + 16 Truthful (1)
    wild_ratio=0.75,       # 75% cho wild train, 25% cho test
    batch_size=32,
    max_length=512,
    balanced_exemplars=True,
    seed=42,
)

# Cắm thẳng vào tsv_main.py
prompts = tsv_data["prompts"]   # [test_prompts, train_prompts, exemplar_prompts]
labels = tsv_data["labels"]     # [test_labels, train_labels, exemplar_labels]

# Hoặc dùng dưới dạng PyTorch DataLoader:
train_loader = tsv_data["train_loader"]
for batch in train_loader:
    prompts = batch["prompts"]         # Shape: [batch_size, 1, seq_len]
    labels = batch["labels"]           # Shape: [batch_size]
    mask = batch["attention_mask"]     # Shape: [batch_size, 1, seq_len]
```

---

## 5. Chạy script kiểm thử

```bash
# Package Attribution Unlearning (Modes 1, 2: SFT, DPO, GA):
python test_unlearning_dataloader.py

# TSV DataLoader (Exemplar, Wild, Test, Collate):
python test_tsv_dataloader.py
```