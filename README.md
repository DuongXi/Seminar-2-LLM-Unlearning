
# pkg-halluc: Giảm thiểu Package Hallucination

| Method | Loại | Nguồn |
| --- | --- | --- |
| **Base** | model gốc| -- |
| **GA** (Gradient Ascent) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | 
| **NPO** (Negative Preference Optimization) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | 
| **GA-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh, không tri-mask | 
| **NPO-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh, không tri-mask | 

## Cấu trúc thư mục

```
├── configs/            # File config JSON --config cho scripts/*.sh, xem "Cấu hình"
├── data/               # dataset PackageHallucination cho Python + data/eval/
├── data_gen/           # script gen full dataset PackageHallucination
├── notebooks/          # notebook Kaggle
├── scripts/            
│   ├── common.sh              # thiết lập dùng chung
│   ├── quickstart.sh           # chạy full pipeline
│   ├── download_model.sh        # tải base model + sanity-check
│   ├── build_data.sh             # dựng dữ liệu tri-mask cho GA/NPO
│   ├── train_ga.sh, train_npo.sh              # train GA/NPO (qua pkg_halluc/training/au/train_au.py)
│   ├── train_ga_plain.sh, train_npo_plain.sh  # train GA-plain/NPO-plain
│   ├── train_au_common.sh, train_plain_common.sh  # phần dùng chung
│   ├── eval.sh                                 # eval 1 checkpoint bất kỳ (base/ga/npo/ga_plain/npo_plain)
│   └── report.sh                                # gộp eval_runs/* thành bảng kết quả cuối
└── pkg_halluc/         # code Python, vai trò từng file xem pkg_halluc/README.md
    ├── common/             # dùng chung: preset model, load model, tải model, đọc config, tiện ích lấy từ AU
    ├── package_loader/     # dataset/dataloader, dựng dữ liệu tri-mask (tri-mask, prompt, tokenizer setup...)
    ├── training/           # training 4 method
    ├── evaluation/         # eval 1 checkpoint + report; generation/ (sinh code, hỏi package), detection/
```

## Yêu cầu trước khi chạy

- Python >= 3.10
- GPU CUDA

## Cài đặt

```bash
git clone https://github.com/DuongXi/Seminar-2-LLM-Unlearning
cd Seminar-2-LLM-Unlearning

pip install -r requirements.txt 
```

## Cấu hình

Mỗi bash script trong `scripts/` nhận tham số qua flag dòng lệnh, với giá
trị mặc định sẵn ngay trong script (xem `--help` của từng script). Muốn dung san tham so tu cac file
config co san truyền `--config <file>`:

```bash
bash scripts/train_ga.sh --config configs/default.json
bash scripts/eval.sh --config configs/default.json --tag ga --model-path ...
```

`--config` nạp giá trị từ file JSON làm **mặc định**; flag nào gõ thêm sau
đó trên dòng lệnh vẫn **overwrite**,không cần sửa file JSON để chạy thử 1 giá trị khác:

```bash
bash scripts/train_ga.sh --config configs/default.json --lr 2e-5
```

| File | Dùng cho |
| --- | --- |
| `configs/default.json` | Qwen2.5-Coder-1.5B |
| `configs/smoke_test.json` | chạy thử nhanh (25 prompt đánh giá, dữ liệu retain/forget mọi method giới hạn 40 dòng/split|
| `configs/qwen2.5-coder-1.5b.json`, `configs/qwen2.5-coder-3b.json`, `configs/qwen2.5-1.5b.json` | preset Qwen 2.5 Coder (1.5B / 3B) |
| `configs/llama3.2-1b.json`, `configs/llama3.2-3b.json` | preset Llama 3.2 (1B / 3B) |
| `configs/deepseek-coder-1.3b.json` | preset DeepSeek Coder 1.3B |


## Hướng dẫn chạy train
### 0. Xem hướng dẫn chi tiết
bash scripts/train_ga.sh --help
bash scripts/train_ga_plain.sh --help
bash scripts/train_npo.sh --help
bash scripts/train_npo_plain.sh --help

### 1. Lệnh cơ bản

```bash
bash scripts/train_ga.sh --model deepseek-coder-1.3b
```

Script tự đổi tên preset thành id Hugging Face, kiểm tra base model trong `models/`, train xong thi ghi kết quả vào `checkpoints/`

`--model` nhận tên preset (ví dụ `qwen-0.5b`, `qwen-1.5b`, `qwen-3b`, `llama3.2-1b`, `deepseek-coder-1.3b`) hoặc id Hugging Face đầy đủ. Tên không nằm trong bảng preset (ví dụ chỉ gõ `qwen`) sẽ báo không tìm thấy model. Bỏ `--model` thì dùng `Qwen/Qwen2.5-Coder-0.5B-Instruct`.

### 2. Giá trị mặc định khi chỉ truyền `--model`

Không có `--config` thì không đọc file JSON nào, toàn bộ tham số lấy từ giá trị mặc định:

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--lr` | `1e-5` | Learning rate |
| `--epochs` | `3` | Số epoch|
| `--seed` | `42` | Seed|
| `--dtype` | `auto` | Kiểu số của trọng số: bfloat16 |
| `--use-lora` | tắt | Mặc định là **full fine-tune** (train toàn bộ trọng số) |
| Early stopping | bật | |
| Batch | 1 × 16 | Mỗi step gom 16 mẫu (batch 1, gradient accumulation 16) |

### 3. Chạy bằng file config

```bash
bash scripts/train_ga.sh --config configs/deepseek-coder-1.3b.json
bash scripts/train_ga.sh --config configs/deepseek-coder-1.3b.json --epochs 5   # flag thêm sẽ ghi đè giá trị trong file json
```

Lưu ý: các file `configs/*.json` dang bật sẵn LoRA (`use_lora: true`), vì nếu dùng kaggle free, dung lượng của output sẽ vuợt quá mức kaggle cho phép, gây lỗi out of memory, còn khi không có `--config` mặc định là full fine-tune

### 4. Lệnh full, không phụ thuộc file JSON

Qua script, ghi rõ mọi flag (giá trị dưới đây bằng đúng mặc định):
```bash
bash scripts/train_ga.sh \
  --model deepseek-coder-1.3b \
  --model-path models/deepseek-ai/deepseek-coder-1.3b-instruct \
  --save-tag ga \
  --out-dir checkpoints/deepseek-coder-1.3b-instruct_ga \
  --lr 1e-5 \
  --epochs 3 \
  --seed 42 \
  --dtype auto \
  --val-ratio 0.1 \
  --eval-steps 25 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.0
```

Co thể thêm `--use-lora --lora-rank 16` để train LoRA, `--disable-early-stopping` để tắt early stopping


## Cách chạy pipeline

```bash
bash scripts/download_model.sh --model Qwen/Qwen2.5-Coder-0.5B-Instruct   # tải model gốc, sanity-check
bash scripts/build_data.sh --model Qwen/Qwen2.5-Coder-0.5B-Instruct        # dựng dữ liệu tri-mask cho GA/NPO

bash scripts/train_ga.sh        --model Qwen/Qwen2.5-Coder-0.5B-Instruct
bash scripts/train_npo.sh       --model Qwen/Qwen2.5-Coder-0.5B-Instruct
bash scripts/train_ga_plain.sh  --model Qwen/Qwen2.5-Coder-0.5B-Instruct
bash scripts/train_npo_plain.sh --model Qwen/Qwen2.5-Coder-0.5B-Instruct

bash scripts/eval.sh --tag base      --model-path models/Qwen/Qwen2.5-Coder-0.5B-Instruct
bash scripts/eval.sh --tag ga        --model-path checkpoints/Qwen2.5-Coder-0.5B-Instruct_ga
bash scripts/eval.sh --tag npo       --model-path checkpoints/Qwen2.5-Coder-0.5B-Instruct_npo
bash scripts/eval.sh --tag ga_plain  --model-path checkpoints/Qwen2.5-Coder-0.5B-Instruct_ga_plain
bash scripts/eval.sh --tag npo_plain --model-path checkpoints/Qwen2.5-Coder-0.5B-Instruct_npo_plain

bash scripts/report.sh   # gộp eval_runs/* thành bảng kết quả cuối + lưu CSV
```

Hoặc chạy hết 1 lượt bằng:

```bash
bash scripts/quickstart.sh --model Qwen/Qwen2.5-Coder-0.5B-Instruct
```