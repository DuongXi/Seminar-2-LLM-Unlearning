
# pkg-halluc: Giảm thiểu Package Hallucination

| Method | Loại | Nguồn |
| --- | --- | --- |
| **Base** | tham chiếu (model gốc, chưa đổi gì) | -- |
| **GA** (Gradient Ascent) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | lấy từ code của paper *Adaptive Unlearning* (AU) |
| **NPO** (Negative Preference Optimization) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | lấy từ code của AU |
| **GA-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh,không tri-mask | code riêng|
| **NPO-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh, không tri-mask | code riêng |
| **Representation Steering** | *chưa cài đặt* | |


## Cấu trúc thư mục

```
├── configs/            # File config JSON
├── data/               # dataset PackageHallucination cho Python
├── data_gen/           # script gen full dataset PackageHallucination
├── notebooks/          # notebook Kaggle
├── scripts/            # quickstart.sh -- script de chạy full pipeline
├── steering_ref/       # code TSV lưu trữ (nguyên bản tu paper) 
├── pkg_halluc/         # code pipeline
│   ├── paths.py            # nhận diện môi trường Kaggle-vs-local (run on kaggle or local)
│   ├── config.py           # đọc config
│   ├── cli.py              # CLI `pkg_halluc`
│   ├── model_setup.py      # tải model gốc
│   ├── package_loader/     # data_loader
│   ├── deps/               # tải + vá + copy code của paper AU
│   ├── tok_data/           # dựng dữ liệu train tĩnh cho GA/NPO
│   ├── scripts/            # train_plain.py -- train GA-plain/NPO-plain, không qua AU
│   ├── methods/            # định nghĩa từng method (base/ga/npo/ga_plain/npo_plain)
│   ├── eval/               # đánh giá dùng chung cho từng method
│   ├── report/             # gộp eval_runs/* thành bảng kết quả cuối
│   ├── templates/          # script support chay
│   └── utils/              # chạy subprocess dùng chung cho mọi bước pipeline
└── README.md
```

## Yêu cầu trước khi chạy

- Python >= 3.10
- GPU CUDA

## Cài đặt

```bash
git clone https://github.com/DuongXi/Seminar-2-LLM-Unlearning



pip install -e .          # cài pkg_halluc + các dependency đã pin trong pyproject.toml
```

Sau bước này có lệnh `pkg_halluc`

## Lấy code của paper AU

```bash
# Cách A: đã có sẵn file zip AU trên máy 
pkg_halluc fetch-deps --au-src /path/to/Adaptive-Unlearning-952E.zip

# Cách B: trên Kaggle, gắn repo AU làm Dataset qua "Add Input" --
# không cần truyền --au-src, tự động tìm thấy
pkg_halluc fetch-deps
```

## Cấu hình

| File | Dùng để làm gì |
| --- | --- |
| `configs/default.json` | Qwen2.5-Coder-1.5B, quy mô đánh giá thực tế (150 prompt) |
| `configs/smoke_test.json` | nhỏ (25 prompt đánh giá, dữ liệu retain/forget của mọi method giới hạn 40 dòng/split) dùng để kiểm tra môi trường|
| `configs/qwen2.5-coder-1.5b.json`, `configs/qwen2.5-coder-3b.json`, `configs/qwen2.5-1.5b.json` | preset Qwen 2.5 Coder (1.5B / 3B) |
| `configs/llama3.2-1b.json`, `configs/llama3.2-3b.json` | preset Llama 3.2 (1B / 3B) |
| `configs/deepseek-coder-1.3b.json` | preset DeepSeek Coder 1.3B |

Các field dưới đây lấy theo giá trị mặc định gốc trong `DEFAULT_CONFIG` (mọi
preset ở trên đều override `use_lora: true` và `eval.batch_size` thấp hơn --
4 đến 8 tùy size model -- đây chỉ là giá trị fallback khi config không ghi):

```jsonc
{
  "model_name": "qwen2.5-0.5b",      // tên preset (xem MODEL_PRESETS trong config.py) hoặc id HF đầy đủ
  "seed": 42,
  "dtype": "auto",                    // "auto" | "bfloat16" | "float16" -- auto tự chọn bf16 nếu GPU hỗ trợ
  "data": {
    "max_train_samples_per_split": null // set null de chay full data, set số nhỏ để chạy thử nhanh
  },
  "eval": {
    "n_eval_prompts": 150,
    "batch_size": 16,                 // gặp OOM lúc eval thì hạ số này
    "package_modes": [1, 2]          
  },
  "methods": {
    "base": { "enabled": true },
    "ga":   { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "npo":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "ga_plain":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "npo_plain": { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "steering":    { "enabled": false } 
  },
  "deps": {
    "adaptive_unlearning": { "source": null }   // set chỗ này, hoặc dùng --au-src de lay code AU
  }
}
```

`use_lora: true` -- train LoRA adapter (qua PEFT) thay vì fine-tune toàn bộ
trọng số: chỉ adapter nhỏ cần gradient/optimizer, base model đóng băng. Dùng
khi model lớn không đủ VRAM để full fine-tune (vd T4 16GB trên
Kaggle).

## Cách chạy

Mỗi bước là 1 subcommand riêng để có thể dừng và chạy lại giữa chừng

```
pkg_halluc fetch-deps            # tải + vá code AU
pkg_halluc download-model        # tải model gốc, kiểm tra nhanh việc sinh text
pkg_halluc build-data            # dựng dữ liệu prompt + response để train GA/NPO
pkg_halluc train --method all    # train all method (co the thay all = ga/npo/ga_plain/npo_plain neu muon train tung method)
pkg_halluc evaluate --method all # eval hallucination (co the thay all = ga/npo/ga_plain/npo_plain neu muon eval tung method)
pkg_halluc report                # gộp eval_runs/* thành bảng kết quả cuối + lưu CSV

pkg_halluc run-all               # chạy hết các bước trên theo thứ tự
```

Mọi subcommand đều nhận `--config`, `--work-dir`, `--model`, `--seed`; xem
`pkg_halluc <subcommand> --help` để biết thêm

### Chạy local

```bash
pkg_halluc fetch-deps --config configs/smoke_test.json --au-src /path/to/Adaptive-Unlearning-952E.zip
pkg_halluc download-model --config configs/smoke_test.json
pkg_halluc build-data --config configs/smoke_test.json
pkg_halluc train --config configs/smoke_test.json --method all
pkg_halluc evaluate --config configs/smoke_test.json --method all
pkg_halluc report --config configs/smoke_test.json
```
