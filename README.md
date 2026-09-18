
# pkg-halluc: Giảm thiểu Package Hallucination

| Method | Loại | Nguồn |
| --- | --- | --- |
| **Base** | tham chiếu (model gốc, chưa đổi gì) | -- |
| **GA** (Gradient Ascent) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | lấy từ code của paper *Adaptive Unlearning* (AU) |
| **NPO** (Negative Preference Optimization) | fine-tune toàn bộ trọng số, dữ liệu tĩnh, mask theo token (tri-mask) | lấy từ code của AU |
| **GA-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh, không tri-mask | code riêng |
| **NPO-plain** | fine-tune toàn bộ trọng số, cùng dữ liệu tĩnh, không tri-mask | code riêng |

## Early stopping (`ga_plain` / `npo_plain`)

**Cách hoạt động:** một phần (`val_ratio`) của tập **retain** được tách ra
làm `eval_dataset`, Trainer eval định kỳ mỗi `eval_steps`, và
`EarlyStoppingCallback` sẽ dừng train nếu `eval_loss` không cải thiện sau
`early_stopping_patience` lần eval liên tiếp (`load_best_model_at_end=True`
nên checkpoint cuối cùng luôn là checkpoint tốt nhất, không phải checkpoint
tại bước dừng).

**Vì sao chỉ tách từ tập retain, không tách từ forget:** loss tổng của
GA/NPO-plain là `lambda_retain * L_retain + lambda_forget * L_forget`, trong
đó `L_forget` là *âm* của cross-entropy trên forget-set (ascent target) ->
không có đáy, càng train càng "cải thiện" vô hạn. Nếu early stop dựa trên
loss tổng (gồm cả `L_forget`) thì tiêu chí này gần như không bao giờ
plateau. `GAPlainTrainer`/`NPOPlainTrainer.compute_loss` (xem
`pkg_halluc/training/plain/plain_trainers.py`) đã tự động cho `L_forget = 0` khi một batch
không có dòng nào thuộc split `"forget"` nên chỉ cần trỏ `eval_dataset` vào
một tập con **toàn retain**, `eval_loss` mà Trainer tự tính ra sẽ tự nhiên
chỉ còn là `lambda_retain * L_retain`: một tín hiệu bounded, phản ánh đúng
việc model có đang giữ được utility (retain) hay không, mà không cần sửa
`compute_loss`.

## Cấu trúc thư mục

```
├── configs/            # File config JSON -- --config cho scripts/*.sh, xem "Cấu hình"
├── data/               # dataset PackageHallucination cho Python + data/eval/ (từ AU, xem "Code của paper AU")
├── data_gen/           # script gen full dataset PackageHallucination
├── notebooks/          # notebook Kaggle
├── scripts/            # TOÀN BỘ entrypoint đều ở đây -- xem "Cách chạy"
│   ├── common.sh              # thiết lập dùng chung (path, biến môi trường GPU) -- source, không tự chạy
│   ├── quickstart.sh           # chạy full pipeline, gọi lần lượt các script bên dưới
│   ├── download_model.sh        # tải base model + sanity-check
│   ├── build_data.sh             # dựng dữ liệu tri-mask tĩnh cho GA/NPO
│   ├── train_ga.sh, train_npo.sh              # train GA/NPO (qua pkg_halluc/training/au/train_au.py)
│   ├── train_ga_plain.sh, train_npo_plain.sh  # train GA-plain/NPO-plain (không qua AU)
│   ├── train_au_common.sh, train_plain_common.sh  # phần thân dùng chung, không tự chạy trực tiếp
│   ├── eval.sh                                 # eval 1 checkpoint bất kỳ (base/ga/npo/ga_plain/npo_plain)
│   └── report.sh                                # gộp eval_runs/* thành bảng kết quả cuối
└── pkg_halluc/         # code Python mà scripts/*.sh gọi tới (chạy bằng `python -m`), vai trò từng file xem pkg_halluc/README.md
    ├── common/             # dùng chung: preset model, load model, tải model, đọc config, tiện ích lấy từ AU
    ├── package_loader/     # dataset/dataloader, dựng dữ liệu tri-mask (tri-mask, prompt, tokenizer setup...)
    ├── training/           # au/ (GA, NPO tri-mask lấy từ AU) và plain/ (GA-plain, NPO-plain, không qua AU)
    ├── evaluation/         # eval 1 checkpoint + report; generation/ (sinh code, hỏi package), detection/ (chấm hallucination)
    └── tests/              # script kiểm thử thủ công dataloader
```

## Yêu cầu trước khi chạy

- Python >= 3.10
- GPU CUDA

## Cài đặt

```bash
git clone https://github.com/DuongXi/Seminar-2-LLM-Unlearning
cd Seminar-2-LLM-Unlearning

pip install -r requirements.txt   # cài các dependency đã pin (torch không nằm trong này -- xem ghi chú trong file)
```

Không cần `pip install` chính project (không có `setup.py`). Mọi thứ chạy
bằng `bash scripts/<tên script>.sh`; các script tự đặt `PYTHONPATH` nên chạy
được từ bất kỳ thư mục nào. Muốn chạy trực tiếp 1 file Python thì đứng ở
thư mục gốc repo và dùng `python -m`, vd
`python -m pkg_halluc.training.au.train_au --help`.

## Code của paper AU

Không cần tải/fetch gì nữa -- phần code của AU (Adaptive Unlearning) mà
GA/NPO và eval cần đã được chọn lọc và gộp thẳng vào `pkg_halluc/` (đã trim
bớt phần các loss function AU có mà project này không chạy tới). File lấy từ
AU đều có dòng `# AU` ở đầu file; danh sách nằm trong `pkg_halluc/README.md`.

## Cấu hình

Mỗi bash script trong `scripts/` nhận tham số qua flag dòng lệnh, với giá
trị mặc định hợp lý sẵn ngay trong script (xem `--help` của từng script để
biết đầy đủ). Muốn đổi nhiều tham số cùng lúc mà không phải gõ từng flag,
truyền `--config <file>`:

```bash
bash scripts/train_ga.sh --config configs/default.json
bash scripts/eval.sh --config configs/default.json --tag ga --model-path ...
```

`--config` nạp giá trị từ file JSON làm **mặc định**; flag nào gõ thêm sau
đó trên dòng lệnh (dù đứng trước hay sau `--config`) vẫn **đè lên được**,
không cần sửa file JSON để chạy thử 1 giá trị khác:

```bash
# lấy hết từ default.json, chỉ đổi mỗi lr
bash scripts/train_ga.sh --config configs/default.json --lr 2e-5
```

| File | Dùng cho |
| --- | --- |
| `configs/default.json` | Qwen2.5-Coder-1.5B, quy mô đánh giá thực tế (150 prompt) |
| `configs/smoke_test.json` | chạy thử nhanh (25 prompt đánh giá, dữ liệu retain/forget mọi method giới hạn 40 dòng/split, early stopping dễ trigger hơn) |
| `configs/qwen2.5-coder-1.5b.json`, `configs/qwen2.5-coder-3b.json`, `configs/qwen2.5-1.5b.json` | preset Qwen 2.5 Coder (1.5B / 3B) |
| `configs/llama3.2-1b.json`, `configs/llama3.2-3b.json` | preset Llama 3.2 (1B / 3B) |
| `configs/deepseek-coder-1.3b.json` | preset DeepSeek Coder 1.3B |

Mỗi script chỉ đọc đúng phần JSON liên quan tới nó (vd `train_ga.sh` đọc
`methods.ga` + `model_name/seed/dtype` ở gốc; `eval.sh` đọc mục `eval`;
`train_ga_plain.sh` đọc thêm `data.max_train_samples_per_split`) -- xem
`--help` của từng script để biết chính xác field nào được dùng. Cơ chế đọc
nằm ở `pkg_halluc/common/read_config.py` (in file JSON ra dạng biến shell) +
hàm `load_config` trong `scripts/common.sh`.

**Lưu ý:** một số siêu tham số (beta/temperature/alpha/gamma của NPO,
weight_decay, warmup_steps, batch_size, gradient_accumulation_steps,
lr_scheduler_type, `lambda_eos`, `code_temp`/`package_temp` lúc eval, ...)
hiện **chưa** nằm trong config -- vẫn cố định trong source (xem
`pkg_halluc/training/au/train_au.py`, `pkg_halluc/training/plain/train_plain.py`,
`pkg_halluc/evaluation/eval_variant.py`).
Muốn expose thêm cái nào thành flag/config thì báo, làm tiếp.

`--use-lora` -- train LoRA adapter (qua PEFT) thay vì fine-tune toàn bộ
trọng số: chỉ adapter nhỏ cần gradient/optimizer, base model đóng băng. Dùng
khi model lớn không đủ VRAM để full fine-tune (vd T4 16GB trên Kaggle).

## Cách chạy

Model chính của project là `Qwen/Qwen2.5-Coder-0.5B-Instruct` trên Kaggle T4
-- mọi script bên dưới đều mặc định model này nếu không truyền `--model`.
Có thể thay hết các `--model ...` bên dưới bằng `--config configs/default.json`
(hoặc config khác) nếu muốn -- xem mục "Cấu hình" ở trên.

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

Mỗi script độc lập, dừng giữa chừng rồi chạy lại từng bước riêng vẫn được.
Mỗi script có `--help` liệt kê đầy đủ flag + giá trị mặc định (lr, epochs,
seed, dtype, use-lora, ...).
