> 🇻🇳 Đây là bản dịch tiếng Việt của [`README.md`](README.md) trong thư mục
> này. Nếu có chỗ nào dịch lệch ý, **bản tiếng Anh là bản chính thức**.

# data/

Thư mục này **để trống là có chủ đích**. Theo mặc định, pipeline đọc prompt
seed, prompt đánh giá, snapshot package PyPI, và danh sách false-positive
thẳng từ repo Adaptive Unlearning (AU) đã tải về (xem
`pkg_halluc/deps/fetch.py` và `paths.py`):

| Cái gì                       | Nguồn mặc định (bên trong repo AU đã tải)                       |
| ----------------------------- | ------------------------------------------------------------- |
| Prompt seed (19 cái)          | `New_Data_Set/prompts.jsonl`                                   |
| Prompt đánh giá (~4.900 cái)  | `Package_Hallucination_Testing/Data/prompts.jsonl`              |
| Snapshot package PyPI         | `Package_Hallucination_Testing/Data/pypi_package_names.csv`     |
| Danh sách package false-positive | `Package_Hallucination_Testing/Data/false_positive_packages.csv`|

Nhóm không giữ 1 bản copy thứ 2 của các file đó trong repo này -- giữ thêm
1 bản chỉ tạo ra 1 nguồn sự thật thứ 2 phải đồng bộ theo, dễ lệch nhau.

Nếu bạn muốn đánh giá trên **tập prompt của riêng bạn** (vd 1 mẫu khác, hoặc
prompt bằng ngôn ngữ khác), thả 1 file `.jsonl`/`.csv` cùng định dạng vào
đây rồi trỏ config vào nó:

```json
{
  "eval": { "eval_prompts_path": "data/my_prompts.jsonl" }
}
```

Bất cứ thứ gì bạn thêm vào `data/` đều được git track (chỗ này dành cho file
nhỏ, tự chọn lọc tay) -- còn mọi thứ trong `work_dir/` lúc chạy (repo đã tải,
model, dữ liệu tri-mask dựng ra, kết quả eval) thì không, xem `.gitignore`.
