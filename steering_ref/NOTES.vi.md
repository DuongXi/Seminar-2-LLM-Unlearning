
# Tài liệu tham khảo TSV (lưu trữ, chưa tích hợp)


```bibtex
@inproceedings{park2025steer,
  title={Steer {LLM} Latents for Hallucination Detection},
  author={Seongheon Park and Xuefeng Du and Min-Hsuan Yeh and Haobo Wang and Yixuan Li},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

**Trạng thái: thuần lưu trữ.**
## Vì sao chỉ lấy 2 file 

Phương pháp gốc của TSV (1 vector học được qua MLE trên mô hình phân phối
von-Mises-Fisher, bán giám sát bằng pseudo-label qua Optimal Transport --
xem Section 4 của paper) giải quyết 1 bài toán khác với kế hoạch trong
proposal: TSV cần cơ chế pseudo-labeling vì *nhãn thật đắt đỏ* trong bối
cảnh của họ (câu trả lời QA này có đúng sự thật không? -- cần người hoặc
GPT-4o chấm). Package hallucination không gặp vấn đề đó: 1 tên package có
thật hay không là **tra cứu PyPI/npm, miễn phí và chính xác tuyệt đối** --
đúng là thứ mà `build_valid_packages.py` / `package_detection.py` trong
project này đã làm sẵn rồi. Không có chuyện thiếu dữ liệu gán nhãn cần khắc
phục, nên không có lý do gì phải làm lại vòng lặp train bán giám sát của
TSV.

Cái **thật sự** dùng lại được, không phụ thuộc việc vector steering được
tính bằng phương pháp thống kê nào, là **cơ chế cộng 1 vector vào hidden
state của model tại 1 layer chỉ định, lúc đang generate** -- đây là hạ tầng
activation-engineering dùng chung, không gắn với mục tiêu train riêng của
TSV. Đó chính là `llm_layers.py`, giữ nguyên toàn bộ:

| Thành phần | Làm gì |
| --- | --- |
| `get_layers`, `get_layers_path`, `find_longest_modulelist`, `find_module`, `get_nested_attr`/`set_nested_attr` | Không phụ thuộc kiến trúc cụ thể: tìm danh sách decoder layer và các submodule có tên (`mlp`, `self_attn`, ...) của model mà không hard-code theo 1 class model cụ thể nào. |
| `TSVLayer` | Chính là bước can thiệp: `x -> x + lam * v` (Eq. 2 trong paper: `h^(l) <- h^(l) + lambda*v`). Dùng chung -- hoạt động với **bất kỳ** vector `v` nào, không riêng gì vector train theo mục tiêu của TSV. Đây đúng là thứ mà mục "Intervention: add the (optionally confidence-scaled) steering vector to the hidden state at generation time" trong proposal cần. |
| `add_tsv_layers`, `LlamaDecoderLayerWrapper` | Monkey-patch 1 model đã load để chèn `TSVLayer` vào đúng layer chỉ định, tại 1 trong 3 vị trí: residual stream (`component='res'`), output của MLP (`'mlp'`), hoặc output của attention (`'attn'`). Khớp với kế hoạch ablation trong proposal ("multi-layer steering") và ablation của chính paper (Fig. 3a: residual stream ở các layer đầu-giữa cho kết quả tốt nhất). |

`hidden_state_utils.py::get_last_non_padded_token_rep` (trích từ
`train_utils.py`) là tiện ích nhỏ để lấy ra "hidden state của token thật
cuối cùng" từ 1 batch có padding -- cần cho cả RQ1 (probing: trích activation
tại vị trí sinh tên package) lẫn việc dựng steering vector (mean-difference
cần vector activation của từng ví dụ để lấy trung bình).

**Không lấy vào:**

- `sinkhorn_knopp.py`, và phần còn lại của `train_utils.py` (pseudo-labeling
  bằng OT, cập nhật centroid theo EMA) -- máy train bán giám sát riêng của
  TSV, không cần (xem lý do ở trên).
- `tsv_main.py` -- script gốc đầy đủ. Không copy vào đây, nhưng đáng đọc
  trực tiếp từ `tsv-main.zip` nếu muốn xem `add_tsv_layers` / `TSVLayer`
  được dùng từ đầu tới cuối như thế nào (hàm `test_model` trong đó là ví dụ
  rõ nhất về cách dùng lúc inference).
- `cache_utils.py` -- bản copy đóng băng của 1 class `transformers.Cache`
  nội bộ cũ, giữ lại chỉ để type-hint. Không cần; `transformers==4.57.6`
  (bản đang pin trong project này) đã có class riêng của nó rồi.
- `data_indices/*.npy`, `gen.sh`, `gt.sh`, `train.sh`, `tsv.yml`,
  `requirements.txt` -- gắn với các benchmark QA riêng của TSV (TruthfulQA /
  TriviaQA / SciQ / NQ Open) và môi trường Python 3.8.15. Không áp dụng
  được ở đây.

## Vấn đề đã biết / cần chỉnh sửa trước khi dùng

Chưa có gì bên dưới được sửa cả -- đây là danh sách để người tiếp theo cầm
lên biết mà làm, không phải lời hứa là code chạy được ngay:

- **Dtype bị hard-code.** `TSVLayer.forward` gọi `.half()` vô điều kiện.
  Project này tự chọn dtype theo từng GPU (`config.py::resolve_dtype` --
  bfloat16 nếu hỗ trợ, không thì float16); `llm_layers.py` cần tôn trọng
  điều đó thay vì mặc định luôn là fp16.
- **`.cuda()`/xử lý device bị hard-code ở phần còn lại của TSV** (không nằm
  trong 2 file giữ lại ở đây, nhưng đáng biết nếu quay lại đọc `tsv_main.py`
  để tham khảo) -- project này nhìn chung dùng `device_map="auto"`.
- **Lệch phiên bản `transformers`.** `LlamaDecoderLayerWrapper.forward` gọi
  `self.llama_decoder_layer.self_attn(...)` với danh sách tham số viết tay
  khớp với bản `transformers` mà TSV được build lúc đó (đủ cũ để pin Python
  3.8.15). `transformers==4.57.6` (bản đang pin trong project này) có thể
  có chữ ký hàm `self_attn` khác -- **cần kiểm tra lại với đúng bản đang cài
  trước khi dùng**, không được mặc định là chạy được.
- **Chưa test với Qwen2.5.** `LlamaDecoderLayerWrapper` của TSV có 1 nhánh
  riêng cho `model_name == 'qwen2.5-7B'` (bỏ qua không truyền
  `position_embeddings` vào `self_attn`), ngụ ý là nó *từng* chạy được với
  1 model Qwen2.5 tại thời điểm nào đó -- nhưng đó là với bản `transformers`
  cũ hơn mà TSV pin, không phải bản của project này. Cần kiểm tra lại cụ thể
  với `Qwen/Qwen2.5-0.5B-Instruct` (model mặc định của project này).
- **Phía padding.** `get_last_non_padded_token_rep` giả định chuỗi được
  **right-padding** (nó lấy chỉ số `hidden_states[i, lengths[i]-1, :]`, chỉ
  đúng là token thật cuối cùng nếu phần padding nằm *sau* nó). Đáng lưu ý:
  chính pipeline đánh giá của project này từng hiện cảnh báo của
  `transformers` về việc phát hiện right-padding trên 1 model decoder-only
  (xem cấu hình tokenizer trong `Package_Hallucination_Testing/generate_code.py`,
  qua code AU đã lấy về) -- nên giả định của hàm này có khớp với cấu hình
  tokenizer mà bản cài đặt steering sau này dùng hay không thì chưa chắc.
  Cần kiểm tra `padding_side` một cách tường minh, không được mặc định.

## Ánh xạ sang kế hoạch Representation Steering trong proposal

Từ `Proposal.docx`, mục Methods > Representation Steering:

| Bước trong proposal | Bắt đầu từ đâu |
| --- | --- |
| "Probing: extract hidden states at the package-name generation position across all layers" | `hidden_state_utils.get_last_non_padded_token_rep` + `output_hidden_states=True` trên 1 lượt forward pass HF bình thường (xem `tsv_main.py::get_ex_data` trong file zip gốc để thấy mẫu cách làm -- không copy vào đây vì phần code nối xung quanh nó là đặc thù của TSV). |
| "Steering vector construction: mean-difference vector between real- and hallucinated-package activation clusters" | Code hoàn toàn mới, TSV không có phần này (TSV tính 1 vector *học được* qua MLE, không phải công thức mean-difference / kiểu ITI đóng dạng). Nhãn có sẵn miễn phí từ chính bước tra cứu PyPI của project này, như giải thích ở trên -- không cần pseudo-labeling. |
| "Intervention: add the ... steering vector to the hidden state at generation time" | `llm_layers.add_tsv_layers` + `TSVLayer`, sau khi sửa các vấn đề liệt kê ở trên. |
| "Variants ... multi-layer steering; adaptive steering strength scaled by probe confidence" | `add_tsv_layers` hiện đã hỗ trợ chọn 1 `str_layer` + `component` duy nhất; multi-layer nghĩa là gọi nó (hoặc 1 bản tổng quát hoá hơn) tại nhiều layer cùng lúc -- bản gốc cũng chưa cài đặt sẵn phần này. |

Xem thêm docstring đầu file `pkg_halluc/methods/steering.py` để biết
phần này khớp vào đâu trong registry method của pipeline khi có người bắt
tay vào code.
