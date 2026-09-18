# Phương pháp 5: Representation Steering (Activation Engineering)

Tài liệu này mô tả chi tiết phương pháp **Representation Steering** được thêm vào dự án *Package Hallucination Mitigation*. Phương pháp này được cài đặt dựa trên cảm hứng từ các framework đánh giá Safety của LLM (như Axbench, SteeringSafety) và paper TSV (ICML 2025).

---

## 1. Tổng quan phương pháp (Overview)

Thay vì fine-tune trọng số (như thuật toán GA, NPO) hay lọc từ ngữ ở đầu ra (như PackMonitor), **Representation Steering** can thiệp trực tiếp vào luồng suy nghĩ (hidden states) của LLM trong quá trình sinh text (inference).

**Công thức lõi:**
$$h^{(L)} \leftarrow h^{(L)} + \lambda \cdot \mathbf{v}$$

*   $h^{(L)}$: Trạng thái ẩn (hidden state) tại layer thứ $L$.
*   $\mathbf{v}$: Steering Vector (Vector định hướng) - tính bằng khoảng cách giữa nhóm package thật và package ảo.
*   $\lambda$: Cường độ can thiệp (Steering strength).

Nhờ cách này, mô hình được "kéo" ra khỏi vùng không gian hallucination mà **không cần thay đổi bất kỳ trọng số (weights) gốc nào**.

---

## 2. Chi tiết các file đã cài đặt

Toàn bộ code của phương pháp được đóng gói độc lập, tuân thủ đúng kiến trúc `MethodSpec` của pipeline chung, không làm ảnh hưởng đến các method khác.

```text
src/pkg_halluc/
├── methods/
│   ├── steering.py                 # File cấu hình chính, kết nối với pipeline của nhóm
│   ├── steering_vector_utils.py    # Chứa thuật toán phân tích CSV và tính Mean Difference Vector
│   └── llm_layers.py               # Chứa custom module (TSVLayer) để can thiệp vào Residual Stream
├── templates/steering/
│   └── steering_eval_adapter.py    # Adapter giúp load vector và patch model khi chạy đánh giá
└── config.py                       # (Đã sửa) Thêm hyperparameter cho Steering
```

---

## 3. Các điểm nhấn Kỹ thuật & Tối ưu (Engineering Highlights)

Để đưa các lý thuyết từ bài báo vào thực tế dự án chạy trên Qwen2.5, tôi đã phải thiết kế và fix các vấn đề sau:

### 3.1. Kỹ thuật trích xuất Vector tại "Boundary Token"
*Trong file `steering_vector_utils.py`*
*   Không lấy vector trung bình của toàn bộ câu, mà viết hàm `_find_package_boundary()` để **dò đúng vị trí token ngay trước khi tên thư viện bắt đầu được sinh ra**. 
*   Sử dụng thuật toán **Mean Difference**: $v = mean(Real\_Activations) - mean(Hallucinated\_Activations)$ phân chia dựa trên nhãn đối chiếu từ PyPI (cột `pip_valid` và `pip_hallucinated`).

### 3.2. Sửa 4 lỗi chí mạng từ mã nguồn gốc của Paper TSV
*Trong file `llm_layers.py`*
1.  **Lỗi ép kiểu cứng (Hard-coded Dtype):** Code gốc dùng hàm `.half()` ép kiểu FP16, làm sập mô hình nếu đang chạy `bfloat16`. Tôi đã sửa thành `x.to(dtype=x.dtype)`.
2.  **Khả năng tương thích Qwen2.5:** Code gốc gán cứng tên `if model == 'qwen2.5-7B'`. Tôi đã đập bỏ và thay bằng thư viện `inspect` (kiểm tra động signature của `self_attn` xem có cần `position_embeddings` không). Điều này giúp code **chạy được trên mọi kiến trúc model** (Qwen 0.5B, Llama 3...).
3.  **Tương thích thư viện Transformers:** Xử lý linh hoạt việc API Attention trả về 2 output hoặc 3 output tùy phiên bản Transformers.
4.  **Cơ chế Padding:** Cài đặt bắt buộc `padding_side="right"` để đảm bảo trích xuất chính xác token cuối cùng của chuỗi khi batching.

---

## 4. Hướng dẫn sử dụng (How to run)

Code đã được tích hợp hoàn toàn tự động vào CLI `pkg_halluc`. Để chạy thử nghiệm, bạn làm các bước sau:

**Bước 1:** Chuẩn bị data. Đảm bảo file kết quả sinh code (VD: `LLM_LY_results.csv`) được đặt tại `data/LLM_LY_results.csv`.

**Bước 2:** Bật cấu hình. Mở file `configs/default.json` và đảm bảo `methods.steering.enabled` = `true`.

**Bước 3:** Chạy lệnh tính Steering Vector. Hệ thống sẽ tự động quét qua các layer (từ 4 đến 12), tính vector và lưu cache lại layer tốt nhất.
```bash
pkg_halluc prepare --method steering
```

**Bước 4:** Đánh giá độ hiệu quả (So sánh với Base, GA, NPO).
```bash
pkg_halluc evaluate --method steering
```
*(Kết quả đánh giá sẽ tự động sinh ra thư mục `eval_runs/steering/FINAL_RESULTS.csv` giống hệt các method khác).*
