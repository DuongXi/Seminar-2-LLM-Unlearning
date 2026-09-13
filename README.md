# Readme
Tham số
--model_path	Model nền (frozen) dùng để trích xuất hidden state
--data_path	File dữ liệu huấn luyện	
--output_dir	Nơi lưu steering vector + metadata	
--batch_size	Số ví dụ xử lý cùng lúc khi trích xuất hidden state	
--num_epochs	Số vòng lặp gradient descent để huấn luyện vector	
--layer_start / --layer_end	Khoảng layer sẽ thử

Vì --data_path kết thúc bằng .csv, code sẽ gọi hàm _records_from_results_csv. Hàm này lấy 4 cột từ file CSV:

Prompts — đề bài gốc
Answers — code do model sinh ra
pip_valid — danh sách tên package có thật (label = 1), tìm được trong Answers qua cú pháp import X
pip_hallucinated — danh sách tên package ảo giác (label = 0), cũng tìm trong Answers

Với mỗi tên trong 2 danh sách này, code tìm lại đúng vị trí ký tự của tên đó trong Answers (ví dụ dòng import numpy), cắt văn bản Answers đến ngay trước tên đó, ghép với Prompts, rồi tokenize để lấy position (vị trí token cuối cùng ngay trước khi tên package "sẽ được sinh ra"). Kết quả là một ví dụ huấn luyện: {"prompt": ..., "label": 0 hoặc 1, "positions": [vị trí đó]}.

