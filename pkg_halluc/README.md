
## `common/` -- dùng chung nhiều bước

| File | Vai trò  | 
| --- | --- |
| `model_presets.py` | Bảng tên preset model -> id Hugging Face 
| `model_setup.py` | Library Tải model, build model + tokenizer, sinh thử kiểm tra môi trường 
| `download_model.py` | Tải base model + sanity-check
| `read_config.py` | Đọc `model_config/*.json`
| `tri_mask_utils.py` | Prefix prompt, đọc dữ liệu tri-mask, collator, LoRA, `load_model_auto` (train GA/NPO và eval) 

## `package_loader/` -- dataset và dữ liệu tri-mask

| File | Vai trò  
| --- | --- | 
| `prompt_config.py` | Prompt hỏi package, thông số token theo dòng model  
| `utils.py` | Setup tokenizer, đọc CSV/JSONL, parse package, map token-ký tự
| `unlearn_loader.py` | `PackageUnlearningDataset` (tập forget/retain) và các dataloader  
| `tsv_loader.py` | Dataset/dataloader theo định dạng TSV 
| `generate_tri_mask.py` | Dựng record tri-mask (0 bỏ qua, 1 retain, 2 forget) từ dataset  
| `build_tri_mask_data.py` | Dựng file JSONL retain/forget cho GA/NPO  | 

## `training/` -- train các method

| File | Vai trò  
| --- | ---  | 
| `tri_mask/train_tri_mask.py` | Train GA/NPO tri-mask trên dữ liệu tĩnh, có early stopping (val tách từ retain)  
| `tri_mask/ga_trainer.py` | `GradientAscentTrainer`  
| `tri_mask/npo_trainer.py` | `NPOTrainer`  
| `plain/train_plain.py` | Train GA-plain/NPO-plain
| `plain/plain_trainers.py` | `GAPlainTrainer`, `NPOPlainTrainer`  

## `evaluation/` -- đánh giá và báo cáo

