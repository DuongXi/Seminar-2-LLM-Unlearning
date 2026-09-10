import os
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def is_lora_model(model_path: str) -> bool:
    """Check whether a saved model directory contains a LoRA adapter."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))

def load_model(model_path:str=None,
               device_map="auto", 
               dtype=torch.float16
               ):
    tokenizer, model = None, None

    if model_path is not None:
        if is_lora_model(model_path=model_path):
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg.get("base_model_name_or_path", model_path)
            print(f"Detected LoRA adapter at {model_path}")
            print(f"Loading base model from {base_model_path} ...")

            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=dtype,
                device_map=device_map
            )
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
            print("LoRA adapter merged into base model.")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        else:
            base      = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=dtype, 
                device_map=device_map
            )
            model = base
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer, model





