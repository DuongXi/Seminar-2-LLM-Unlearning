import argparse
import os
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
from peft import PeftModel
import random

def is_lora_model(model_path: str) -> bool:
    """Check whether a saved model directory contains a LoRA adapter."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))

def load_model(#adapter_path:str="vanilla", 
               #base_name:str=None,
               model_path:str=None,
               device_map="auto", 
               dtype=torch.float16, 
               attn_implementation=None
               ):

    # Optional args, most likely not going to change
    extra_kwargs = {}
    if dtype is not None:
        extra_kwargs["torch_dtype"] = dtype
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

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
                device_map=device_map,
                dtype=dtype
                **extra_kwargs,
            )
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
            print("LoRA adapter merged into base model.")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        else:
            base      = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=dtype, 
                device_map=device_map,
                **extra_kwargs,
            )
            model = base
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # elif base_name is not None:
    #     tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    #     base      = AutoModelForCausalLM.from_pretrained(
    #         base_name, 
    #         dtype=dtype, 
    #         device_map=device_map,
    #         **extra_kwargs,
    #         )
    #     if adapter_path == "vanilla":
    #         model = base
    #     else:
    #         model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)

    return tokenizer, model





