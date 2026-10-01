"""
Unified backend loader for package-hallucination evaluation.

Provides `load_backend(...)` returning a Backend object with a uniform
`.generate(system_prompt, user_prompt, max_new_tokens, is_reasoning_model)`
method, plus an `HFBackend` wrapper for vanilla HuggingFace models.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from pkg_halluc.CAA.wrapper import ModelWrapper


# Reasoning-model post-processing
def strip_reasoning_preamble(text: str) -> str:
    end = text.find("</think>")
    return text[end + len("</think>"):].strip() if end != -1 else text


# Backend classes
class Backend:
    name: str = "base"

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512,
                 is_reasoning_model: bool = False) -> str:
        raise NotImplementedError

    def set_multiplier(self, value: float) -> None:
        pass

    def describe(self) -> str:
        return self.name


class HFBackend(Backend):
    name = "hf"

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512,
                 is_reasoning_model: bool = False) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        response = self.tokenizer.decode(
            outputs.sequences[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        if is_reasoning_model:
            response = strip_reasoning_preamble(response)
        return response


class CAABackend(Backend):
    name = "caa"

    def __init__(self, wrapper, active_layers):
        self.wrapper = wrapper
        self.active_layers = active_layers

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512,
                 is_reasoning_model: bool = False) -> str:
        response = self.wrapper.generate_text(
            user_input=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        if is_reasoning_model:
            response = strip_reasoning_preamble(response)
        return response

    def set_multiplier(self, value: float) -> None:
        for layer_idx, hook in self.wrapper.hooks.items():
            if hook.add_activations is not None:
                hook.multiplier = float(value)

    def describe(self) -> str:
        info = ", ".join(
            f"layer {i}: mult={self.wrapper.hooks[i].multiplier}"
            for i in self.active_layers
        )
        return f"caa({info})"



# Public entry point
def load_backend(
    caa_bundle: Optional[str] = None,
    device: str = "cuda",
    dtype: Optional[str] = None,
    multiplier: Optional[float] = None,
    local_files_only: bool = True,
    verbose: bool = True,
) -> Backend:
    """
    Load a CAA backend from a steering bundle directory.

    Vanilla HF models are handled by hallu_pkg.py via utils.model.load_model
    + HFBackend, not through this function.
    """
    if not caa_bundle:
        raise ValueError("load_backend requires caa_bundle.")
    if not os.path.isdir(caa_bundle):
        raise FileNotFoundError(f"CAA bundle not found: {caa_bundle}")

    if verbose:
        print(f"[caa_loader] loading CAA bundle from {caa_bundle}")

    wrapper = ModelWrapper.from_steering_bundle(
        bundle_dir=caa_bundle,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )

    active = [i for i, h in wrapper.hooks.items()
              if h.add_activations is not None]
    if not active:
        raise RuntimeError(
            f"No active steering vectors found in {caa_bundle}")

    backend = CAABackend(wrapper, active_layers=active)

    if multiplier is not None:
        backend.set_multiplier(multiplier)
        if verbose:
            print(f"[caa_loader] multiplier overridden to {multiplier}")

    if verbose:
        print(f"[caa_loader] backend ready: {backend.describe()}")
    return backend