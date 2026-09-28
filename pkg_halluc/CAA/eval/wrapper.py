"""
Universal Model Steering Wrapper for CAA
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pkg_halluc.common.model_presets import resolve_model_name
from utils import (
    get_transformer_layers,
)


class SteeringHook:
    """
    Forward hook applied to a transformer layer to add steering vector activations.
    Non-invasive: works cleanly with any model architecture and supports both
    prompt prefill and autoregressive KV-cache generation steps.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.add_activations: Optional[t.Tensor] = None
        self.multiplier: float = 1.0
        self.from_position: Optional[int] = None
        self.last_activations: Optional[t.Tensor] = None
        self.calc_dot_product_with: Optional[t.Tensor] = None
        self.dot_products: List[Any] = []

    def set_add(self, activations: t.Tensor, multiplier: float = 1.0, from_pos: Optional[int] = None):
        self.add_activations = activations
        self.multiplier = float(multiplier)
        self.from_position = from_pos

    def reset(self):
        self.add_activations = None
        self.multiplier = 1.0
        self.from_position = None
        self.last_activations = None
        self.calc_dot_product_with = None
        self.dot_products = []

    def __call__(self, module, inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        hidden_states = outputs[0] if is_tuple else outputs
        self.last_activations = hidden_states

        if self.add_activations is None:
            return outputs

        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        vec = (self.multiplier * self.add_activations).to(device=device, dtype=dtype)
        if vec.dim() == 1:
            vec = vec.view(1, 1, hidden_dim)
        elif vec.dim() == 2:
            vec = vec.unsqueeze(0)

        # Autoregressive generation step with KV-cache (single new token generated)
        if seq_len == 1:
            augmented = hidden_states + vec
        else:
            # Prompt prefill step
            augmented = hidden_states.clone()
            if self.from_position is not None and 0 <= self.from_position < seq_len:
                augmented[:, self.from_position:, :] += vec
            else:
                # Default: steer the last token predicting the subsequent response
                augmented[:, -1:, :] += vec

        if is_tuple:
            return (augmented,) + outputs[1:]
        return augmented


class ModelWrapper:
    """
    Universal Model Wrapper for CAA Activation Addition / Steering.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        from_config: bool = False,
        local_files_only: bool = True,
    ):
        self.device = device or ("cuda" if t.cuda.is_available() else "cpu")
        self.use_chat = use_chat

        # Resolve the model identifier.
        if model_name_or_path:
            self.model_name_path = resolve_model_name(model_name_or_path)
        elif config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            raw_name = cfg_data.get("model_name")
            if not raw_name:
                raise ValueError(f"Model name missing from config: {config_path}")
            self.model_name_path = resolve_model_name(raw_name)
        else:
            raise ValueError("Provide model_name_or_path or config_path.")

        # Select the compute dtype.
        if dtype:
            self.torch_dtype = getattr(t, dtype, t.float16)
        elif self.device == "cuda":
            self.torch_dtype = t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        else:
            self.torch_dtype = t.float32

        # Load the tokenizer.
        token = hf_token or os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path,
            token=token,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model.
        if from_config:
            cfg = AutoConfig.from_pretrained(
                self.model_name_path,
                token=token,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForCausalLM.from_config(cfg)
            self.model = self.model.to(dtype=self.torch_dtype, device=self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_path,
                token=token,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )

        if override_model_weights_path is not None and os.path.exists(override_model_weights_path):
            self.model.load_state_dict(t.load(override_model_weights_path, map_location=self.device))

        self.model.eval()

        # Attach steering hooks to transformer layers.
        self.layers = get_transformer_layers(self.model)
        self.num_layers = len(self.layers)
        self.hooks: Dict[int, SteeringHook] = {}
        self.hook_handles = []

        for i, layer in enumerate(self.layers):
            hook = SteeringHook(layer_idx=i)
            handle = layer.register_forward_hook(hook)
            self.hooks[i] = hook
            self.hook_handles.append(handle)

    def set_save_internal_decodings(self, value: bool):
        pass

    def set_from_positions(self, pos: int):
        for hook in self.hooks.values():
            hook.from_position = pos

    def set_add_activations(self, layer: int, activations: t.Tensor, multiplier: float = 1.0):
        """Register or update a steering vector to be added at a given layer."""
        if layer in self.hooks:
            self.hooks[layer].set_add(activations, multiplier=multiplier)

    def reset_all(self):
        """Reset all active steering vectors and internal buffers across all layers."""
        for hook in self.hooks.values():
            hook.reset()

    def get_last_activations(self, layer: int) -> Optional[t.Tensor]:
        """Retrieve output activations from the most recent forward pass at layer."""
        if layer in self.hooks:
            return self.hooks[layer].last_activations
        return None

    def format_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        model_output: Optional[str] = None,
    ) -> Tuple[List[int], int]:
        """
        Format user/system prompt with model's native chat template.
        Returns: (token_ids, prompt_token_length)
        """
        # Modern HuggingFace chat models (Llama 3.2, Qwen 2.5, DeepSeek Coder)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
        except Exception:
            # Use plain text when the tokenizer has no chat template.
            raw_text = f"{system_prompt}\n\n{user_input}\n\nAssistant: " if system_prompt else f"{user_input}\n\nAssistant: "
            prompt_ids = self.tokenizer.encode(raw_text, add_special_tokens=True)

        prompt_len = len(prompt_ids)
        if model_output:
            suffix_ids = self.tokenizer.encode(model_output, add_special_tokens=False)
            prompt_ids = prompt_ids + suffix_ids

        return prompt_ids, prompt_len

    def generate(self, tokens: t.Tensor, max_new_tokens: int = 100) -> str:
        """Autoregressive text generation with active steering hooks."""
        with t.no_grad():
            tokens = tokens.to(self.device)
            prompt_len = tokens.size(1)

            # Set instruction boundary if not manually set
            for hook in self.hooks.values():
                if hook.from_position is None:
                    hook.from_position = prompt_len - 1

            generated_ids = self.model.generate(
                inputs=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    def generate_text(
        self,
        user_input: str,
        model_output: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 100,
    ) -> str:
        """Format text prompt, execute steered generation, and return decoded text."""
        token_ids, prompt_len = self.format_prompt(
            user_input=user_input,
            system_prompt=system_prompt,
            model_output=model_output,
        )
        tokens_tensor = t.tensor([token_ids], device=self.device)

        # Set steering start position at the prompt-response boundary
        self.set_from_positions(prompt_len - 1)

        with t.no_grad():
            output_ids = self.model.generate(
                inputs=tokens_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Slice newly generated tokens
            new_tokens = output_ids[0, prompt_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_logits(self, tokens: t.Tensor) -> t.Tensor:
        """Execute forward pass and return output logits."""
        with t.no_grad():
            tokens = tokens.to(self.device)
            return self.model(tokens).logits

    def get_logits_from_text(
        self,
        user_input: str,
        model_output: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> t.Tensor:
        """Format text, configure steering boundary, and compute next-token logits."""
        token_ids, prompt_len = self.format_prompt(
            user_input=user_input,
            system_prompt=system_prompt,
            model_output=model_output,
        )
        tokens_tensor = t.tensor([token_ids], device=self.device)

        # Steer activations at the last token to directly impact the next token prediction
        self.set_from_positions(len(token_ids) - 1)
        return self.get_logits(tokens_tensor)

    def score_response(
        self,
        user_input: str,
        response: str,
        system_prompt: Optional[str] = None,
    ) -> float:
        token_ids, prompt_len = self.format_prompt(
            user_input=user_input,
            system_prompt=system_prompt,
            model_output=response,
        )
        response_ids = token_ids[prompt_len:]
        if not response_ids:
            raise ValueError("Cannot score an empty response.")

        tokens = t.tensor([token_ids], device=self.device)
        self.set_from_positions(prompt_len - 1)
        logits = self.get_logits(tokens)[0, prompt_len - 1 : -1]
        targets = t.tensor(response_ids, device=logits.device)
        token_log_probs = t.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1))
        return token_log_probs.mean().item()

    def __del__(self):
        """Cleanup forward hook handles upon deletion."""
        if hasattr(self, "hook_handles"):
            for h in self.hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass