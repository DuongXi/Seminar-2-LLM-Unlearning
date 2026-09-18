"""Architecture-agnostic activation-engineering utilities (TSV paper, Apache-2.0).

Adapted from *Steer LLM Latents for Hallucination Detection* (Park et al., ICML 2025).
Source: https://github.com/deeplearning-wisc/tsv (Apache-2.0).

Changes vs upstream (steering_reference/tsv/llm_layers.py):
1. TSVLayer.forward: cast vector to input dtype instead of hard-coded .half()
2. LlamaDecoderLayerWrapper: handles 2-return and 3-return self_attn conventions
3. position_embeddings passed only when model signature accepts it (inspect)
4. get_last_non_padded_token_rep: documented right-padding requirement
"""
from __future__ import annotations

import inspect
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def get_last_non_padded_token_rep(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Return last real token hidden state per sequence (right-padding assumed)."""
    mask = attention_mask.squeeze()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    lengths = mask.long().sum(dim=1)
    batch_size, seq_len, hidden_size = hidden_states.shape
    return torch.stack([hidden_states[i, lengths[i] - 1, :] for i in range(batch_size)])


def get_nested_attr(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path: str, value) -> None:
    parts = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(parts[:-1]))
    setattr(parent, parts[-1], value)


def find_longest_modulelist(model: nn.Module, path: str = "") -> tuple:
    longest_path, longest_len = path, 0
    for name, child in model.named_children():
        child_path = f"{path}.{name}" if path else name
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = child_path
        sub_path, sub_len = find_longest_modulelist(child, child_path)
        if sub_len > longest_len:
            longest_len = sub_len
            longest_path = sub_path
    return longest_path, longest_len


def find_module(block: nn.Module, keywords: list) -> nn.Module:
    for name, module in block.named_modules():
        if any(kw in name for kw in keywords):
            return module
    names = [n for n, _ in block.named_modules()]
    raise ValueError(f"Could not find any of {keywords} in: {names}")


def get_layers_path(model: PreTrainedModel) -> str:
    path, _ = find_longest_modulelist(model)
    return path


def get_layers(model: PreTrainedModel) -> nn.ModuleList:
    return get_nested_attr(model, get_layers_path(model))


class TSVLayer(nn.Module):
    """Adds a steering vector h <- h + lambda * v, optionally at specific positions."""

    def __init__(self, tsv: Tensor, lam):
        super().__init__()
        self.register_buffer("tsv", tsv.float())
        self.lam = lam[0] if isinstance(lam, (list, tuple)) else float(lam)

    def forward(self, x: Tensor, steering_positions=None) -> Tensor:
        if self.tsv is None:
            return x
        # Cast to input dtype -- fixes the hard-coded .half() bug
        vector = self.lam * self.tsv.to(device=x.device, dtype=x.dtype)

        if steering_positions is None:
            return x + vector
        if isinstance(steering_positions, (list, tuple)) and len(steering_positions) == 0:
            return x
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, hidden_size], got {tuple(x.shape)}")

        if torch.is_tensor(steering_positions):
            positions = steering_positions.detach().flatten().tolist()
        elif isinstance(steering_positions, int):
            positions = [steering_positions]
        else:
            positions = list(steering_positions)

        if len(positions) != x.shape[0]:
            raise ValueError(
                f"steering_positions length ({len(positions)}) != batch size ({x.shape[0]})"
            )

        mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        for i, pos in enumerate(positions):
            if isinstance(pos, int) and 0 <= pos < x.shape[1]:
                mask[i, pos] = 1.0
        return x + mask.unsqueeze(-1) * vector


def _attn_accepts_position_embeddings(self_attn: nn.Module) -> bool:
    try:
        sig = inspect.signature(self_attn.forward)
        return "position_embeddings" in sig.parameters
    except (ValueError, TypeError):
        return False


class LlamaDecoderLayerWrapper(nn.Module):
    """Wraps a decoder layer to inject steering after the MLP (residual stream)."""

    def __init__(self, decoder_layer: nn.Module, tsv_layer: TSVLayer) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.tsv_layer = tsv_layer
        self._attn_takes_pos_emb = _attn_accepts_position_embeddings(decoder_layer.self_attn)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        steering_positions=None,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.decoder_layer.input_layernorm(hidden_states)

        attn_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        if self._attn_takes_pos_emb and position_embeddings is not None:
            attn_kwargs["position_embeddings"] = position_embeddings
        attn_kwargs.update(kwargs)

        attn_outputs = self.decoder_layer.self_attn(**attn_kwargs)
        if len(attn_outputs) == 3:
            hidden_states, self_attn_weights, present_key_value = attn_outputs
        else:
            hidden_states, self_attn_weights = attn_outputs
            present_key_value = past_key_value

        hidden_states = residual.to(hidden_states.device) + hidden_states
        residual = hidden_states
        hidden_states = self.decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.tsv_layer(hidden_states, steering_positions=steering_positions)
        return hidden_states


def add_tsv_layers(
    model: PreTrainedModel,
    tsv: Tensor,
    alpha,
    str_layer: int,
    component: str = "res",
) -> None:
    """Patch model to inject steering vector at decoder layer str_layer.

    Parameters
    ----------
    model      : HuggingFace CausalLM (on device).
    tsv        : Steering vector [hidden_size].
    alpha      : Steering strength lambda (scalar or 1-elem list).
    str_layer  : Decoder layer index to instrument.
    component  : "res" (residual stream) | "mlp" | "attn"
    """
    layers = get_layers(model)
    tsv_layer = TSVLayer(tsv, alpha)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    attn_keywords = ["self_attn"]

    if component == "mlp":
        layer = layers[str_layer]
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, tsv_layer)
    elif component == "attn":
        layer = layers[str_layer]
        original_attn = find_module(layer, attn_keywords)
        layer.self_attn = nn.Sequential(original_attn, tsv_layer)
    elif component == "res":
        layers[str_layer] = LlamaDecoderLayerWrapper(layers[str_layer], tsv_layer)
    else:
        raise ValueError(f"Unknown component {component!r}. Choose 'res', 'mlp', or 'attn'.")

    logger.info("Steering installed: layer=%d component=%s strength=%s", str_layer, component, alpha)
