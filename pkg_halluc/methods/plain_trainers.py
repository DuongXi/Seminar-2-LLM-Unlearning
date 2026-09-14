"""GA-plain / NPO-plain -- Gradient Ascent and NPO without AU's tri-mask.

Loss math mirrors AU's formulas (GA: negative mean CE on forget; NPO:
softplus(beta * log p) + KL to a temperature-scaled self, reference-free)
so only the retain/forget granularity differs, not the loss shape. No
separate EOS term (AU's trainers give EOS its own weight)
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import Trainer


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable 0.0, for when a batch has no retain (or forget) examples."""
    return (logits[0, 0, 0] * 0.0).squeeze()


def _row_masks(split_type: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    retain_row = torch.tensor([s == "retain" for s in split_type], device=device).unsqueeze(1)
    forget_row = torch.tensor([s == "forget" for s in split_type], device=device).unsqueeze(1)
    return retain_row, forget_row


def _check_labels_in_vocab(labels: torch.Tensor, vocab_size: int | None) -> None:
    if vocab_size is None:
        return
    bad = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
    if bad.any():
        offending = labels[bad][0].item()
        raise RuntimeError(
            f"Label id {offending} is out of range for vocab_size={vocab_size}. "
            "The dataset was likely tokenized with a different tokenizer than "
            "the model being trained."
        )


class GAPlainTrainer(Trainer):
    """Gradient Ascent, no tri-mask: descend CE on retain, ascend (negative CE) on forget."""

    def __init__(
        self,
        *args: Any,
        lambda_retain: float = 1.0,
        lambda_forget: float = 0.5,
        vocab_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_retain = lambda_retain
        self.lambda_forget = lambda_forget
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs: Any):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        outputs = model(**model_inputs)
        logits = outputs.logits

        labels = inputs["labels"]
        attn = inputs["attention_mask"]
        split_type = inputs["split_type"]  # list[str], one per example

        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        attn = attn[:, 1:]

        _check_labels_in_vocab(labels, self.vocab_size)

        V = logits.size(-1)
        ce_per_tok = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels)

        valid = (labels != -100) & (attn == 1)
        retain_row, forget_row = _row_masks(split_type, labels.device)
        retain_mask = valid & retain_row
        forget_mask = valid & forget_row

        L_retain = ce_per_tok[retain_mask].mean() if retain_mask.any() else _zero_like(logits)
        L_forget = -ce_per_tok[forget_mask].mean() if forget_mask.any() else _zero_like(logits)

        loss = self.lambda_retain * L_retain + self.lambda_forget * L_forget

        step = int(self.state.global_step)
        if model.training and step > 0 and (step % max(1, self.args.logging_steps) == 0):
            self.log(
                {
                    "loss_retain": float(L_retain.detach().cpu()),
                    "loss_forget": float(L_forget.detach().cpu()),
                    "loss_total": float(loss.detach().cpu()),
                }
            )
        return (loss, outputs) if return_outputs else loss


class NPOPlainTrainer(Trainer):
    """NPO, no tri-mask: standard CE on retain; softplus(beta*logp) + KL-to-self on forget.

    Reference-free, same as AU: the "reference" distribution is the model's
    own temperature-scaled softmax, not a separate frozen checkpoint.
    """

    def __init__(
        self,
        *args: Any,
        beta: float = 0.1,
        lambda_retain: float = 1.0,
        lambda_forget: float = 0.5,
        temperature: float = 2.0,
        alpha: float = 0.7,
        gamma: float = 0.3,
        vocab_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.lambda_retain = lambda_retain
        self.lambda_forget = lambda_forget
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs: Any):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        outputs = model(**model_inputs)
        logits = outputs.logits

        labels = inputs["labels"]
        attn = inputs["attention_mask"]
        split_type = inputs["split_type"]

        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        attn = attn[:, 1:]

        _check_labels_in_vocab(labels, self.vocab_size)

        V = logits.size(-1)
        ce_per_tok = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels)

        valid = (labels != -100) & (attn == 1)
        retain_row, forget_row = _row_masks(split_type, labels.device)
        retain_mask = valid & retain_row
        forget_mask = valid & forget_row

        L_retain = ce_per_tok[retain_mask].mean() if retain_mask.any() else _zero_like(logits)

        if forget_mask.any():
            forget_logits = logits[forget_mask]  # [N, V]
            forget_labels = labels[forget_mask]  # [N]

            log_probs = F.log_softmax(forget_logits, dim=-1)
            model_log_probs = torch.gather(
                log_probs, dim=-1, index=forget_labels.unsqueeze(-1)
            ).squeeze(-1)

            core_term = F.softplus(self.beta * model_log_probs)

            ref_logits = forget_logits / self.temperature
            ref_probs = F.softmax(ref_logits, dim=-1)
            kl_div = F.kl_div(log_probs, ref_probs, reduction="none").sum(dim=-1)

            L_forget = self.alpha * core_term.mean() + self.gamma * kl_div.mean()
            forget_confidence = torch.exp(model_log_probs).mean()
            forget_diversity = kl_div.mean()
        else:
            L_forget = _zero_like(logits)
            forget_confidence = _zero_like(logits)
            forget_diversity = _zero_like(logits)

        loss = self.lambda_retain * L_retain + self.lambda_forget * L_forget

        step = int(self.state.global_step)
        if model.training and step > 0 and (step % max(1, self.args.logging_steps) == 0):
            self.log(
                {
                    "loss_retain": float(L_retain.detach().cpu()),
                    "loss_forget": float(L_forget.detach().cpu()),
                    "forget_confidence": float(forget_confidence.detach().cpu()),
                    "forget_diversity": float(forget_diversity.detach().cpu()),
                    "loss_total": float(loss.detach().cpu()),
                }
            )
        return (loss, outputs) if return_outputs else loss
