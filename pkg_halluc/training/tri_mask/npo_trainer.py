# Tri-mask training
import torch
import torch.nn.functional as F

from transformers import Trainer


class NPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        beta: float = 0.1,
        lambda_retain: float = 5,
        lambda_forget: float = 0.5,
        lambda_eos: float = 1.0, 
        vocab_size: int = None,
        temperature: float = 2,
        alpha: float = 0.7,
        gamma: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.lambda_retain = lambda_retain
        self.lambda_forget = lambda_forget
        self.lambda_eos = lambda_eos  
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        outputs = model(**model_inputs)
        logits = outputs.logits

        labels = inputs["labels"]
        tri_mask = inputs["tri_mask"]
        attn = inputs["attention_mask"]

        logits = logits[:, :-1, :]  
        labels = labels[:, 1:]  
        tri_mask = tri_mask[:, 1:]
        attn = attn[:, 1:]

        if self.vocab_size is not None:
            bad = (labels != -100) & ((labels < 0) | (labels >= self.vocab_size))
            if bad.any():
                # Lấy 1 id lỗi để báo cụ thể
                offending = labels[bad][0].item()
                raise RuntimeError(
                    f"Label id {offending} vượt ngoài phạm vi vocab_size={self.vocab_size}. "
                )

        V = logits.size(-1)
        ce_per_tok = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(
            labels
        )  

        valid = (labels != -100) & (attn == 1)
        retain_mask = valid & (tri_mask == 1)
        forget_mask = valid & (tri_mask == 2)

        eos_token_id = getattr(self.processing_class, "eos_token_id", None)

        if eos_token_id is not None:
            retain_mask_no_eos = retain_mask & (labels != eos_token_id)
            eos_mask = valid & (labels == eos_token_id) & (tri_mask == 1)
        else:
            retain_mask_no_eos = retain_mask
            eos_mask = None

        if retain_mask_no_eos.any():
            L_retain = ce_per_tok[retain_mask_no_eos].mean()
        else:
            L_retain = (logits[0, 0, 0] * 0.0).squeeze()

        if forget_mask.any():
            forget_logits = logits[forget_mask] 
            forget_labels = labels[forget_mask]  

            log_probs = F.log_softmax(forget_logits, dim=-1) 
            model_log_probs = torch.gather(
                log_probs, dim=-1, index=forget_labels.unsqueeze(-1)
            ).squeeze(
                -1
            )  

            core_term = F.softplus(self.beta * model_log_probs)

            ref_logits = forget_logits / self.temperature  
            ref_probs = F.softmax(ref_logits, dim=-1) 

            kl_div = F.kl_div(log_probs, ref_probs, reduction="none").sum(
                dim=-1
            )  

        
            L_forget = self.alpha * core_term.mean() + self.gamma * kl_div.mean()

        
            forget_confidence = torch.exp(model_log_probs).mean()
            forget_diversity = kl_div.mean()
        else:
            L_forget = (logits[0, 0, 0] * 0.0).squeeze()
            forget_confidence = (logits[0, 0, 0] * 0.0).squeeze()
            forget_diversity = (logits[0, 0, 0] * 0.0).squeeze()

        if eos_mask is not None and eos_mask.any():
            L_eos = ce_per_tok[eos_mask].mean()
        else:
            L_eos = (logits[0, 0, 0] * 0.0).squeeze()

        loss = (
            self.lambda_retain * L_retain
            + self.lambda_forget * L_forget
            + self.lambda_eos * L_eos
        )

        step = int(self.state.global_step)
        # early stopping
        if model.training and step > 0 and (step % max(1, self.args.logging_steps) == 0):
            logs = {
                "loss_retain": float(L_retain.detach().cpu()),
                "loss_forget": float(L_forget.detach().cpu()),
                "loss_eos": float(L_eos.detach().cpu()),
                "forget_confidence": float(forget_confidence.detach().cpu()),
                "forget_diversity": float(forget_diversity.detach().cpu()),
            }
            self.log(logs)
        return (loss, outputs) if return_outputs else loss
