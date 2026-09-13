# Extracted verbatim from `train_utils.py` in the TSV repo
# (https://github.com/deeplearning-wisc/tsv, Apache-2.0 -- see ./LICENSE).
# Only this one function was pulled out; the rest of train_utils.py
# (Optimal Transport pseudo-labeling, centroid EMA updates, TSV's training
# loop) is TSV-specific and not included here -- see ../NOTES.md for why.
#
# Not adapted or modified in any way yet -- see ../NOTES.md "Known issues /
# adaptation needed before use" before wiring this into the pipeline.

import torch


def get_last_non_padded_token_rep(hidden_states, attention_mask):
    """
    Get the last non-padded token's representation for each sequence in the batch.
    """
    # Find the length of each sequence by summing the attention mask (1 for real tokens, 0 for padding)
    lengths = attention_mask.squeeze().sum(dim=1).long()

    # Index the last non-padded token for each sequence
    batch_size, max_seq_len, hidden_size = hidden_states.size()
    last_token_reps = torch.stack([hidden_states[i, lengths[i]-1, :] for i in range(batch_size)])

    return last_token_reps
