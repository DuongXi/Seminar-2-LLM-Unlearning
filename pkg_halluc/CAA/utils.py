import torch

def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    mask = position_ids >= from_id
    mask = mask.unsqueeze(-1)

    matrix += mask.float() * vector
    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_transformer_layers(model):
    candidates = (
        model,
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(getattr(model, "model", None), "decoder", None),
    )
    for candidate in candidates:
        for attribute in ("layers", "h"):
            layers = getattr(candidate, attribute, None) if candidate is not None else None
            if layers is not None:
                return layers
    raise ValueError(f"Could not locate transformer layers for {type(model).__name__}")