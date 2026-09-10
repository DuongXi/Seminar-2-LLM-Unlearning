import json
from datasets import Dataset

def _iter_objs(text):
    """
    Yield JSON objects from JSONL, a JSON array, or concatenated objects.
    """
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(text, i)
        yield obj
        i = j

def load_split(path, mode=None):
    """
    Load
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = list(_iter_objs(text))

    if mode is not None:
        records = [r for r in records if r.get("mode") == mode]

    keep = {"input_ids", "attention_mask", "labels"}
    records = [{k: v for k, v in r.items() if k in keep} for r in records]

    if not records:
        raise ValueError(f"{path}: no records (mode={mode}).")
    if not records[0]:
        raise ValueError(f"{path}: records lack keys {keep}. "
                         f"Got: {list(records[0].keys())}")

    return Dataset.from_list(records)