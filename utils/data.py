import json
import re

from typing import List
import pandas as pd

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

#
# Pkg hallu data utils
#
def _load_words(source: str, key: str):
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)[key]

# Message construction
def build_messages(system_prompt: str, user_prompt: str):
    """Forward the schema's system_prompt and user_prompt as chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

def normalize_python_package(name: str) -> str:
    """PEP 503 normalization: lowercase, collapse [-_.]+ to '-'."""
    if not name or not isinstance(name, str):
        return name
    name = re.sub(r"\d+\.\s*", "", name)
    name = re.sub(r"(?<=.)\n(?=.)", " ", name)
    name = re.sub(r"\n", "", name)
    name = re.sub(r"[-_.]+", "-", name)
    name = name.strip(" `.-_")
    return name.lower()


def delete_dupes_and_empty(packages):
    no_dupes = list(set(packages))
    no_dupes = [i for i in no_dupes if len(i) > 2]
    return [x for x in no_dupes if x]


def strip_code_blocks(text: str) -> str:
    """Remove ```...``` blocks (closed or unclosed) but keep pip install lines."""
    if not text or not isinstance(text, str):
        return text

    pip_installs = re.findall(r"pip\s+install\s+[^\n]+", text, re.IGNORECASE)

    cleaned = re.sub(r"```[\w]*\n.*?\n```", "\n", text, flags=re.DOTALL)
    unclosed = re.search(r"```[\w]*\n.*", cleaned, re.DOTALL)
    if unclosed:
        m = re.search(r"```", cleaned)
        if m:
            cleaned = cleaned[:m.start()]

    cleaned = re.sub(r"`[^`\n]*\n(?:(?!\d+[\.\)]\s)[^`])*`", "\n",
                     cleaned, flags=re.DOTALL)

    if pip_installs:
        cleaned = cleaned + "\n" + "\n".join(pip_installs)
    return cleaned


def detect_repetitive_loop(text: str, min_cycle_items: int = 3,
                           min_cycles: int = 3) -> int:
    """Return char position of the second occurrence of a repeating cycle, or 0."""
    lines = text.split("\n")
    normalized = []
    for line in lines:
        clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if clean:
            normalized.append(clean)

    if len(normalized) < min_cycle_items * min_cycles:
        return 0

    for cycle_len in range(min_cycle_items,
                           min(21, len(normalized) // min_cycles + 1)):
        pattern = normalized[:cycle_len]
        reps = 1
        for i in range(cycle_len, len(normalized), cycle_len):
            if normalized[i:i + cycle_len] == pattern:
                reps += 1
            else:
                break
        if reps >= min_cycles:
            char_pos, counted = 0, 0
            for line in lines:
                if counted >= cycle_len:
                    return char_pos
                if re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip():
                    counted += 1
                char_pos += len(line) + 1
            return char_pos
    return 0


# Text / package extraction helpers
def extract_final_response(text: str) -> str:
    """Strip reasoning-model preamble up to and including </think>."""
    end = text.find("</think>")
    return text[end + len("</think>"):].strip() if end != -1 else text


def check_packages(package_list, package_names, false_positives):
    in_set, not_in_set = [], []
    for item in package_list:
        if " " in item or item in ("None", "nan"):
            continue
        if item in package_names:
            in_set.append(item)
        elif item not in false_positives:
            not_in_set.append(item)
    return in_set, not_in_set


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda x: x.replace("\x00", "").replace("\r\n", "\n")
            if isinstance(x, str) else x
        )
    return df