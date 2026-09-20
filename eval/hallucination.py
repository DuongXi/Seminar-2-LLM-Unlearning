#!/usr/bin/env python3
"""
Package hallucination evaluation — single-file version.

Runs two queries against a model:
  Mode 1 (code -> packages): given a code snippet, name the packages it needs.
  Mode 2 (problem -> packages): given a problem statement, recommend packages.

Then compares each named package against a PyPI whitelist to separate valid
recommendations from hallucinations, and writes per-sample results plus a
FINAL_RESULTS.csv summary.

The only input is --master_file, which must be a JSONL with two columns:

    {"Prompts": "...", "Answers": "..."}

  - Prompts  : the coding problem (used by mode 2)
  - Answers  : the code snippet    (used by mode 1)

Usage:
  python package_hallucination.py \
      --model_path  /content/models/deepseek-ai/deepseek-coder-1.3b-instruct \
      --data_path   /content/data/DeepSeek_1B \
      --master_file /content/data/DeepSeek_1B/train_test_split/master_train.json \
      [--reasoning_model] [--log_level verbose]
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from typing import List

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model import load_model

# ─────────────────────────────────────────────────────────────────────────────
# Filtering word lists (loaded from eval.json)
# ─────────────────────────────────────────────────────────────────────────────
def _load_words(source: str, key: str):
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)[key]

DELETE_WORDS       = _load_words("./eval.json", "delete")
GENERIC_FRAMEWORKS = _load_words("./eval.json", "frameworks")
LEGIT_PKGS         = _load_words("./eval.json", "legitimate_packages")
FUNC_PREFIXES      = _load_words("./eval.json", "func_prefixes")


# ─────────────────────────────────────────────────────────────────────────────
# Query prefixes
# ─────────────────────────────────────────────────────────────────────────────
MODE1_PREFIXES = [
    "You are a coding assistant that determines Python packages necessary to execute code. "
    ,"Respond with only a list of Python packages, separated by commas and no additional text or formatting. "
    ,"Your response must begin with the name of a Python package."
    ,"Which Python packages are required to run this code:"
]

MODE2_PREFIXES = [
    "You are a coding assistant that recommends Python packages that would be helpful to solve given problems. "
    ,"Respond with only a list of Python packages, separated by commas and no additional text or formatting. "
    ,"Your response must begin with the name of a Python package."
    ,"What Python packages would be useful in solving the following coding problem:"
]


def get_random_prefix(mode: int) -> str:
    return random.choice(MODE1_PREFIXES if mode == 1 else MODE2_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Text / package extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_final_response(text: str) -> str:
    """Strip reasoning-model preamble up to and including </think>."""
    end = text.find("</think>")
    return text[end + len("</think>"):].strip() if end != -1 else text


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


def normalize_pip(name):
    if pd.isnull(name):
        return name
    if not isinstance(name, str):
        name = str(name)
    name = re.sub(r"[()'\"]", "", name)
    return re.sub(r"[-_.]+", "-", name).strip(' "`.-').lower()


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


def parse_pip_install(text):
    if not isinstance(text, (str, bytes)):
        return []
    matches = re.findall(r"pip\s+install\s+(?P<package_name>\S+)", text)
    return [m for m in matches if not m.startswith("-")] or []


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


def extract_and_clean_packages(package_string: str,
                               detect_loops: bool = True) -> List[str]:
    """Parse a model response into a list of normalized package names."""
    package_string = strip_code_blocks(package_string)

    for marker in (r"\bExplanation:", r"\bNote:", r"\bPlease note",
                   r"\bImportant:", r"\bAdditional notes:"):
        m = re.search(marker, package_string, re.IGNORECASE)
        if m:
            package_string = package_string[:m.start()]
            break

    if detect_loops:
        loop_start = detect_repetitive_loop(package_string)
        if loop_start > 0:
            package_string = package_string[:loop_start]

    # Numbered-list priority
    numbered = re.findall(r"(\d+[\.\)])\s*[`']?([a-zA-Z0-9\-_\.]+)[`']?",
                          package_string)
    from_numbered = False
    if numbered and len(numbered) >= 2:
        numbered_pkgs = []
        for number, pkg in numbered:
            pkg = pkg.strip()
            if not pkg or len(pkg) <= 2:
                continue
            line_pat = (re.escape(number) + r"\s*[`']?" + re.escape(pkg) +
                        r"[`']?([^\n]*)")
            line_match = re.search(line_pat, package_string)
            if line_match:
                full = line_match.group(0)
                if any(re.search(p, full, re.IGNORECASE) for p in
                       (r"\bfrom\b", r"\bclass\b", r"\bfunction\b",
                        r"\bdefined in\b", r"\(defined")):
                    continue
            numbered_pkgs.append(pkg)
        if len(numbered_pkgs) >= 2:
            parts = numbered_pkgs
            from_numbered = True

    if not from_numbered:
        text = re.sub(r"```[a-z]*\n?", "", package_string)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\\n\d+\.", ",", text)
        text = re.sub(r"\\n", ",", text)
        text = re.sub(r"\n\d+\.", ",", text)
        text = re.sub(r"\n", ",", text)
        text = re.sub(r"`([a-zA-Z0-9\-_\.]+)`", r",\1,", text)
        text = re.sub(r"'([a-zA-Z0-9\-_\.]+)'", r",\1,", text)
        parts = text.split(",")

    cleaned = []
    for part in parts:
        pkg = part.strip()
        if not pkg:
            continue
        pkg = pkg.strip(".,;:!?'\"()[]{}")
        pkg = re.sub(r"^[-*]\s*", "", pkg)
        pkg = re.sub(r"^\d+[\.\)]\s*", "", pkg)
        pkg = pkg.replace(" and ", " ").replace(" or ", " ")
        words = pkg.split()
        if not words:
            continue
        pkg = words[0]
        pkg = re.sub(r'[:"\'()\[\]]', "", pkg)
        pkg = re.sub(r"[^a-zA-Z0-9\-_\.]", "", pkg)

        if not pkg or len(pkg) <= 2 or pkg.isdigit() or pkg.lower() in DELETE_WORDS:
            continue
        if pkg.lower() in GENERIC_FRAMEWORKS and "-" not in pkg:
            continue

        if "." in pkg:
            if pkg.count(".") >= 2:
                continue
            if any(p.lower() in {"contrib", "instrumentation", "config"}
                   for p in pkg.split(".")):
                continue

        if pkg.lower() not in LEGIT_PKGS:
            var_pat = (r"^.*_(string|data|input|output|result|value|text|file|"
                       r"path|name|var|obj|object|item|list|dict|array|buffer|"
                       r"stream|content|response|request|str|num|int|float|id|"
                       r"idx|key|val|ptr|ref|temp|tmp|flag|bool|char|byte)s?$")
            if re.match(var_pat, pkg, re.IGNORECASE):
                continue
            if any(pkg.lower().startswith(p) and len(pkg) > len(p)
                   for p in FUNC_PREFIXES):
                continue
            if pkg.endswith("_") or pkg.endswith("."):
                continue

        cleaned.append(pkg)

    # Quality gate
    has_structure = (
        "`" in package_string
        or bool(re.search(r"(\n|^)\s*\d+[\.\)]\s+", package_string))
        or bool(re.search(r"(\n|^)\s*[-*]\s+", package_string))
    )
    period_count = package_string.count(".")
    comma_count = package_string.count(",")
    if comma_count >= 1 and period_count < comma_count // 2 + 1:
        has_structure = True

    is_excessive_prose = period_count > comma_count * 2 and period_count > 10
    appears_truncated = (
        any(s in package_string for s in
            ("This command", "Here is an example", "You can install"))
        and not package_string.strip().endswith((".", "!", "?", "`"))
    )
    if not has_structure and (len(package_string) >= 2000
                              or is_excessive_prose or appears_truncated):
        return []

    seen, result = set(), []
    for pkg in cleaned:
        p = normalize_python_package(pkg)
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


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


def check_pips(pip_list, pip_names):
    in_set, not_in_set = [], []
    trans = str.maketrans("", "", "()[]`")
    ver_pat = re.compile(r"([^=<>!~]+)([=<>!~]{1,2}[\d\.]+)?")
    for item in pip_list or []:
        text = item.translate(trans)
        if re.search(r"[+@:\"',{}/\*]", text):
            continue
        for part in text.split():
            if part.startswith("--"):
                continue
            m = ver_pat.match(part)
            if not m:
                continue
            name = normalize_python_package(m.group(1).strip())
            if name.startswith("--") or "requirements" in name:
                continue
            (in_set if name in pip_names else not_in_set).append(name)
    return in_set, not_in_set


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(
        lambda x: x.replace("\x00", "").replace("\r\n", "\n")
        if isinstance(x, str) else x
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_packages(mode, master_file, outfile, 
                      tokenizer:AutoTokenizer, 
                      model:AutoModelForCausalLM,
                      language="Python", is_reasoning_model=False):
    df = pd.read_json(master_file, lines=True)

    if mode == 1:
        if "Answers" not in df.columns:
            raise ValueError(f"{master_file} missing 'Answers' column for mode 1. "
                             f"Found: {df.columns.tolist()}")
        samples = df["Answers"].astype(str).tolist()
    else:
        if "Prompts" not in df.columns:
            raise ValueError(f"{master_file} missing 'Prompts' column for mode 2. "
                             f"Found: {df.columns.tolist()}")
        samples = df["Prompts"].astype(str).tolist()

    with open(outfile, "w", encoding="utf-8") as output:
        for sample in tqdm(samples, desc=f"Mode {mode}", unit="sample"):
            prefix = get_random_prefix(mode)
            messages = [{"role": "user", "content": prefix + sample}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

            response = tokenizer.decode(
                outputs.sequences[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            if is_reasoning_model:
                response = extract_final_response(response)

            json.dump({
                "prefix": prefix,
                "input": sample,
                "full_prompt": prefix + sample,
                "response": response,
            }, output)
            output.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
def package_search_python(df, pypi_set, fp_set):
    df["Test_1"] = df["Test_1"].astype(str).apply(extract_and_clean_packages)
    df["Test_2"] = df["Test_2"].astype(str).apply(extract_and_clean_packages)
    df["Test_1"] = df["Test_1"].apply(delete_dupes_and_empty)
    df["Test_2"] = df["Test_2"].apply(delete_dupes_and_empty)

    df[["valid_1", "hallucinated_1"]] = (
        df["Test_1"].apply(lambda x: check_packages(x, pypi_set, fp_set))
        .apply(pd.Series)
    )
    df[["valid_2", "hallucinated_2"]] = (
        df["Test_2"].apply(lambda x: check_packages(x, pypi_set, fp_set))
        .apply(pd.Series)
    )
    return df


def pip_numbers(df, pypi_set):
    df["pip"] = df["Answers"].apply(parse_pip_install)
    df["pip"] = df["pip"].apply(
        lambda x: [normalize_pip(e) for e in x]
    )
    df[["pip_valid", "pip_hallucinated"]] = (
        df["pip"].apply(lambda x: check_pips(x, pypi_set)).apply(pd.Series)
    )
    df["pip_hallucinated"] = df["pip_hallucinated"].apply(delete_dupes_and_empty)
    return df


def sum_columns(df, index_name):
    return pd.DataFrame({
        "valid_1":          [df["valid_1"].apply(len).sum()],
        "hallucinated_1":   [df["hallucinated_1"].apply(len).sum()],
        "valid_2":          [df["valid_2"].apply(len).sum()],
        "hallucinated_2":   [df["hallucinated_2"].apply(len).sum()],
        "pip_valid":        [df["pip_valid"].apply(len).sum()],
        "pip_hallucinated": [df["pip_hallucinated"].apply(len).sum()],
    }, index=[index_name])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",       
                        required=True)
    parser.add_argument("--data_path",       
                        required=True, help="directory containing pypi_package_names.csv and false_positive_packages.csv")
    parser.add_argument("--master_file",
                        required=True, help="JSONL with columns Prompts and Answers")
    parser.add_argument("--log_level", 
                        default="verbose", help="'off' to disable logging")
    parser.add_argument("--reasoning_model", 
                        action="store_true", help="strip </think> preamble from outputs")
    args = parser.parse_args()

    if args.log_level != "off":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    save_path = os.path.join(args.model_path, "package_hallucination_results")
    os.makedirs(save_path, exist_ok=True)

    logging.info("Loading model from %s", args.model_path)
    tokenizer, model = load_model(model_path=args.model_path,
                                  device_map="cuda",
                                  padding_side="left")

    # ── Phase 1: query the model ─────────────────────────────────────────────
    packages_1 = os.path.join(save_path, "packages_1.json")
    packages_2 = os.path.join(save_path, "packages_2.json")

    logging.info("Query 1: code -> packages")
    generate_packages(1, args.master_file, packages_1, tokenizer, model,
                      is_reasoning_model=args.reasoning_model)

    logging.info("Query 2: problem -> packages")
    generate_packages(2, args.master_file, packages_2, tokenizer, model,
                      is_reasoning_model=args.reasoning_model)

    # ── Phase 2: merge + analyze ─────────────────────────────────────────────
    logging.info("Merging responses")
    master  = pd.read_json(args.master_file, lines=True)
    resp_1  = pd.read_json(packages_1, lines=True)[["response"]].rename(
        columns={"response": "Test_1"})
    resp_2  = pd.read_json(packages_2, lines=True)[["response"]].rename(
        columns={"response": "Test_2"})
    merged  = pd.concat([master, resp_1, resp_2], axis=1)
    merged  = _sanitize_df(merged)

    logging.info("Loading PyPI whitelist")
    pypi = pd.read_csv(os.path.join(args.data_path, "pypi_package_names.csv"),
                       header=None)
    pypi[0] = pypi[0].apply(normalize_python_package)
    pypi_set = set(pypi[0])

    fp = pd.read_csv(os.path.join(args.data_path, "false_positive_packages.csv"),
                     header=None)
    fp_set = set(fp[1])

    logging.info("Scoring")
    scored = package_search_python(merged, pypi_set, fp_set)
    scored = pip_numbers(scored, pypi_set)

    out_csv = os.path.join(save_path, "results.csv")
    scored.to_csv(out_csv, index=False, encoding="utf-8",
                  quoting=csv.QUOTE_ALL, lineterminator="\n")

    totals = sum_columns(scored, "run")
    totals.loc["Totals"] = totals.loc["run"]
    totals.to_csv(os.path.join(save_path, "FINAL_RESULTS.csv"))

    scored.drop(["Prompts", "Answers", "Test_1", "Test_2"],
                axis=1, inplace=True, errors="ignore")
    scored.to_csv(os.path.join(save_path, "PACKAGE_NAMES.csv"), index=False)

    logging.info("Done. Results in %s", save_path)
    logging.info("Totals:\n%s", totals.to_string())


if __name__ == "__main__":
    main()