#!/usr/bin/env python3
"""
Package hallucination evaluation — schema-driven version.

Reads a JSONL where each row carries its own prompting fields:

    {"mode": 1 | 2,
     "system_prompt": "...",
     "user_prompt":   "...",
     "completion":    "<optional, will be overwritten>"}

For each row:
  1. Builds a chat prompt from (system_prompt, user_prompt).
  2. Generates a fresh completion with the model.
  3. Extracts package names from the completion.
  4. Splits them into valid_packages / hallucinated_packages against a
     PyPI whitelist.

Writes per-sample results and a FINAL_RESULTS.csv summary.
"""

import argparse
import csv
import json
import logging
import os
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
# Message construction
# ─────────────────────────────────────────────────────────────────────────────
def build_messages(system_prompt: str, user_prompt: str):
    """Forward the schema's system_prompt and user_prompt as chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


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


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda x: x.replace("\x00", "").replace("\r\n", "\n")
            if isinstance(x, str) else x
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_completions(master_file, outfile,
                         tokenizer: AutoTokenizer,
                         model: AutoModelForCausalLM,
                         is_reasoning_model=False):
    """One completion per row, using that row's own system + user prompt."""
    df = pd.read_json(master_file, lines=True)

    required = {"mode", "system_prompt", "user_prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{master_file} missing columns: {sorted(missing)}")

    with open(outfile, "w", encoding="utf-8") as output:
        for i, row in tqdm(df.iterrows(), total=len(df),
                           desc="Generating", unit="sample"):
            mode          = int(row["mode"])
            system_prompt = str(row.get("system_prompt", "") or "")
            user_prompt   = str(row.get("user_prompt",   "") or "")

            messages = build_messages(system_prompt, user_prompt)

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
                "row_index":     int(i),
                "mode":          mode,
                "system_prompt": system_prompt,
                "user_prompt":   user_prompt,
                "completion":    response,
            }, output)
            output.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
def score_completions(df, pypi_set, fp_set):
    """Extract packages from `completion` and write them into the mode's slots."""
    n = len(df)
    for col, values in {
        "Test_1":         [""] * n,
        "Test_2":         [""] * n,
        "valid_1":        [[] for _ in range(n)],
        "hallucinated_1": [[] for _ in range(n)],
        "valid_2":        [[] for _ in range(n)],
        "hallucinated_2": [[] for _ in range(n)],
    }.items():
        if col not in df.columns:
            df[col] = pd.Series(values, index=df.index, dtype=object)

    for i, row in df.iterrows():
        mode       = int(row["mode"])
        completion = str(row.get("completion", "") or "")

        extracted = delete_dupes_and_empty(
            extract_and_clean_packages(completion)
        )
        valid, halluc = check_packages(extracted, pypi_set, fp_set)

        if mode == 1:
            df.at[i, "Test_1"]         = completion
            df.at[i, "valid_1"]        = valid
            df.at[i, "hallucinated_1"] = halluc
        elif mode == 2:
            df.at[i, "Test_2"]         = completion
            df.at[i, "valid_2"]        = valid
            df.at[i, "hallucinated_2"] = halluc
        else:
            logging.warning("row %s: unknown mode %r, skipping", i, mode)

    return df

def sum_columns(df, index_name):
    """Sum list lengths per column. Expects the valid_/hallucinated_ columns
    to hold Python lists (or list-like) per row."""
    def _len_sum(col):
        return df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

    return pd.DataFrame({
        "valid_1":          [_len_sum("valid_1")],
        "hallucinated_1":   [_len_sum("hallucinated_1")],
        "valid_2":          [_len_sum("valid_2")],
        "hallucinated_2":   [_len_sum("hallucinated_2")],
    }, index=[index_name])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--data_path",   required=True,
                        help="directory containing pypi_package_names.csv and "
                             "false_positive_packages.csv")
    parser.add_argument("--master_file", required=True,
                        help="JSONL with columns: mode, system_prompt, user_prompt")
    parser.add_argument("--log_level", default="verbose",
                        help="'off' to disable logging")
    parser.add_argument("--reasoning_model", action="store_true",
                        help="strip </think> preamble from outputs")
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

    # ── Phase 1: generate ────────────────────────────────────────────────
    completions_file = os.path.join(save_path, "completions.json")
    logging.info("Generating completions")
    generate_completions(args.master_file, completions_file,
                         tokenizer, model,
                         is_reasoning_model=args.reasoning_model)

    # ── Phase 2: merge + score ───────────────────────────────────────────
    logging.info("Merging + scoring")
    master = pd.read_json(args.master_file, lines=True)
    gen    = pd.read_json(completions_file, lines=True)

    # The input schema carries a `completion` column (and possibly pre-existing
    # `valid_packages` / `hallucinated_packages`).  Drop or rename them so the
    # concatenated frame has no duplicate column names.
    if "completion" in master.columns:
        master = master.rename(columns={"completion": "completion_original"})
    if "valid_packages" in master.columns:
        master = master.rename(
            columns={"valid_packages": "valid_packages_original"})
    if "hallucinated_packages" in master.columns:
        master = master.rename(
            columns={"hallucinated_packages": "hallucinated_packages_original"})

    merged = pd.concat(
        [master.reset_index(drop=True),
         gen[["completion"]].reset_index(drop=True)],
        axis=1,
    )

    # Belt-and-braces: drop any residual duplicate column names, keeping the
    # freshly generated one (last occurrence).
    merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]

    merged = _sanitize_df(merged)

    logging.info("Loading PyPI whitelist")
    pypi = pd.read_csv(os.path.join(args.data_path, "pypi_package_names.csv"),
                       header=None)
    pypi[0] = pypi[0].apply(normalize_python_package)
    pypi_set = set(pypi[0])

    fp = pd.read_csv(os.path.join(args.data_path, "false_positive_packages.csv"),
                     header=None)
    fp_set = set(fp[1])

    scored = score_completions(merged, pypi_set, fp_set)

    # ── Phase 3: write ───────────────────────────────────────────────────
    out_csv = os.path.join(save_path, "results.csv")
    scored.to_csv(out_csv, index=False, encoding="utf-8",
                  quoting=csv.QUOTE_ALL, lineterminator="\n")

    scored.drop(["Test_1", "Test_2"], axis=1, inplace=True, errors="ignore")
    scored.to_csv(os.path.join(save_path, "PACKAGE_NAMES.csv"), index=False)

    totals = sum_columns(scored, "run")
    totals.loc["Totals"] = totals.loc["run"]
    totals.to_csv(os.path.join(save_path, "FINAL_RESULTS.csv"))

    logging.info("Done. Results in %s", save_path)
    logging.info("Totals:\n%s", totals.to_string())


if __name__ == "__main__":
    main()