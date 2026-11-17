import pandas as pd
import csv

import os
import logging
import json
import re
import random

from typing import List

from tqdm import tqdm

from utils.model import load_model

#
# Read JSON for lists
#
def load_words(source:str, key:str):
    with open(source, "r", encoding="utf-8") as file:
        data = json.load(file)

    words_list = data[key]
    return words_list

DELETE_WORDS = load_words("./eval.json", "delete")
GENERIC_FRAMEWORKS = load_words("./eval.json", "frameworks")
LEGIT_PKGS = load_words("./eval.json", "legitimate_packages")
FUNC_PREFIXES = load_words("./eval.json", "func_prefixes")


#
# PREFIXES reused from LLM Ghostbuster 
#
MODE1_PREFIXES = [
    "What Python packages are needed to run this code: ",
    "Which pip packages does this code require: ",
    "Name the Python packages this code depends on: ",
    "What packages would I need to pip install to run this: ",
    "Identify the Python package names required by this code: ",
    "What are the package dependencies for this code: ",
    "Which Python packages must be installed to execute this code: ",
    "List the package names needed to run the following code: ",
    "What packages does this code need installed to work: ",
    "Tell me which Python packages this code requires: ",
]

MODE2_PREFIXES = [
    "Which Python packages would help solve this coding problem: ",
    "What Python packages could I use to tackle this problem: ",
    "Name some Python packages that would be relevant for solving this: ",
    "What packages should I consider using for this coding task: ",
    "Which Python packages would be appropriate for this problem: ",
    "What Python packages would you recommend for solving this: ",
    "Suggest Python packages that could help with this coding challenge: ",
    "What packages would be beneficial for implementing a solution to this: ",
    "Which Python packages are well-suited for this problem: ",
    "What Python packages would assist in solving the following task: ",
]

def get_random_prefix(mode: int) -> str:
    if mode == 1:
        return random.choice(MODE1_PREFIXES)
    elif mode == 2:
        return random.choice(MODE2_PREFIXES)


#
# Package dataset processes
#
def delete_dupes_and_empty(packages):
    no_dupes = list(set(packages))
    no_dupes = [i for i in no_dupes if len(i) > 2]
    return [x for x in no_dupes if x]

def extract_final_response(text):
    """
    Extract the final response from reasoning model output.
    For models that use </think> to mark the end of reasoning.
    """
    # Find the position of </think>
    think_end = text.find("</think>")

    if think_end != -1:
        # Return everything after </think>, stripped of whitespace
        return text[think_end + len("</think>") :].strip()

    # If no </think> tag found, return the full text
    # (this handles cases where the model doesn't use reasoning for simple queries)
    return text

def normalize_pip(name):
    if pd.isnull(name):
        return name
    if not isinstance(name, str):
        name = str(name)

    name = re.sub(r"[()\'\"]", "", name)
    return re.sub(r"[-_.]+", "-", name).strip(' "`.-').lower()


def check_packages(package_list, package_names, false_positives):
    in_set = []
    not_in_set = []
    for item in package_list:
        if " " in item or item == "None" or item == "nan":
            continue
        if item in package_names:
            in_set.append(item)
        else:
            if item not in false_positives:
                not_in_set.append(item)

    return in_set, not_in_set


def detect_repetitive_loop(
    text: str, min_cycle_items: int = 3, min_cycles: int = 3
) -> int:
    """
    Detect if text contains repetitive N-item cycles that repeat M times.

    This handles model collapse where outputs get stuck repeating the same
    pattern over and over (e.g., packages 1-4 repeating 20 times).

    Args:
        text: Input text to check for loops
        min_cycle_items: Minimum items in repeating pattern (default 3)
        min_cycles: Minimum repetitions to confirm loop (default 3)

    Returns:
        Character position where SECOND cycle starts (to preserve first occurrence)
        Returns 0 if no loop detected
    """
    lines = text.split("\n")
    normalized_lines = []

    for line in lines:
        # Remove numbered list markers: "1. package" -> "package"
        clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        clean = clean.strip()
        if clean:
            normalized_lines.append(clean)

    if len(normalized_lines) < min_cycle_items * min_cycles:
        return 0

    # Try different cycle lengths
    for cycle_len in range(
        min_cycle_items, min(21, len(normalized_lines) // min_cycles + 1)
    ):
        pattern = normalized_lines[:cycle_len]

        # Check if this pattern repeats
        repetitions = 1
        for i in range(cycle_len, len(normalized_lines), cycle_len):
            chunk = normalized_lines[i : i + cycle_len]
            if chunk == pattern:
                repetitions += 1
            else:
                break

        if repetitions >= min_cycles:
            # Found a loop! Return character position of SECOND cycle
            lines_to_skip = cycle_len
            char_pos = 0
            lines_counted = 0

            for line in lines:
                if lines_counted >= lines_to_skip:
                    return char_pos
                clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
                if clean:
                    lines_counted += 1
                char_pos += len(line) + 1
            return char_pos
    return 0


def normalize_python_package(name: str) -> str:
    """
    Normalize Python package names according to PEP 503.

    In Python packaging, dashes, underscores, and periods are equivalent.
    This function standardizes them all to dashes and lowercases the name.

    Args:
        name: Package name to normalize

    Returns:
        Normalized package name (lowercase, with standardized separators)
    """
    if not name or not isinstance(name, str):
        return name
    # Remove numbered list markers if any remain
    name = re.sub(r"\d+\.\s*", "", name)
    # Handle newlines within names
    name = re.sub(r"(?<=.)\n(?=.)", " ", name)
    name = re.sub(r"\n", "", name)
    # Normalize separators: convert all dashes, underscores, dots to single dash
    name = re.sub(r"[-_.]+", "-", name)
    # Strip leading/trailing special characters and whitespace
    name = name.strip(" `.-_")
    # Convert to lowercase
    return name.lower()


def package_search_python(df, data_path):
    package_names = pd.read_csv(f"{data_path}/pypi_package_names.csv", header=None)
    package_names[0] = package_names[0].apply(normalize_python_package)
    package_names_set = set(package_names[0])
    false_positives = pd.read_csv(
        f"{data_path}/false_positive_packages.csv", header=None
    )
    false_positives_set = set(false_positives[1])

    df["Test_1"] = df["Test_1"].astype(str)
    df["Test_2"] = df["Test_2"].astype(str)

    df["Test_1"] = df["Test_1"].apply(extract_and_clean_packages)
    df["Test_2"] = df["Test_2"].apply(extract_and_clean_packages)

    df["Test_1"] = df["Test_1"].apply(delete_dupes_and_empty)
    df["Test_2"] = df["Test_2"].apply(delete_dupes_and_empty)

    df[["valid_1", "hallucinated_1"]] = (
        df["Test_1"]
        .apply(lambda x: check_packages(x, package_names_set, false_positives_set))
        .apply(pd.Series)
    )
    df[["valid_2", "hallucinated_2"]] = (
        df["Test_2"]
        .apply(lambda x: check_packages(x, package_names_set, false_positives_set))
        .apply(pd.Series)
    )

    return df


def strip_code_blocks(text: str) -> str:
    """
    Remove code blocks (```...```) from text, preserving pip install statements.

    Handles both:
    1. Properly closed code blocks: ```python\n...code...\n```
    2. Unclosed code blocks (token limit hit): ```python\n...code (no closing ticks)

    This prevents Python code identifiers (like nn.Linear, self.attention, etc.)
    from being incorrectly detected as package names.
    """
    if not text or not isinstance(text, str):
        return text

    # Extract all pip install statements first (to preserve them)
    pip_install_pattern = r"pip\s+install\s+[^\n]+"
    pip_installs = re.findall(pip_install_pattern, text, re.IGNORECASE)

    # Step 1: Remove all properly closed code blocks (triple backticks)
    code_block_pattern = r"```[\w]*\n.*?\n```"
    cleaned_text = re.sub(code_block_pattern, "\n", text, flags=re.DOTALL)

    # Step 2: Handle unclosed code blocks (token limit scenario)
    # If there's an opening ``` without a closing ```, remove from there to end
    unclosed_pattern = r"```[\w]*\n.*"
    if re.search(unclosed_pattern, cleaned_text, re.DOTALL):
        match = re.search(r"```", cleaned_text)
        if match:
            # Remove everything from the opening ``` to the end
            cleaned_text = cleaned_text[: match.start()]

    # Step 3: Handle inline code blocks spanning multiple lines
    inline_multiline_pattern = r"`[^`\n]*\n(?:(?!\d+[\.\)]\s)[^`])*`"
    cleaned_text = re.sub(inline_multiline_pattern, "\n", cleaned_text, flags=re.DOTALL)

    # Step 4: Re-add pip install statements
    if pip_installs:
        cleaned_text = cleaned_text + "\n" + "\n".join(pip_installs)

    return cleaned_text


def parse_pip_install(text):
    if not isinstance(text, (str, bytes)):
        return []
    matches = re.findall(r"pip\s+install\s+(?P<package_name>\S+)", text)
    packages = [match for match in matches if not match.startswith("-")]
    return packages if packages else []


def pip_numbers(df, data_path):
    df["pip"] = df["Answers"].apply(parse_pip_install)
    df["pip"] = df["pip"].apply(
        lambda x: [item for item in (normalize_pip(entry) for entry in x)]
    )

    pypi = pd.read_csv(f"{data_path}/pypi_package_names.csv", header=None)
    pypi[0] = pypi[0].apply(normalize_python_package)
    pips = set(pypi[0])

    df[["pip_valid", "pip_hallucinated"]] = (
        df["pip"].apply(lambda x: check_pips(x, pips)).apply(pd.Series)
    )
    df["pip_hallucinated"] = df["pip_hallucinated"].apply(delete_dupes_and_empty)

    return df


def extract_and_clean_packages(
    package_string: str, detect_loops: bool = True
) -> List[str]:
    """
    Parse and clean package names from comma-separated or formatted string.

    **ENHANCED**:
    1. Detects and removes repetitive loops (model collapse)
    2. Prioritizes extraction from numbered lists (avoids prose noise)
    3. Falls back to general extraction if no numbered list

    Format priority:
    1. Numbered lists: "1. package1\n2. package2" [HIGHEST PRIORITY]
    2. Backticks: "`package1`, `package2`"
    3. Bullet lists: "- package1\n- package2"
    4. Comma-separated: "package1, package2, package3"

    Args:
        package_string: String containing package names in various formats
        detect_loops: Whether to detect and remove repetitive loops (default True)

    Returns:
        List of cleaned, normalized package names (empty if unparseable format)
    """
    # ========================================================================
    # STEP 0: Truncate at explanation/note sections (NEW)
    # ========================================================================
    # Stop processing at prose explanation markers
    print(package_string)

    package_string = strip_code_blocks(package_string)

    print(package_string)

    prose_markers = [
        r"\bExplanation:",
        r"\bNote:",
        r"\bPlease note",
        r"\bImportant:",
        r"\bAdditional notes:",
    ]

    for marker in prose_markers:
        match = re.search(marker, package_string, re.IGNORECASE)
        if match:
            # Only keep text before the marker
            package_string = package_string[: match.start()]
            break

    # ========================================================================
    # STEP 1: Loop Detection
    # ========================================================================
    if detect_loops:
        loop_start = detect_repetitive_loop(package_string)
        if loop_start > 0:
            original_len = len(package_string)
            package_string = package_string[:loop_start]
            # logging.info(f"   Truncated from {original_len} to {loop_start} chars")

    # ========================================================================
    # STEP 2: Numbered List Priority (MODIFIED)
    # ========================================================================
    numbered_pattern = r"(\d+[\.\)])\s*[`\']?([a-zA-Z0-9\-_\.]+)[`\']?"
    numbered_matches = re.findall(numbered_pattern, package_string)

    from_numbered_list = False
    if numbered_matches and len(numbered_matches) >= 2:
        numbered_packages = []
        for number, pkg in numbered_matches:
            pkg = pkg.strip()
            if pkg and len(pkg) > 2:
                # Get the full line for this numbered item to check for patterns
                # Find the line containing this number
                line_pattern = (
                    re.escape(number)
                    + r"\s*[`\']?"
                    + re.escape(pkg)
                    + r"[`\']?([^\n]*)"
                )
                line_match = re.search(line_pattern, package_string)

                if line_match:
                    full_line = line_match.group(0)

                    # Skip if line contains descriptive patterns
                    skip_patterns = [
                        r"\bfrom\b",  # "Pipe from multiprocessing"
                        r"\bclass\b",  # "QueueData class"
                        r"\bfunction\b",  # "worker function"
                        r"\bdefined in\b",  # "defined in the code"
                        r"\(defined",  # "(defined in..."
                    ]

                    should_skip = False
                    for pattern in skip_patterns:
                        if re.search(pattern, full_line, re.IGNORECASE):
                            should_skip = True
                            break

                    if not should_skip:
                        numbered_packages.append(pkg)
                else:
                    # Couldn't find full line, keep the package
                    numbered_packages.append(pkg)

        if len(numbered_packages) >= 2:
            parts = numbered_packages
            from_numbered_list = True
        else:
            print(f"\n🔍 Extracted from numbered list:")
            print(numbered_packages)
            print("🔍 END\n")

    # ========================================================================
    # STEP 3: Fallback - General Extraction
    # ========================================================================
    if not from_numbered_list:
        text = re.sub(r"```[a-z]*\n?", "", package_string)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\\n\d+\.", ",", text)
        text = re.sub(r"\\n", ",", text)
        text = re.sub(r"\n\d+\.", ",", text)
        text = re.sub(r"\n", ",", text)
        text = re.sub(r"`([a-zA-Z0-9\-_\.]+)`", r",\1,", text)
        text = re.sub(r"'([a-zA-Z0-9\-_\.]+)'", r",\1,", text)
        parts = text.split(",")


    # ========================================================================
    # STEP 4: Word Filtering
    # ========================================================================


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
        if words:
            pkg = words[0]
        else:
            continue

        pkg = re.sub(r'[:"\'()\[\]]', "", pkg)
        pkg = re.sub(r"[^a-zA-Z0-9\-_\.]", "", pkg)

        if not pkg or len(pkg) <= 2 or pkg.isdigit() or pkg.lower() in DELETE_WORDS:
            continue

        if pkg.lower() in GENERIC_FRAMEWORKS and "-" not in pkg:
            continue

        if "." in pkg:
            dot_count = pkg.count(".")
            if dot_count >= 2:
                continue
            parts_split = pkg.split(".")
            if any(
                p.lower() in {"contrib", "instrumentation", "config"}
                for p in parts_split
            ):
                continue

        if pkg.lower() not in LEGIT_PKGS:
            var_pattern = r"^.*_(string|data|input|output|result|value|text|file|path|name|var|obj|object|item|list|dict|array|buffer|stream|content|response|request|str|num|int|float|id|idx|key|val|ptr|ref|temp|tmp|flag|bool|char|byte)s?$"
            if re.match(var_pattern, pkg, re.IGNORECASE):
                continue

            if any(
                pkg.lower().startswith(prefix) and len(pkg) > len(prefix)
                for prefix in FUNC_PREFIXES
            ):
                continue

            if pkg.endswith("_"):
                continue

            if pkg.endswith("."):
                continue

        cleaned.append(pkg)

    # ========================================================================
    # STEP 5: Quality Filtering
    # ========================================================================
    has_backticks = "`" in package_string
    has_numbered_list = bool(re.search(r"(\n|^)\s*\d+[\.\)]\s+", package_string))
    has_bullet_list = bool(re.search(r"(\n|^)\s*[-*]\s+", package_string))

    period_count = package_string.count(".")
    comma_count = package_string.count(",")
    is_comma_separated = comma_count >= 1 and period_count < comma_count // 2 + 1

    has_clear_structure = (
        has_backticks or has_numbered_list or has_bullet_list or is_comma_separated
    )

    is_excessive_prose = period_count > comma_count * 2 and period_count > 10
    has_reasonable_length = len(package_string) < 2000

    truncation_indicators = [
        "This command",
        "Here is an example",
        "You can install",
    ]
    appears_truncated = any(
        indicator in package_string for indicator in truncation_indicators
    ) and not package_string.strip().endswith((".", "!", "?", "`"))

    if not has_clear_structure and (
        not has_reasonable_length or is_excessive_prose or appears_truncated
    ):
        return []

    # ========================================================================
    # STEP 6: Normalization & Deduplication
    # ========================================================================
    seen = set()
    result = []
    for pkg in cleaned:
        pkg_normalized = normalize_python_package(pkg)

        if not pkg_normalized:
            continue

        if pkg_normalized not in seen:
            seen.add(pkg_normalized)
            result.append(pkg_normalized)

    return result

def check_pips(pip_list, pip_names):
    in_set = []
    not_in_set = []
    translation_table = str.maketrans("", "", "()[]`")
    version_pattern = re.compile(r"([^=<>!~]+)([=<>!~]{1,2}[\d\.]+)?")

    if pip_list:
        for item in pip_list:
            text = item.translate(translation_table)
            # text = re.sub(r"[+@:\"\',{}/\*]", "", text)
            if bool(re.search(r"[+@:\"\',{}/\*]", text)):
                continue
            for part in text.split():
                if part.startswith("--"):
                    continue
                match = version_pattern.match(part)
                if match:
                    # text = re.sub(r"\n", '', text)
                    text = match.group(1).strip()
                    text = normalize_python_package(text)
                    if text.startswith("--") or "requirements" in text:
                        continue
                    if text in pip_names:
                        in_set.append(text)
                    else:
                        not_in_set.append(text)

    return in_set, not_in_set


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # strip NULs and normalize newlines in string columns only
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(
        lambda x: (
            x.replace("\x00", "").replace("\r\n", "\n") if isinstance(x, str) else x
        )
    )
    return df
#
# Generate code
#
def generate_code(
        infile,
        outfile,
        model_path,
        temperature=0.7, top_k=20, top_p=0.9, batch_size=16, is_reasoning_model=False
        ):
    model, tokenizer = load_model(model_path, device_map="auto")

    with open(infile, "r") as file:
        df = pd.read_json(file, lines=True)
        prompts = [row[0] for index, row in df.iterrows()]

    results = []

    for i in tqdm(
        range(0, len(prompts), batch_size), 
        desc="Generating code", unit="batch"
        ):
        batch_prompts = prompts[i : i + batch_size]
        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]

        # Tokenize batch with padding
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        attention_mask = (inputs != tokenizer.pad_token_id).long()

        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            repetition_penalty=1,
        )

        # Decode each output in batch
        for j, output_seq in enumerate(outputs.sequences):
            prompt_length = inputs.shape[1]
            generated_code = tokenizer.decode(
                output_seq[prompt_length:], skip_special_tokens=True
            )
            if is_reasoning_model:
                generated_code = extract_final_response(generated_code)

            results.append(generated_code)

    # Write all results
    with open(outfile, "w", newline="", encoding="utf-8") as output:
        for result in results:
            json.dump(result, output)
            output.write("\n")

#
# Generate packages
#
def generate_packages(
    mode,
    infile,
    outfile,
    model_path,
    language="Python",
    is_reasoning_model=False,
):
    tokenizer, model = load_model(model_path=model_path, 
                                  device_map="cuda")

    with open(infile, "r") as file:
        df = pd.read_json(file, lines=True)
        code = []
        if mode == 1:
            for index, row in df.iterrows():
                code.append(row["Answers"])
        else:
            for index, row in df.iterrows():
                code.append(row[0])

    with open(outfile, "w", newline="", encoding="utf-8") as output:

        if mode == 1:
            prefix = f"Which {language} packages are required to run this code: "
        elif mode == 2:
            prefix = f"What {language} packages would be useful in solving the following coding problem: "

        for sample in tqdm(code, desc="Generating package names", unit="sample"):
            prefix = get_random_prefix(mode)

            messages = [
                {"role": "user", "content": prefix + sample},
            ]

            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            generated_code = tokenizer.decode(
                # outputs.sequences[0, inputs.shape[1] :], skip_special_tokens=True
                outputs.sequences[0, inputs["input_ids"].shape[1]:]
            )
            if is_reasoning_model:
                generated_code = extract_final_response(generated_code)
            json.dump(
                {
                    "prefix": prefix,  # the random prefix used
                    "input": sample,  # the code (mode 1) or prompt (mode 2)
                    "full_prompt": prefix + sample,  # what was actually sent to model
                    "response": generated_code,  # model's raw response
                },
                output,
            )
            output.write("\n")

#
# CP
#
def combine_code_and_prompt(infile1, infile2, outfile):
    prompts = pd.read_json(infile1, lines=True)
    code = pd.read_json(infile2, lines=True)

    # prompts = prompts.reset_index()
    master = pd.concat([prompts, code], axis=1, ignore_index=True)
    master.columns = ["Prompts", "Answers"]
    master.to_json(outfile, orient="records", lines=True)


def sum_columns(df, index_name, language):
    totals = pd.DataFrame(index=[f"{index_name}"])
    totals["valid_1"] = df["valid_1"].apply(len).sum()
    totals["hallucinated_1"] = df["hallucinated_1"].apply(len).sum()
    totals["valid_2"] = df["valid_2"].apply(len).sum()
    totals["hallucinated_2"] = df["hallucinated_2"].apply(len).sum()
    if language == "Python":
        totals["pip_valid"] = df["pip_valid"].apply(len).sum()
        totals["pip_hallucinated"] = df["pip_hallucinated"].apply(len).sum()
    elif language == "Javascript":
        totals["npm_valid"] = df["npm_valid"].apply(len).sum()
        totals["npm_hallucinated"] = df["npm_hallucinated"].apply(len).sum()

    return totals


def merge_prompts_and_packages(prompt_file, query1, query2):
    prompts = pd.read_json(prompt_file, lines=True)
    responses1 = pd.read_json(query1, lines=True)
    responses2 = pd.read_json(query2, lines=True)

    responses1 = responses1[["response"]]
    responses2 = responses2[["response"]]

    responses1.columns = ["Test_1"]
    responses2.columns = ["Test_2"]

    return pd.concat([prompts, responses1, responses2], axis=1)


def process_python_dataset(
    master_file, 
    save_path,
    data_path,
    language,
    package_file_1, 
    package_file_2, 
    result_file_prefix
    ):
    merged_results = merge_prompts_and_packages(
            os.path.join(save_path, master_file),
            os.path.join(save_path, package_file_1),
            os.path.join(save_path, package_file_2),
    )
    merged_results = _sanitize_df(merged_results)
    logging.info("Responses merged into dataframe")

    results = merged_results
    results_tested = package_search_python(results, data_path)
    results_final = pip_numbers(results_tested, data_path)

    out_csv = os.path.join(save_path, f"results.csv")
    results_final.to_csv(
            out_csv,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
    )
    totals = sum_columns(
            results_final, result_file_prefix, language
    )
    return results_final, totals

def detect_packages(
    data_path,
    save_path,
    model_name,
    log_level,
    language,
    master_file="master.json",
    add_string="",
):
    if log_level != "off":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    all_results = []
    all_totals = []

    logging.info(f"Processing package responses...")
    results, totals = process_python_dataset(
        master_file,
        save_path,
        data_path,
        language,
        f"packages_1{add_string}.json",
        f"packages_2{add_string}.json",
        "PH Test",
    )
    all_results.append(results)
    all_totals.append(totals)

    logging.info("Merging all datasets")
    package_names_final = pd.concat(all_results)
    final_totals = pd.concat(all_totals)

    totals_sum = final_totals.sum()
    totals_df = pd.DataFrame([totals_sum], index=["Totals"])
    final_totals = pd.concat([final_totals, totals_df])

    logging.info("Saving final results")
    final_totals.to_csv(os.path.join(save_path, "FINAL_RESULTS.csv"))
    package_names_final.drop(
        ["Prompts", "Answers", "Test_1", "Test_2"], axis=1, inplace=True
    )
    package_names_final.to_csv(
        os.path.join(save_path, "PACKAGE_NAMES.csv"), index=False
    )

    logging.info("Package detection complete. Results saved.")
    logging.info(f"Final Totals: \n {final_totals}")
