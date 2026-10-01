"""Detect package hallucinations by parsing responses and checking PyPI names."""
import pandas as pd
import re
import logging
import os
import csv
from typing import List

from pkg_halluc.evaluation.detection import aggregate_results, custom_parse_python


def normalize_javascript(name):
    if pd.isnull(name):
        return name
    if not isinstance(name, str):
        name = str(name)
    name = re.sub(r"[`]", "", name)
    return name


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


def check_npms(npm_list, npm_names):
    in_set = []
    not_in_set = []
    for item in npm_list:
        if " " in item or item == "None" or item == "nan":
            continue
        if item in npm_names:
            in_set.append(item)
        else:
            not_in_set.append(item)

    return in_set, not_in_set


def detect_repetitive_loop(
    text: str, min_cycle_items: int = 3, min_cycles: int = 3
) -> int:
    """Find repeated output cycles and return the start of the second cycle."""
    lines = text.split("\n")
    normalized_lines = []

    for line in lines:
        # Remove list numbering before comparing repeated lines.
        clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        clean = clean.strip()
        if clean:
            normalized_lines.append(clean)

    if len(normalized_lines) < min_cycle_items * min_cycles:
        return 0

    # Try possible cycle lengths.
    for cycle_len in range(
        min_cycle_items, min(21, len(normalized_lines) // min_cycles + 1)
    ):
        pattern = normalized_lines[:cycle_len]

        # Count consecutive repetitions of the candidate cycle.
        repetitions = 1
        for i in range(cycle_len, len(normalized_lines), cycle_len):
            chunk = normalized_lines[i : i + cycle_len]
            if chunk == pattern:
                repetitions += 1
            else:
                break

        if repetitions >= min_cycles:
            # Return the start position of the repeated cycle.
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
    """Normalize a package name according to PEP 503."""
    if not name or not isinstance(name, str):
        return name

    # Remove any remaining list numbering.
    name = re.sub(r"\d+\.\s*", "", name)

    # Normalize line breaks embedded in a package name.
    name = re.sub(r"(?<=.)\n(?=.)", " ", name)
    name = re.sub(r"\n", "", name)

    # Normalize separators to hyphens.
    name = re.sub(r"[-_.]+", "-", name)

    # Trim surrounding punctuation and whitespace.
    name = name.strip(" `.-_")

    # Package names are case-insensitive.
    return name.lower()


def extract_and_clean_packages(
    package_string: str, detect_loops: bool = True
) -> List[str]:
    """Parse package names from model output and remove common formatting noise."""

    package_string = strip_code_blocks(package_string)

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
            # Ignore explanatory prose after the package list.
            package_string = package_string[: match.start()]
            break

    # Truncate repetitive model output.
    if detect_loops:
        loop_start = detect_repetitive_loop(package_string)
        if loop_start > 0:
            package_string = package_string[:loop_start]

            # Prefer explicitly numbered package lists.
    numbered_pattern = r"(\d+[\.\)])\s*[`\']?([a-zA-Z0-9\-_\.]+)[`\']?"
    numbered_matches = re.findall(numbered_pattern, package_string)

    from_numbered_list = False
    if numbered_matches and len(numbered_matches) >= 2:
        numbered_packages = []
        for number, pkg in numbered_matches:
            pkg = pkg.strip()
            if pkg and len(pkg) > 2:
                # Inspect the complete numbered line before accepting the name.
                line_pattern = (
                    re.escape(number)
                    + r"\s*[`\']?"
                    + re.escape(pkg)
                    + r"[`\']?([^\n]*)"
                )
                line_match = re.search(line_pattern, package_string)

                if line_match:
                    full_line = line_match.group(0)

                    # Skip lines that match descriptive patterns.
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
                    # Keep the package when its full line cannot be found.
                    numbered_packages.append(pkg)

        if len(numbered_packages) >= 2:
            parts = numbered_packages
            from_numbered_list = True
    # Parse the full response when it is not a numbered list.
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

    # Filter common non-package words.
    delete_words = {
        "and",
        "the",
        "optional",
        "it",
        "its",
        "you",
        "which",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
        "a",
        "an",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "also",
        "here",
        "there",
        "once",
        "after",
        "before",
        "since",
        "until",
        "while",
        "heres",
        "here's",
        "theres",
        "there's",
        "lets",
        "let's",
        "now",
        "next",
        "first",
        "second",
        "third",
        "finally",
        "firstly",
        "secondly",
        "sure",
        "okay",
        "well",
        "python",
        "using",
        "use",
        "used",
        "package",
        "packages",
        "library",
        "libraries",
        "module",
        "modules",
        "install",
        "pip",
        "import",
        "code",
        "following",
        "required",
        "necessary",
        "needed",
        "please",
        "note",
        "bash",
        "shell",
        "def",
        "class",
        "return",
        "yield",
        "pass",
        "break",
        "continue",
        "while",
        "try",
        "except",
        "finally",
        "raise",
        "assert",
        "global",
        "nonlocal",
        "lambda",
        "with",
        "del",
        "true",
        "false",
        "none",
        "self",
        "cls",
        "init",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "len",
        "print",
        "input",
        "open",
        "file",
        "read",
        "write",
        "close",
        "encoding",
        "decoded",
        "encoded",
        "string",
        "input",
        "output",
        "result",
        "value",
        "data",
        "text",
        "name",
        "path",
        "url",
        "response",
        "request",
        "error",
        "exception",
        "message",
        "headers",
        "authorization",
        "bearer",
        "token",
        "api",
        "key",
        "create",
        "creating",
        "created",
        "make",
        "making",
        "made",
        "perform",
        "performing",
        "performed",
        "execute",
        "executing",
        "run",
        "running",
        "build",
        "building",
        "install",
        "installing",
        "manipulate",
        "manipulating",
        "handle",
        "handling",
        "process",
        "processing",
        "analyze",
        "analyzing",
        "visualize",
        "visualizing",
        "useful",
        "helpful",
        "important",
        "essential",
        "basic",
        "simple",
        "complex",
        "advanced",
        "provides",
        "allows",
        "enables",
        "supports",
        "includes",
        "contains",
        "features",
        "functionality",
        "capabilities",
        "methods",
        "functions",
        "structure",
        "structures",
        "dynamics",
        "study",
        "analysis",
        "app",
        "agent",
        "dynamic",
        "consider",
        "including",
        "youll",
        "disconnecting",
        "function",
        "however",
        "object",
        "heres",
        "adding",
        "along",
        "mult",
        "multi",
        "dimensional",
    }

    generic_frameworks = {
        "django",
        "flask",
        "elastic",
        "apm",
        "new",
        "vmware",
        "rest",
        "framework",
    }

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

        if not pkg or len(pkg) <= 2 or pkg.isdigit() or pkg.lower() in delete_words:
            continue

        if pkg.lower() in generic_frameworks and "-" not in pkg:
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

        legitimate_packages = {
            "requests",
            "responses",
            "attrs",
            "datasets",
            "datatable",
        }

        if pkg.lower() not in legitimate_packages:
            var_pattern = r"^.*_(string|data|input|output|result|value|text|file|path|name|var|obj|object|item|list|dict|array|buffer|stream|content|response|request|str|num|int|float|id|idx|key|val|ptr|ref|temp|tmp|flag|bool|char|byte)s?$"
            if re.match(var_pattern, pkg, re.IGNORECASE):
                continue

            func_prefixes = [
                "print",
                "get",
                "set",
                "read",
                "write",
                "load",
                "save",
                "fetch",
                "send",
                "receive",
            ]
            if any(
                pkg.lower().startswith(prefix) and len(pkg) > len(prefix)
                for prefix in func_prefixes
            ):
                continue

            if pkg.endswith("_"):
                continue

            if pkg.endswith("."):
                continue

        cleaned.append(pkg)

    # Reject responses that are mostly prose or appear truncated.
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

    # Normalize package names and remove duplicates.
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

    df["Test_1"] = df["Test_1"].apply(custom_parse_python.delete_dupes_and_empty)
    df["Test_2"] = df["Test_2"].apply(custom_parse_python.delete_dupes_and_empty)

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
    """Remove code blocks while preserving pip install commands."""
    if not text or not isinstance(text, str):
        return text

    # Preserve pip install commands before removing code blocks.
    pip_install_pattern = r"pip\s+install\s+[^\n]+"
    pip_installs = re.findall(pip_install_pattern, text, re.IGNORECASE)

    # Remove closed code blocks.
    code_block_pattern = r"```[\w]*\n.*?\n```"
    cleaned_text = re.sub(code_block_pattern, "\n", text, flags=re.DOTALL)

    # Remove an unfinished code block through the end of the response.
    unclosed_pattern = r"```[\w]*\n.*"
    if re.search(unclosed_pattern, cleaned_text, re.DOTALL):
        match = re.search(r"```", cleaned_text)
        if match:
            cleaned_text = cleaned_text[: match.start()]

    # Remove multiline inline code blocks.
    inline_multiline_pattern = r"`[^`\n]*\n(?:(?!\d+[\.\)]\s)[^`])*`"
    cleaned_text = re.sub(inline_multiline_pattern, "\n", cleaned_text, flags=re.DOTALL)

    # Restore the preserved install commands.
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
    df["pip_hallucinated"] = df["pip_hallucinated"].apply(
        custom_parse_python.delete_dupes_and_empty
    )

    return df


def check_pips(pip_list, pip_names):
    in_set = []
    not_in_set = []
    translation_table = str.maketrans("", "", "()[]`")
    version_pattern = re.compile(r"([^=<>!~]+)([=<>!~]{1,2}[\d\.]+)?")

    if pip_list:
        for item in pip_list:
            text = item.translate(translation_table)
            if bool(re.search(r"[+@:\"\',{}/\*]", text)):
                continue
            for part in text.split():
                if part.startswith("--"):
                    continue
                match = version_pattern.match(part)
                if match:
                    text = match.group(1).strip()
                    text = normalize_python_package(text)
                    if text.startswith("--") or "requirements" in text:
                        continue
                    if text in pip_names:
                        in_set.append(text)
                    else:
                        not_in_set.append(text)

    return in_set, not_in_set


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean string columns in a DataFrame."""
    # Remove NUL characters and normalize line endings.
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(
        lambda x: (
            x.replace("\x00", "").replace("\r\n", "\n") if isinstance(x, str) else x
        )
    )
    return df


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

    def process_python_dataset(
        master_file, package_file_1, package_file_2, result_file_prefix
    ):
        merged_results = aggregate_results.merge_prompts_and_packages(
            os.path.join(save_path, master_file),
            os.path.join(save_path, package_file_1),
            os.path.join(save_path, package_file_2),
        )
        merged_results = sanitize_df(merged_results)
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
        totals = aggregate_results.sum_columns(
            results_final, result_file_prefix, language
        )
        return results_final, totals

    all_results = []
    all_totals = []

    logging.info(f"Processing package responses...")
    results, totals = process_python_dataset(
        master_file,
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
