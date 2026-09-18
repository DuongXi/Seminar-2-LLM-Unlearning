"""Quét câu lệnh import trần trong code sinh ra để tính tỉ lệ import hallucination."""
import argparse
import ast
import csv
import json
import re
import sys

import pandas as pd


def normalize_python_package(name: str) -> str:
    """Chuẩn hoá tên package theo PEP 503 (gộp dấu phân cách, viết thường)."""
    if not name or not isinstance(name, str):
        return name
    name = re.sub(r"\d+\.\s*", "", name)
    name = re.sub(r"(?<=.)\n(?=.)", " ", name)
    name = re.sub(r"\n", "", name)
    name = re.sub(r"[-_.]+", "-", name)
    name = name.strip(" `.-_")
    return name.lower()


IMPORT_TO_PYPI_ALIAS = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "pil": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "google": "google-api-python-client",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "jwt": "pyjwt",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "requests_oauthlib": "requests-oauthlib",
    "serial": "pyserial",
    "usb": "pyusb",
    "OpenSSL": "pyopenssl",
    "Crypto": "pycryptodome",
    "nacl": "pynacl",
    "markdown_it": "markdown-it-py",
}

STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()


def extract_top_level_imports(code: str):
    """Lấy tên module cấp cao nhất từ các câu lệnh import, code lỗi cú pháp thì dùng regex."""
    modules = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    modules.append(node.module.split(".")[0])
    except SyntaxError:
        for line in code.splitlines():
            line = line.strip()
            m = re.match(r"^import\s+([a-zA-Z_][\w.]*)", line)
            if m:
                modules.append(m.group(1).split(".")[0])
                continue
            m = re.match(r"^from\s+([a-zA-Z_][\w.]*)\s+import\s", line)
            if m:
                modules.append(m.group(1).split(".")[0])
    return modules


def classify_imports(modules, valid_pypi_names, false_positives):
    """Phân loại module thành valid/hallucinated, bỏ stdlib và các false positive."""
    valid, hallucinated = [], []
    seen = set()
    for m in modules:
        if not m or m in seen:
            continue
        seen.add(m)
        if m in STDLIB_MODULES:
            continue
        pypi_name = IMPORT_TO_PYPI_ALIAS.get(m, m)
        norm = normalize_python_package(pypi_name)
        if norm in valid_pypi_names:
            valid.append(m)
        elif norm not in false_positives:
            hallucinated.append(m)
    return valid, hallucinated


def load_name_set(json_or_csv_path):
    """Đọc tập tên package từ file JSON hoặc CSV (mỗi dòng 1 tên)."""
    if json_or_csv_path.endswith(".json"):
        with open(json_or_csv_path, encoding="utf-8") as f:
            return set(json.load(f))
    names = set()
    with open(json_or_csv_path, encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if n:
                names.add(normalize_python_package(n))
    return names


def scan_master_file(master_json_path, valid_pypi_names, false_positives, answer_key="Answers"):
    """Quét từng dòng master, trả về (records, tổng valid, tổng hallucinated)."""
    records = []
    with open(master_json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            code = rec.get(answer_key, "") or ""
            modules = extract_top_level_imports(code)
            valid, hall = classify_imports(modules, valid_pypi_names, false_positives)
            records.append({
                "prompt": rec.get("Prompts"),
                "import_valid": valid,
                "import_hallucinated": hall,
            })
    total_valid = sum(len(r["import_valid"]) for r in records)
    total_hall = sum(len(r["import_hallucinated"]) for r in records)
    return records, total_valid, total_hall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master_json", required=True)
    ap.add_argument("--pypi_names", required=True)
    ap.add_argument("--false_positive_csv", default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    valid_names = load_name_set(args.pypi_names)
    fps = set()
    if args.false_positive_csv:
        with open(args.false_positive_csv, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1].strip():
                    fps.add(normalize_python_package(row[1]))

    records, total_valid, total_hall = scan_master_file(args.master_json, valid_names, fps)

    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)

    total = total_valid + total_hall
    rate = 100.0 * total_hall / total if total else float("nan")
    print(f"import_valid={total_valid}  import_hallucinated={total_hall}  "
          f"Import-Hallucination-Rate={rate:.2f}%  -> {args.out_csv}")


if __name__ == "__main__":
    main()
