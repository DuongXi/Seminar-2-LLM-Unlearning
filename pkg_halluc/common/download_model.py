"""Base model Download from Hugging Face"""
from __future__ import annotations

import argparse
from pathlib import Path

from pkg_halluc.common import model_setup


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Tên preset model")
    ap.add_argument("--models-dir", required=True, type=Path, help="Thư mục lưu model đã tải")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--skip-sanity-check", action="store_true", help="Bỏ qua bước sinh thử 1 câu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_path = model_setup.download_base_model(args.model, args.models_dir)
    if not args.skip_sanity_check:
        model_setup.sanity_check_generation(model_path, dtype=args.dtype)


if __name__ == "__main__":
    main()
