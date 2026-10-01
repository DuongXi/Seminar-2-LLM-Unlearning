"""Print the available CAA vector layers for a model suffix"""
import argparse
import json
import os
import re


def discover_vector_layers(vectors_dir: str, model_suffix: str) -> list[int]:
    available_layers = set()
    if os.path.isdir(vectors_dir):
        pattern = re.compile(rf"vec_layer_(\d+)_{re.escape(model_suffix)}\.pt")
        for filename in os.listdir(vectors_dir):
            match = pattern.fullmatch(filename)
            if match:
                available_layers.add(int(match.group(1)))

    summary_path = os.path.join(
        vectors_dir, f"extraction_summary_{model_suffix}.json"
    )
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                target_layers = json.load(handle).get("target_layers")
        except (OSError, ValueError):
            target_layers = None

        if isinstance(target_layers, list):
            return sorted(
                available_layers.intersection(
                    layer for layer in target_layers if isinstance(layer, int)
                )
            )

    return sorted(available_layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vectors_dir")
    parser.add_argument("model_suffix")
    args = parser.parse_args()

    for layer in discover_vector_layers(args.vectors_dir, args.model_suffix):
        print(layer)


if __name__ == "__main__":
    main()