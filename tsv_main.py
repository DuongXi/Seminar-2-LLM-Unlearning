"""Train package-specific TSV vectors with a frozen HuggingFace causal LM.

Each dataset record supplies the input-token index at the package-generation
boundary. For a causal LM, the hidden state at index ``p`` is the state used
to predict token ``p + 1``. Therefore, when the package is the next generated
token, ``p`` is normally the final token index of the prompt. The package
string is deliberately never searched for in the prompt: a package can be
generated without appearing in the input, and string matching would measure
the wrong representation.

--data_path also accepts a .csv file in the real dataset schema (Prompts,
Answers, Test_1, Test_2, valid_1, hallucinated_1, valid_2, hallucinated_2,
pip, pip_valid, pip_hallucinated - the format of LLM_LY_results.csv /
LLM_AT_results.csv). Only the "pip" surface (bare imports scanned directly
from generated code) is usable as training data this way: it's the only
column whose package names have a concrete character position inside real
generated text (the Answers column) to compute a boundary index from.
Test_1/Test_2 are themselves already-extracted name lists, not raw response
text, in this schema - there is no text to locate a position within, so they
cannot be converted into boundary-position training examples. See
_records_from_results_csv for the exact conversion.
"""

import argparse
import ast
import csv
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("package_tsv")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_package_representation(
    hidden_states: torch.Tensor,
    package_positions: Sequence[Sequence[int]],
    strategy: str = "first_token",
) -> torch.Tensor:
    """Extract one representation per sample from boundary token positions.

    ``package_positions`` contains hidden-state indices, not indices of
    package text. An index identifies the state immediately before the next
    token is generated. A list of indices permits future span strategies;
    ``first_token`` is the default because a package name starts at the
    generation boundary.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            "Expected hidden states [batch, seq_len, hidden_size], got "
            f"{tuple(hidden_states.shape)}"
        )
    if len(package_positions) != hidden_states.shape[0]:
        raise ValueError("There must be one boundary position entry per batch sample")

    aliases = {"first": "first_token", "last": "last_token"}
    strategy = aliases.get(strategy, strategy)
    if strategy not in {"first_token", "last_token", "mean"}:
        raise ValueError("strategy must be first_token, last_token, or mean")

    representations = []
    for batch_index, positions in enumerate(package_positions):
        if not positions:
            raise ValueError(f"Sample {batch_index} has an empty boundary position set")
        if any(position < 0 or position >= hidden_states.shape[1] for position in positions):
            raise ValueError(
                f"Invalid package-generation boundary in sample {batch_index}: {positions}"
            )
        if strategy == "mean":
            representation = hidden_states[batch_index, positions].mean(dim=0)
        elif strategy == "last_token":
            representation = hidden_states[batch_index, positions[-1]]
        else:
            representation = hidden_states[batch_index, positions[0]]
        representations.append(representation)
    return torch.stack(representations, dim=0)


def _positions_from_value(value: Any) -> List[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
    if isinstance(value, dict):
        if "index" in value:
            return _positions_from_value(value["index"])
        if "start" in value and "end" in value:
            start, end = value["start"], value["end"]
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("Boundary start/end must be integers")
            if end <= start:
                raise ValueError("Boundary end must be greater than start")
            return list(range(start, end))
    raise ValueError(
        "package_generation_position must be an integer, list of integers, "
        "or {index}, {start, end}"
    )


_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)", re.MULTILINE)
_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import", re.MULTILINE)


def _parse_list_cell(value: Any) -> List[str]:
    """Parse a str(list)-style CSV cell, e.g. "['numpy', 'rds']", the same
    format eval_variant.py's steering pipeline writes for list-valued
    columns. Returns [] for empty/unparseable cells rather than raising,
    since a blank cell just means "no packages detected" for that row."""
    value = (value or "").strip() if isinstance(value, str) else value
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _find_import_span(answer_text: str, package_name: str) -> Any:
    """Locate `package_name` as written in an `import X` / `from X import`
    statement inside `answer_text`. Returns the (start, end) character span
    of the name itself, or None if it can't be found. This mirrors the same
    import/from-import syntax eval_variant.py's _extract_bare_imports uses to
    pull names out in the first place, so a name that came from the 'pip'
    column should always be relocatable here - if it isn't, the row is
    skipped rather than guessed at (see caller)."""
    escaped = re.escape(package_name)
    for pattern in (
        re.compile(rf"^\s*import\s+({escaped})\b", re.MULTILINE),
        re.compile(rf"^\s*from\s+({escaped})\b", re.MULTILINE),
    ):
        match = pattern.search(answer_text)
        if match:
            return match.span(1)
    return None


def _records_from_results_csv(csv_path: str, tokenizer: Any) -> List[Dict[str, Any]]:
    """Build package_generation_position training records directly from a
    results CSV (Prompts, Answers, pip, pip_valid, pip_hallucinated columns -
    see module docstring for why only the 'pip' surface is usable here).

    For every package name in pip_valid (label=1) / pip_hallucinated
    (label=0) on a row, the training "prompt" is:

        Prompts + "\\n\\n" + <Answers text up to, but not including, that
        package name>

    i.e. the text the model would have seen immediately before writing that
    package name, with the boundary position set to the last token of that
    truncated text (matching the "hidden state predicts the first package
    token" convention used everywhere else in this file). The
    Prompts/Answers join format is an explicit assumption (plain
    newline-separated concatenation, no chat template) since the original
    generation-time template isn't available here - adjust
    _records_from_results_csv if your real pipeline formats context
    differently.
    """
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"{csv_path} contains no data rows")
    required_columns = {"Prompts", "Answers", "pip_valid", "pip_hallucinated"}
    missing_columns = required_columns - set(rows[0].keys())
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing_columns)}")

    records: List[Dict[str, Any]] = []
    skipped_unmatched = 0
    for row in rows:
        prompt_text = row["Prompts"]
        answer_text = row["Answers"]
        for names, label in (
            (_parse_list_cell(row["pip_valid"]), 1),
            (_parse_list_cell(row["pip_hallucinated"]), 0),
        ):
            for name in names:
                span = _find_import_span(answer_text, name)
                if span is None:
                    skipped_unmatched += 1
                    continue
                start, _end = span
                truncated_answer = answer_text[:start]
                full_text = f"{prompt_text}\n\n{truncated_answer}"
                token_ids = tokenizer(full_text)["input_ids"]
                if not token_ids:
                    skipped_unmatched += 1
                    continue
                records.append(
                    {
                        "prompt": full_text,
                        "label": label,
                        "positions": [len(token_ids) - 1],
                    }
                )

    if skipped_unmatched:
        LOGGER.info(
            "Skipped %d package name(s) from %s that could not be relocated "
            "inside their row's Answers text",
            skipped_unmatched,
            csv_path,
        )
    labels_seen = {record["label"] for record in records}
    if labels_seen != {0, 1}:
        raise ValueError(
            f"{csv_path} produced training examples for only one class "
            f"({labels_seen}) after conversion - need both pip_valid and "
            "pip_hallucinated examples that can be relocated in Answers"
        )
    return records


def load_package_dataset(data_path: str, tokenizer: Any = None) -> List[Dict[str, Any]]:
    """Load JSON/JSONL records with explicit generation-boundary positions,
    or a results .csv file (see _records_from_results_csv - requires
    `tokenizer` to compute positions, since a .csv file only has raw text,
    not precomputed token indices).

    Accepted position fields (JSON/JSONL only) are
    ``package_generation_position`` (preferred), ``package_position``, and
    ``package_positions``. Values are zero-based token indices in the
    unpadded prompt, and point to the hidden state that predicts the first
    package token. No package text lookup is performed for JSON/JSONL input.
    """
    if data_path.lower().endswith(".csv"):
        if tokenizer is None:
            raise ValueError(
                "Loading a .csv results file requires a tokenizer to "
                "compute boundary positions - pass one to load_package_dataset"
            )
        return _records_from_results_csv(data_path, tokenizer)

    with open(data_path, "r", encoding="utf-8") as data_file:
        if data_path.lower().endswith(".jsonl"):
            records: Any = [json.loads(line) for line in data_file if line.strip()]
        else:
            records = json.load(data_file)
    if isinstance(records, dict):
        records = records.get("data", records.get("samples"))
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must contain a non-empty JSON array or JSONL file")

    resolved = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "prompt" not in record or "label" not in record:
            raise ValueError(f"Sample {index} must contain prompt and label")
        try:
            label = int(record["label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sample {index} label must be 0 or 1") from error
        if label not in (0, 1):
            raise ValueError(f"Sample {index} label must be 0 (hallucinated) or 1 (real)")

        position_value = next(
            (
                record.get(field)
                for field in (
                    "package_generation_position",
                    "package_position",
                    "package_positions",
                )
                if field in record
            ),
            None,
        )
        if position_value is None:
            raise ValueError(
                f"Sample {index} must provide package_generation_position; "
                "it is the zero-based prompt-token index whose hidden state "
                "predicts the first package token"
            )
        positions = _positions_from_value(position_value)
        if not positions or any(position < 0 for position in positions):
            raise ValueError(f"Sample {index} has invalid boundary positions: {positions}")
        resolved.append(
            {"prompt": str(record["prompt"]), "label": label, "positions": positions}
        )

    labels = [record["label"] for record in resolved]
    if set(labels) != {0, 1}:
        raise ValueError("The dataset must contain both classes: 0 hallucinated and 1 real")
    return resolved


def _stratified_split(
    records: List[Dict[str, Any]], eval_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    train, evaluation = [], []
    for label in (0, 1):
        class_records = [record for record in records if record["label"] == label]
        rng.shuffle(class_records)
        if len(class_records) < 2:
            raise ValueError("Each class needs at least two samples for a split")
        eval_count = min(
            max(1, round(len(class_records) * eval_fraction)), len(class_records) - 1
        )
        evaluation.extend(class_records[:eval_count])
        train.extend(class_records[eval_count:])
    return train, evaluation


def _batch_records(records: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


@torch.no_grad()
def collect_layer_representations(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    layers: Sequence[int],
    batch_size: int,
    device: torch.device,
    strategy: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    representations = {layer: [] for layer in layers}
    labels = []
    for batch in _batch_records(records, batch_size):
        encoded = tokenizer(
            [record["prompt"] for record in batch],
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        if outputs.hidden_states is None:
            raise ValueError("Model did not expose hidden states")
        positions = [record["positions"] for record in batch]
        for layer in layers:
            state_index = layer + 1
            if state_index >= len(outputs.hidden_states):
                raise ValueError(f"Model did not return hidden states for layer {layer}")
            representations[layer].append(
                extract_package_representation(
                    outputs.hidden_states[state_index], positions, strategy
                )
            )
        labels.extend(record["label"] for record in batch)

    result = {layer: torch.cat(values, dim=0) for layer, values in representations.items()}
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    for layer, values in result.items():
        if values.ndim != 2 or values.shape[1] != model.config.hidden_size:
            raise ValueError(f"Layer {layer} returned incompatible shape {tuple(values.shape)}")
    return result, label_tensor


def _train_layer(
    representations: torch.Tensor,
    labels: torch.Tensor,
    lr: float,
    num_epochs: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_size = representations.shape[-1]
    centroids = torch.stack(
        [representations[labels == label].mean(dim=0) for label in (0, 1)]
    ).detach()
    centroids = F.normalize(centroids.float(), dim=-1)
    # The pretrained model is frozen; this is the only trainable parameter.
    tsv = nn.Parameter(torch.zeros(hidden_size, device=representations.device))
    optimizer = torch.optim.AdamW([tsv], lr=lr)
    normalized_representations = representations.float()
    for _ in range(num_epochs):
        shifted = F.normalize(normalized_representations + tsv, dim=-1)
        logits = shifted @ centroids.T / temperature
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return tsv.detach(), centroids


def _score(
    representations: torch.Tensor,
    labels: torch.Tensor,
    tsv: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float,
) -> float:
    shifted = F.normalize(representations.float() + tsv, dim=-1)
    probabilities = torch.softmax(shifted @ centroids.T / temperature, dim=-1)[:, 1]
    return float(roc_auc_score(labels.cpu().numpy(), probabilities.cpu().numpy()))


def _resolve_layers(args: argparse.Namespace, num_layers: int) -> List[int]:
    if args.layers:
        layers = list(dict.fromkeys(args.layers))
    else:
        start = 0 if args.layer_start is None else args.layer_start
        end = num_layers - 1 if args.layer_end is None else args.layer_end
        layers = list(range(start, end + 1))
    if not layers or any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"Layer indices must be in [0, {num_layers - 1}]")
    return layers


def _config_value(config: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train package-specific TSV vectors")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--layer_start", type=int)
    parser.add_argument("--layer_end", type=int)
    parser.add_argument(
        "--representation_strategy",
        choices=["first_token", "last_token", "mean"],
        default="first_token",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_fraction", type=float, default=0.25)
    parser.add_argument("--cos_temp", type=float, default=0.1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_epochs < 1 or args.cos_temp <= 0:
        raise ValueError("batch_size and num_epochs must be positive; cos_temp must be positive")
    if not 0 < args.eval_fraction < 1:
        raise ValueError("eval_fraction must be in (0, 1)")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    LOGGER.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=model_dtype)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_layers = _config_value(model.config, ("num_hidden_layers", "n_layer", "num_layers"))
    hidden_size = _config_value(model.config, ("hidden_size", "n_embd", "d_model"))
    if num_layers is None or hidden_size is None:
        raise ValueError("Model config must expose a compatible layer count and hidden size")
    layers = _resolve_layers(args, int(num_layers))

    LOGGER.info("Loading package dataset...")
    records = load_package_dataset(args.data_path, tokenizer)
    LOGGER.info("Number of samples: %d", len(records))
    LOGGER.info("Real package samples: %d", sum(record["label"] == 1 for record in records))
    LOGGER.info("Hallucinated package samples: %d", sum(record["label"] == 0 for record in records))
    LOGGER.info("Hidden size: %d", hidden_size)
    LOGGER.info("Number of layers: %d", num_layers)

    train_records, eval_records = _stratified_split(records, args.eval_fraction, args.seed)
    LOGGER.info("Extracting package-generation-boundary representations...")
    train_reps, train_labels = collect_layer_representations(
        model, tokenizer, train_records, layers, args.batch_size, device, args.representation_strategy
    )
    eval_reps, eval_labels = collect_layer_representations(
        model, tokenizer, eval_records, layers, args.batch_size, device, args.representation_strategy
    )

    vectors, scores = {}, {}
    for layer in layers:
        LOGGER.info("Training layer %d...", layer)
        vector, centroids = _train_layer(
            train_reps[layer], train_labels, args.lr, args.num_epochs, args.cos_temp
        )
        score = _score(eval_reps[layer], eval_labels, vector, centroids, args.cos_temp)
        vectors[str(layer)] = vector.cpu()
        scores[str(layer)] = score
        LOGGER.info("Layer %d AUROC: %.6f", layer, score)

    best_layer = max(layers, key=lambda layer: scores[str(layer)])
    best_auroc = scores[str(best_layer)]
    LOGGER.info("Best layer: %d", best_layer)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {"best_layer": best_layer, "vectors": vectors},
        os.path.join(args.output_dir, "steering_vector.pt"),
    )
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "model_path": args.model_path,
                "hidden_size": int(hidden_size),
                "num_hidden_layers": int(num_layers),
                "best_layer": best_layer,
                "best_auroc": best_auroc,
                "label_semantics": {"0": "hallucinated", "1": "real"},
                "representation_strategy": args.representation_strategy,
                "position_semantics": (
                    "zero-based unpadded prompt-token index; hidden state at this "
                    "index predicts the first package token"
                ),
            },
            metadata_file,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "layer_scores.json"), "w", encoding="utf-8") as scores_file:
        json.dump({str(layer): scores[str(layer)] for layer in layers}, scores_file, indent=2)
    LOGGER.info("Saving steering vector...")


if __name__ == "__main__":
    main()
