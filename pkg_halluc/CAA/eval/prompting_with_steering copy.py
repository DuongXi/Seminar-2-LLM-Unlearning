"""
Contrastive Activation Addition (CAA) Steering Evaluation Script
Supports universal architectures: Llama 3.2, Qwen 2.5 Coder, DeepSeek Coder, etc.
Evaluates model behavior shifts across layers and steering vector multipliers.

Usage Example:
python CAA/prompting_with_steering.py --model_name_or_path llama3.2-1b --layers 10 12 14 --multipliers 0.0 0.5 1.0 2.0 --type ab
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import torch as t
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure CAA directory and project root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for p in [SCRIPT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

from pkg_halluc.CAA.extractor.llama_wrapper import ModelWrapper, LlamaWrapper
from pkg_halluc.CAA.extractor.helpers import get_a_b_probs, resolve_model_name, make_tensor_save_suffix
from pkg_halluc.CAA.eval.steering_settings import SteeringSettings
from pkg_halluc.CAA.extractor.behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    PACKAGE_HALLUCINATION,
    get_results_dir,
)

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def process_item_ab(
    item: Dict[str, Any],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, Any]:
    question: str = item["question"]
    matching_behavior: str = item["answer_matching_behavior"]
    not_matching_behavior: str = item["answer_not_matching_behavior"]

    # Model output prefix "(" prompts the model for choice option
    logits = model.get_logits_from_text(
        user_input=question,
        model_output="(",
        system_prompt=system_prompt,
    )
    a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)

    # Determine choice: (A) vs (B)
    matching_is_a = "(A)" in matching_behavior or matching_behavior.strip() == "A"
    matching_prob = a_prob if matching_is_a else b_prob
    not_matching_prob = b_prob if matching_is_a else a_prob

    chosen = "A" if a_prob >= b_prob else "B"
    is_correct = (chosen == "A" and matching_is_a) or (chosen == "B" and not matching_is_a)

    return {
        "question": question,
        "answer_matching_behavior": matching_behavior,
        "answer_not_matching_behavior": not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "matching_prob": matching_prob,
        "not_matching_prob": not_matching_prob,
        "chosen": chosen,
        "is_matching": is_correct,
    }


def process_item_open_ended(
    item: Dict[str, Any],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, Any]:
    question = item["question"]
    generated_text = model.generate_text(
        user_input=question,
        system_prompt=system_prompt,
        max_new_tokens=100,
    )
    return {
        "question": question,
        "model_output": generated_text.strip(),
    }


def process_item_tqa_mmlu(
    item: Dict[str, Any],
    model: ModelWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, Any]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item.get("category", "general")
    logits = model.get_logits_from_text(
        user_input=prompt,
        model_output="(",
        system_prompt=system_prompt,
    )
    a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def test_steering(
    layers: List[int],
    multipliers: List[float],
    settings: SteeringSettings,
    model: Optional[ModelWrapper] = None,
    dataset_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
):
    save_results_dir = get_results_dir(settings.behavior)
    os.makedirs(save_results_dir, exist_ok=True)

    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }

    # 1. Load Evaluation Data
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    elif settings.type == "ab":
        test_data = get_ab_test_data(settings.behavior)
    elif settings.type == "open_ended":
        test_data = get_open_ended_test_data(settings.behavior)
    elif settings.type == "truthful_qa":
        test_data = get_truthful_qa_data()
    else:
        test_data = get_mmlu_data()

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"[*] Loaded {len(test_data)} test items for behavior='{settings.behavior}' (type={settings.type})", flush=True)

    # 2. Initialize Model if not provided
    if model is None:
        print(f"[*] Initializing ModelWrapper for {settings.model_name}...", flush=True)
        model = ModelWrapper(
            hf_token=HUGGINGFACE_TOKEN,
            model_name_or_path=settings.model_name,
            config_path=settings.config_path,
            use_chat=not settings.use_base_model,
            override_model_weights_path=settings.override_model_weights_path,
        )

    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")

    # 3. Iterate through Layers and Multipliers
    summary_records = []

    for layer in layers:
        model_name_path = model.model_name_path
        if settings.override_vector_model is not None:
            model_name_path = settings.override_vector_model

        target_layer = settings.override_vector if settings.override_vector is not None else layer

        # Retrieve steering vector
        try:
            vector = get_steering_vector(
                settings.behavior,
                target_layer,
                model_name_path,
                normalized=False,
            )
        except FileNotFoundError:
            # Fallback to normalized
            vector = get_steering_vector(
                settings.behavior,
                target_layer,
                model_name_path,
                normalized=True,
            )

        vector = vector.to(device=model.device, dtype=model.torch_dtype)

        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )

            if os.path.exists(save_filename) and not overwrite:
                print(f"[!] Found existing {os.path.basename(save_filename)} - skipping")
                continue

            results = []
            desc = f"Steering Layer {layer} (x{multiplier:+.1f})"
            for item in tqdm(test_data, desc=desc):
                model.reset_all()

                # Add steering vector at target layer
                if abs(multiplier) > 1e-6:
                    model.set_add_activations(layer, vector, multiplier=multiplier)

                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)

            # Compute evaluation summary metrics for AB test
            summary = {
                "layer": layer,
                "multiplier": multiplier,
                "num_samples": len(results),
            }

            if settings.type == "ab" and len(results) > 0:
                matching_count = sum(1 for r in results if r.get("is_matching", False))
                accuracy = matching_count / len(results)
                hallu_rate = 1.0 - accuracy
                mean_margin = sum(r["matching_prob"] - r["not_matching_prob"] for r in results) / len(results)
                mean_match_prob = sum(r["matching_prob"] for r in results) / len(results)

                summary.update({
                    "truthful_accuracy": accuracy,
                    "hallucination_rate": hallu_rate,
                    "mean_matching_prob": mean_match_prob,
                    "mean_margin": mean_margin,
                })

                print(
                    f"--> [Layer {layer}, Mult {multiplier:+.1f}] "
                    f"Accuracy: {accuracy * 100:.1f}% | "
                    f"Hallucination Rate: {hallu_rate * 100:.1f}% | "
                    f"Margin: {mean_margin:+.4f}",
                    flush=True,
                )

            summary_records.append(summary)

            payload = {
                "summary": summary,
                "results": results,
            }

            with open(save_filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)

    # Save complete sweep summary
    summary_path = os.path.join(save_results_dir, f"sweep_summary_{settings.get_formatted_model_name()}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=4)
    print(f"\n[+] Steering evaluation completed. Results saved to {save_results_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Contrastive Activation Addition (CAA) Steering")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config json (e.g. model_config/llama3.2-1b.json)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llama3.2-1b",
        help="Model preset (llama3.2-1b, qwen2.5-coder-1.5b, deepseek-coder-1.3b) or HF model identifier",
    )
    parser.add_argument("--layers", nargs="+", type=int, required=True, help="Transformer layer indices to steer")
    parser.add_argument("--multipliers", nargs="+", type=float, required=True, help="Multipliers for steering vector")
    parser.add_argument(
        "--behavior",
        type=str,
        default=PACKAGE_HALLUCINATION,
        help="Target behavior (default: package_hallucination)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to custom evaluation test dataset json",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="ab",
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
        help="Evaluation format: 'ab' (binary choice) or 'open_ended'",
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"])
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test sample count for quick sweeps")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--from_config", action="store_true", default=False, help="Initialize architecture without weights")

    args = parser.parse_args()

    # Resolve model name from config if provided
    model_name = args.model_name_or_path
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)

    steering_settings = SteeringSettings(
        behavior=args.behavior,
        type=args.type,
        model_name=model_name,
        config_path=args.config,
        system_prompt=args.system_prompt,
        override_vector=args.override_vector,
        override_vector_model=args.override_vector_model,
        use_base_model=args.use_base_model,
        override_model_weights_path=args.override_model_weights_path,
    )

    test_steering(
        layers=args.layers,
        multipliers=args.multipliers,
        settings=steering_settings,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
