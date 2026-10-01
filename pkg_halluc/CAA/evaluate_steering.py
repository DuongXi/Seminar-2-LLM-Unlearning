"""
Contrastive Activation Addition (CAA) Steering Evaluation Script
Supports universal architectures: Llama 3.2, Qwen 2.5 Coder, DeepSeek Coder, etc.
Usage Example:
python CAA/eval/prompting_with_steering.py --config model_config/default.json --dataset_path path/to/pairs.json --layers 10 12 14 --multipliers 0.0 0.5 1.0 2.0
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from os import PathLike
import torch
# Ensure CAA directory and project root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAA_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(CAA_DIR))
for p in [SCRIPT_DIR, CAA_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

from pkg_halluc.CAA.steering_settings import SteeringSettings
from pkg_halluc.CAA.utils import make_tensor_save_suffix
from pkg_halluc.CAA.wrapper import ModelWrapper

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def get_steering_vector(layer, model_name_path: str, vectors_dir: str | PathLike[str]):
    return torch.load(get_vector_path(layer, model_name_path, vectors_dir))

def get_vector_path(layer, model_name_path: str, vectors_dir: str | PathLike[str]) -> str:
    suffix = make_tensor_save_suffix(layer, model_name_path)
    return os.path.join(os.fspath(vectors_dir), f"vec_layer_{suffix}.pt")

def process_contrastive_pair(
    item: Dict[str, Any],
    model: ModelWrapper,
) -> Dict[str, Any]:
    question: str = item["question"]
    matching_behavior: str = item["answer_matching_behavior"]
    not_matching_behavior: str = item["answer_not_matching_behavior"]

    matching_score = model.score_response(
        user_input=question,
        response=matching_behavior,
    )
    not_matching_score = model.score_response(
        user_input=question,
        response=not_matching_behavior,
    )
    is_matching = matching_score >= not_matching_score

    return {
        "question": question,
        "answer_matching_behavior": matching_behavior,
        "answer_not_matching_behavior": not_matching_behavior,
        "matching_score": matching_score,
        "not_matching_score": not_matching_score,
        "score_margin": matching_score - not_matching_score,
        "chosen": "matching" if is_matching else "not_matching",
        "is_matching": is_matching,
    }


def evaluate_pairwise_steering(
    layers: List[int],
    multipliers: List[float],
    settings: SteeringSettings,
    model: Optional[ModelWrapper] = None,
    dataset_path: Optional[str] = None,
    vectors_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    best_model_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
):
    if not vectors_dir or not results_dir:
        raise ValueError("Provide main_path or explicit vector and results directories.")
    save_results_dir = results_dir
    os.makedirs(save_results_dir, exist_ok=True)
    if best_model_dir is None:
        best_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(save_results_dir)),
            "best_model_bundle",
        )

    if not dataset_path or not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Pairwise evaluation dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    if max_samples:
        evaluation_data = evaluation_data[:max_samples]

    print(f"[*] Loaded {len(evaluation_data)} paired completions.", flush=True)

    # Initialize the model if one was not provided.
    if model is None:
        print(f"[*] Initializing ModelWrapper for {settings.model_name}...", flush=True)
        model = ModelWrapper(
            hf_token=HUGGINGFACE_TOKEN,
            model_name_or_path=settings.model_name,
            config_path=settings.config_path,
            use_chat=not settings.use_base_model,
            override_model_weights_path=settings.override_model_weights_path,
            from_config=settings.from_config,
            local_files_only=settings.local_files_only,
        )

    # Evaluate each layer and multiplier.
    summary_records = []
    best_candidate = None

    def consider_best(summary: Dict[str, Any], layer: int, vector: torch.Tensor):
        nonlocal best_candidate
        if not summary.get("num_samples") or "matching_accuracy" not in summary or "mean_margin" not in summary:
            return

        score = (float(summary["matching_accuracy"]), float(summary["mean_margin"]))
        if best_candidate is None or score > best_candidate["score"]:
            best_candidate = {
                "score": score,
                "summary": summary,
                "layer": layer,
                "vector": vector.detach().to(device="cpu").clone(),
            }

    for layer in layers:
        model_name_path = model.model_name_path
        if settings.override_vector_model is not None:
            model_name_path = settings.override_vector_model

        target_layer = settings.override_vector if settings.override_vector is not None else layer

        # Load the steering vector.
        vector = get_steering_vector(
            target_layer,
            model_name_path,
            vectors_dir,
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
                with open(save_filename, "r", encoding="utf-8") as f:
                    cached_summary = json.load(f).get("summary", {})
                if (
                    isinstance(cached_summary, dict)
                    and cached_summary.get("num_samples") == len(evaluation_data)
                ):
                    summary_records.append(cached_summary)
                    consider_best(cached_summary, layer, vector)
                else:
                    print("[!] Cached result sample count differs; excluding it from best selection.")
                continue

            results = []
            desc = f"Steering Layer {layer} (x{multiplier:+.1f})"
            for item in tqdm(evaluation_data, desc=desc):
                model.reset_all()

                # Add steering vector at target layer
                if abs(multiplier) > 1e-6:
                    model.set_add_activations(layer, vector, multiplier=multiplier)

                result = process_contrastive_pair(
                    item=item,
                    model=model,
                )
                results.append(result)

            # Summarize pairwise completion ranking.
            summary = {
                "layer": layer,
                "multiplier": multiplier,
                "num_samples": len(results),
            }

            if results:
                matching_count = sum(1 for r in results if r.get("is_matching", False))
                accuracy = matching_count / len(results)
                pairwise_failure_rate = 1.0 - accuracy
                mean_margin = sum(r["score_margin"] for r in results) / len(results)
                mean_match_score = sum(r["matching_score"] for r in results) / len(results)

                summary.update({
                    "matching_accuracy": accuracy,
                    "pairwise_failure_rate": pairwise_failure_rate,
                    "mean_matching_score": mean_match_score,
                    "mean_margin": mean_margin,
                })

                print(
                    f"--> [Layer {layer}, Mult {multiplier:+.1f}] "
                    f"Pairwise accuracy: {accuracy * 100:.1f}% | "
                    f"Pairwise failure rate: {pairwise_failure_rate * 100:.1f}% | "
                    f"Margin: {mean_margin:+.4f}",
                    flush=True,
                )

            consider_best(summary, layer, vector)
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
    if best_candidate is not None:
        model.reset_all()
        model.save_steering_bundle(
            best_model_dir,
            layer=best_candidate["layer"],
            vector=best_candidate["vector"],
            multiplier=float(best_candidate["summary"]["multiplier"]),
        )
        print(
            f"[+] Best bundle saved to {best_model_dir} "
            f"(layer={best_candidate['layer']}, "
            f"multiplier={best_candidate['summary']['multiplier']}, "
            f"accuracy={best_candidate['score'][0]:.4f}, "
            f"margin={best_candidate['score'][1]:+.4f})",
            flush=True,
        )
    else:
        print("[!] No completed pairwise scores; best-model bundle was not saved.", flush=True)
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
        default=None,
        help="Model preset (llama3.2-1b, qwen2.5-coder-1.5b, deepseek-coder-1.3b) or HF model identifier",
    )
    parser.add_argument("--main_path", type=str, default=None, help="Model data directory from config data.main_path")
    parser.add_argument("--vectors_dir", type=str, default=None, help="CAA vector directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Evaluation results directory")
    parser.add_argument("--best_model_dir", type=str, default=None, help="Directory for the best model and CAA vector bundle")
    parser.add_argument("--layers", nargs="+", type=int, required=True, help="Transformer layer indices to steer")
    parser.add_argument("--multipliers", nargs="+", type=float, required=True, help="Multipliers for steering vector")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override the generated pairwise test dataset")
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test sample count for quick sweeps")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--from_config", action="store_true", default=False, help="Initialize architecture without weights")
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model files only from the local Hugging Face cache",
    )

    args = parser.parse_args()

    # Resolve model name from config if provided
    cfg = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found at: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    model_name = args.model_name_or_path or cfg.get("model_name")
    if not model_name:
        raise ValueError("Model name not specified! Set model_name in config or provide --model_name_or_path.")
    main_path = args.main_path or cfg.get("data", {}).get("main_path")
    if not main_path:
        raise ValueError("Model data main_path not specified! Set data.main_path in config or provide --main_path.")
    vectors_dir = args.vectors_dir
    if vectors_dir is None:
        vectors_dir = os.path.join(main_path, "contrastive", "vectors")
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(main_path, "contrastive", "results")
    dataset_path = args.dataset_path or os.path.join(
        main_path, "contrastive", "generate", "generate_dataset_val.json"
    )
    steering_settings = SteeringSettings(
        model_name=model_name,
        config_path=args.config,
        from_config=args.from_config,
        local_files_only=args.local_files_only,
        override_vector=args.override_vector,
        override_vector_model=args.override_vector_model,
        use_base_model=args.use_base_model,
        override_model_weights_path=args.override_model_weights_path,
    )

    evaluate_pairwise_steering(
        layers=args.layers,
        multipliers=args.multipliers,
        settings=steering_settings,
        dataset_path=dataset_path,
        vectors_dir=vectors_dir,
        results_dir=results_dir,
        best_model_dir=args.best_model_dir,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
