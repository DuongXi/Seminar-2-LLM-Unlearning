import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from utils.model import load_model
from pkg_halluc.CAA.wrapper import ModelWrapper
from utils.data import (
    _has_chat_template,
    _build_prompt,
    _extract_solution
)


BENCHMARKS = {
    "humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}


# ---------------------------------------------------------------------------
# CAA position helper
# ─────────────────────────────────────────────────────────────────────────────
def _set_caa_position(caa_wrapper, prompt_len: int):
    """Steer from the last prompt token (the one that predicts the first
    generated token).

    ModelWrapper.generate_text uses `prompt_len - 1` as the from_position,
    so we match that here. The SteeringHook handles autoregressive decode
    steps automatically — for seq_len==1 it adds the vector unconditionally.
    """
    if caa_wrapper is None:
        return
    caa_wrapper.set_from_positions(max(prompt_len - 1, 0))


def generate_solutions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: dict,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    model_name: str = "",
    caa_wrapper=None,
) -> dict:
    """Generate `n_samples` completions per problem.

    If caa_wrapper is not None, the CAA hooks are active and the
    from_position is set to the end of the prompt before each generate call.
    """
    model.eval()
    solutions = {}
    task_ids = list(problems.keys())
    print(f"  Generating solutions for {len(task_ids)} problems …")
    print(f"  Chat template: "
          f"{'yes' if _has_chat_template(tokenizer) else 'no — using fallback'}")
    if caa_wrapper is not None:
        active = [i for i, h in caa_wrapper.hooks.items()
                  if h.add_activations is not None]
        info = ", ".join(
            f"layer={i} mult={caa_wrapper.hooks[i].multiplier}"
            for i in active
        )
        print(f"  CAA: {info}")

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    for i in range(0, len(task_ids), batch_size):
        batch_ids = task_ids[i:i + batch_size]
        raw_prompts = [problems[tid]["prompt"] for tid in batch_ids]
        formatted = [_build_prompt(p, tokenizer, model_name) for p in raw_prompts]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # CAA: steer from the end of the prompt.
        # With left padding (default for generation), position -1 is the last
        # real prompt token for every row in the batch, so a single
        # from_position works for all batch entries.
        _set_caa_position(caa_wrapper, inputs["input_ids"].shape[1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_return_sequences=n_samples,
                **gen_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, (tid, raw_prompt) in enumerate(zip(batch_ids, raw_prompts)):
            task_completions = []
            for k in range(n_samples):
                idx = j * n_samples + k
                generated_ids = outputs[idx][input_len:]
                completion = tokenizer.decode(generated_ids,
                                              skip_special_tokens=True)
                solution = _extract_solution(raw_prompt, completion)
                task_completions.append(solution)
            solutions[tid] = task_completions

        done = min(i + batch_size, len(task_ids))
        print(f"    {done}/{len(task_ids)} tasks done", end="\r")

    print()
    return solutions


# ---------------------------------------------------------------------------
# EvalPlus results postprocess
# ---------------------------------------------------------------------------
def parse_scores_from_stdout(stdout: str) -> dict:
    import re
    dict_match = re.search(r"\{['\"]base['\"].*\}", stdout, re.DOTALL)
    if dict_match:
        try:
            import ast
            return ast.literal_eval(dict_match.group(0))
        except Exception:
            pass

    scores: dict = {}
    section = None
    for line in stdout.splitlines():
        line = line.strip()
        if ("base" in line.lower()
                and "extra" not in line.lower()
                and "plus" not in line.lower()):
            section = "base"
        elif "plus" in line.lower() or "extra" in line.lower():
            section = "plus"
        m = re.match(r"pass@(\d+):\s*([\d.]+)", line)
        if m:
            k, v = m.group(1), float(m.group(2))
            key = f"pass@{k}"
            if section:
                scores.setdefault(section, {})[key] = v
            else:
                scores[key] = v

    return scores


def extract_pass_at_k(eval_results: dict) -> dict:
    summary = {}
    for key in ("pass@1", "pass@10", "base", "plus"):
        if key in eval_results:
            summary[key] = eval_results[key]
    if "eval" in eval_results:
        summary["raw"] = eval_results["eval"]
    return summary


# ---------------------------------------------------------------------------
# EvalPlus wrappers
# ---------------------------------------------------------------------------
def load_evalplus_problems(dataset: str) -> dict:
    if dataset == "humaneval":
        return get_human_eval_plus()
    elif dataset == "mbpp":
        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_evalplus_evaluate(solutions_file: Path, dataset: str, results_dir: Path):
    print(f"  Running evalplus evaluate on {solutions_file.name} …")

    stem = solutions_file.stem
    for stale in [
        solutions_file.parent / f"{stem}.eval_results.json",
        solutions_file.parent / f"{stem}_eval_results.json",
        results_dir / f"{stem}.eval_results.json",
        results_dir / f"{stem}_eval_results.json",
    ]:
        if stale.exists():
            print(f"  Removing stale results file: {stale.name}")
            stale.unlink()

    cmd = [
        sys.executable,
        "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(solutions_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(results_dir))

    stdout = result.stdout
    stderr = result.stderr

    if stdout:
        print(stdout)
    (results_dir / "evalplus_stdout.txt").write_text(stdout)
    if stderr:
        (results_dir / "evalplus_stderr.txt").write_text(stderr)

    if result.returncode != 0:
        print(f"  [WARN] evalplus evaluate exited with code {result.returncode}")
        if stderr:
            print(stderr[-2000:])

    candidates = [
        solutions_file.parent / f"{stem}.eval_results.json",
        solutions_file.parent / f"{stem}_eval_results.json",
        results_dir / f"{stem}.eval_results.json",
        results_dir / f"{stem}_eval_results.json",
        Path.cwd() / f"{stem}.eval_results.json",
        Path.cwd() / f"{stem}_eval_results.json",
    ]

    eval_json = None
    for c in candidates:
        if c.exists():
            eval_json = c
            break

    if eval_json:
        dest = results_dir / eval_json.name
        if eval_json.resolve() != dest.resolve():
            shutil.copy(eval_json, dest)
        with open(dest) as f:
            return json.load(f)

    print("  [INFO] eval_results JSON not found — parsing scores from stdout.")
    scores = parse_scores_from_stdout(stdout)
    if scores:
        parsed_path = results_dir / f"{stem}_eval_results.json"
        with open(parsed_path, "w") as f:
            json.dump(scores, f, indent=2)
        return scores

    print("  [WARN] Could not extract scores from stdout either. "
          "Check evalplus_stdout.txt for details.")
    return {}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_benchmark(
    model_path: str,
    benchmarks: list,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    results_root: str,
    caa_bundle: str = None,
    multiplier: float = None,
    tok_pad_side: str = "left",
):
    # ── Determine display name and output root ─────────────────────────
    if caa_bundle:
        ckpt = Path(caa_bundle).resolve()
        model_name = ckpt.name  # e.g. "best_model_bundle"
    else:
        ckpt = Path(model_path).resolve()
        model_name = ckpt.name

    print(f"\n{'='*60}")
    if caa_bundle:
        print(f"CAA bundle : {ckpt}")
    else:
        print(f"Checkpoint : {ckpt}")
    print(f"Model name : {model_name}")
    print(f"Benchmarks : {benchmarks}")
    print(f"{'='*60}")

    # ── Backend loading ────────────────────────────────────────────────
    caa_wrapper = None
    if caa_bundle:
        caa_wrapper = ModelWrapper.from_steering_bundle(
            bundle_dir=str(ckpt),
            device="cuda",
        )
        if multiplier is not None:
            for hook in caa_wrapper.hooks.values():
                if hook.add_activations is not None:
                    hook.multiplier = float(multiplier)

        model = caa_wrapper.model
        tokenizer = caa_wrapper.tokenizer

        active = [i for i, h in caa_wrapper.hooks.items()
                  if h.add_activations is not None]
        info = ", ".join(
            f"layer={i} mult={caa_wrapper.hooks[i].multiplier}"
            for i in active
        )
        print(f"[backend] caa({info})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_model(
            model_path=model_path,
            device_map=device,
            padding_side=tok_pad_side,
        )
        print("[backend] vanilla")

    all_scores = {}

    for benchmark in benchmarks:
        print(f"\n--- Benchmark: {benchmark.upper()} ---")

        if results_root:
            results_dir = Path(results_root) / benchmark / model_name
        else:
            results_dir = ckpt / "evalplus_results" / benchmark
        results_dir.mkdir(parents=True, exist_ok=True)

        problems = load_evalplus_problems(benchmark)

        solutions = generate_solutions(
            model,
            tokenizer,
            problems,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            model_name=model_name,
            caa_wrapper=caa_wrapper,
        )

        solutions_file = results_dir / f"{model_name}_{benchmark}_samples.jsonl"
        os.makedirs(os.path.dirname(solutions_file), exist_ok=True)
        with open(solutions_file, "w") as f:
            for task_id, completions in solutions.items():
                for completion in completions:
                    f.write(json.dumps(
                        {"task_id": task_id, "solution": completion}) + "\n")
        print(f"  Solutions written to {solutions_file}")

        eval_results = run_evalplus_evaluate(solutions_file, benchmark,
                                             results_dir)
        scores = extract_pass_at_k(eval_results)
        all_scores[benchmark] = scores
        print(f"  Scores: {scores}")

    # ── Summary ────────────────────────────────────────────────────────
    summary_path = results_dir.parent / f"{model_name}_evalplus_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_payload = {
        "model": model_name,
        "checkpoint": str(ckpt),
        "scores": all_scores,
    }
    if caa_bundle:
        summary_payload["caa"] = {
            "bundle": str(ckpt),
            "multiplier": (multiplier if multiplier is not None
                           else "bundle_default"),
        }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EvalPlus (HumanEval+ / MBPP+) on HF or CAA checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", nargs="+", default=None,
                        help="One or more paths to model checkpoint directories.")
    parser.add_argument("--caa_bundle", default=None,
                        help="Path to a CAA steering bundle. If provided, "
                             "--model is ignored.")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="Optional CAA multiplier override.")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["humaneval", "mbpp"],
                        choices=["humaneval", "mbpp"],
                        help="Which benchmarks to run.")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Root directory to store results. If omitted, "
                             "results go inside the checkpoint dir.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.caa_bundle and not args.model:
        raise SystemExit("Provide --model or --caa_bundle.")

    if args.multiplier is not None and not args.caa_bundle:
        raise SystemExit("--multiplier only applies with --caa_bundle.")

    all_results = {}

    if args.caa_bundle:
        if not Path(args.caa_bundle).exists():
            raise SystemExit(f"[SKIP] CAA bundle not found: {args.caa_bundle}")
        scores = run_benchmark(
            model_path=None,
            benchmarks=args.benchmarks,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            results_root=args.results_dir,
            caa_bundle=args.caa_bundle,
            multiplier=args.multiplier,
        )
        all_results[args.caa_bundle] = scores
    else:
        for model_path in args.model:
            if not Path(model_path).exists():
                print(f"[SKIP] Checkpoint not found: {model_path}")
                continue
            scores = run_benchmark(
                model_path=model_path,
                benchmarks=args.benchmarks,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
                results_root=args.results_dir,
            )
            all_results[model_path] = scores

    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print("=" * 60)
    for ckpt, scores in all_results.items():
        print(f"\n{ckpt}")
        for bench, s in scores.items():
            print(f"  {bench}: {s}")


if __name__ == "__main__":
    main()