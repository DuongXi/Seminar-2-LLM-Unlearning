#!/usr/bin/env python3
"""
run_evalplus.py

Runs EvalPlus (HumanEval+ and MBPP+) on one or more model checkpoints.
Supports both plain HuggingFace checkpoints and LoRA fine-tuned checkpoints
(via peft/transformers). Results are saved in the same directory as the
checkpoint, under a subfolder named after the benchmark.

Usage:
    python run_evalplus.py --checkpoints /path/to/ckpt1 /path/to/ckpt2 [options]

Requirements:
    pip install evalplus transformers peft accelerate torch
"""

import argparse
import json
from pathlib import Path
import sys
import subprocess
import shutil
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from utils.model import load_model


BENCHMARKS = {
    "humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

# Instruction wrappers for common model families when no chat template exists.
# Maps a substring of the model name/path (lowercased) to a (prefix, suffix) pair
# that wraps the raw problem prompt.
_FALLBACK_TEMPLATES: list[tuple[str, tuple[str, str]]] = [
    (
        "deepseek",
        ("{prompt}\n\n", ""),
    ),  # deepseek-coder-instruct uses plain prompts well
    ("codellama", ("[INST] {prompt} [/INST]\n", "")),
    ("mistral", ("[INST] {prompt} [/INST]\n", "")),
    ("llama", ("[INST] {prompt} [/INST]\n", "")),
    (
        "starchat",
        ("<|system|>\n<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n", ""),
    ),
    (
        "wizardcoder",
        (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n",
            "",
        ),
    ),
]


def _has_chat_template(tokenizer) -> bool:
    """Return True if the tokenizer has a usable chat template."""
    tmpl = getattr(tokenizer, "chat_template", None)
    return bool(tmpl)


def _build_prompt(raw_prompt: str, tokenizer, model_name: str) -> str:
    """
    Wrap the raw problem prompt in the model's preferred format.

    Priority:
      1. tokenizer.apply_chat_template  (most accurate for instruct models)
      2. Fallback per-family templates  (for models without a chat template)
      3. Raw prompt as-is               (base / completion models)
    """
    instruction = (
        "Complete the following Python function. "
        "Return ONLY the completed function body with no extra commentary.\n\n"
        + raw_prompt
    )

    if _has_chat_template(tokenizer):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # fall through

    name_lower = model_name.lower()
    for key, (prefix_tmpl, suffix) in _FALLBACK_TEMPLATES:
        if key in name_lower:
            return prefix_tmpl.format(prompt=instruction) + suffix

    # Base / completion model — feed the raw prompt directly so the model
    # continues the function signature naturally.
    return raw_prompt


def _extract_solution(raw_prompt: str, completion: str) -> str:
    """
    Build the final solution string that evalplus will execute.

    evalplus executes the `solution` field as a standalone Python snippet.
    For HumanEval the prompt already contains the function signature; we
    prepend it so the generated body is syntactically complete.

    We also strip any markdown fences the model may have added.
    """
    # Strip markdown code fences
    code = completion
    fence_start = re.search(r"```(?:python)?\n?", code)
    if fence_start:
        code = code[fence_start.end() :]
    code = code.split("```")[0]  # drop everything after a closing fence

    # If the completion already re-states the function signature, use it as-is.
    if (
        raw_prompt.strip().splitlines()[0].strip().startswith("def ")
        and raw_prompt.strip().splitlines()[0].strip() in code
    ):
        return code

    # Otherwise prepend the original prompt (signature + docstring) so the
    # result is a complete, runnable function definition.
    return raw_prompt + code


def generate_solutions(
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    problems: dict,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    model_name: str = "",
) -> dict:
    """
    Generate `n_samples` completions per problem.
    Returns a dict: {task_id: [solution_str, ...]}
    where each solution_str is a complete, runnable Python snippet.
    """
    model.eval()
    solutions = {}
    task_ids = list(problems.keys())
    print(f"  Generating solutions for {len(task_ids)} problems …")
    print(
        f"  Chat template: {'yes' if _has_chat_template(tokenizer) else 'no — using fallback'}"
    )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    for i in range(0, len(task_ids), batch_size):
        batch_ids = task_ids[i : i + batch_size]
        raw_prompts = [problems[tid]["prompt"] for tid in batch_ids]
        formatted = [_build_prompt(p, tokenizer, model_name) for p in raw_prompts]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

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
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
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
    """
    Extract pass@k scores from evalplus stdout.

    evalplus (≥0.3) prints something like:

        {'base': {'pass@1': 0.7317}, 'plus': {'pass@1': 0.6890}}

    or older versions:

        base tests
        pass@1: 0.7317

        base + extra tests
        pass@1: 0.6890
    """
    import re

    # Try the dict repr format first (newer evalplus)
    dict_match = re.search(r"\{['\"]base['\"].*\}", stdout, re.DOTALL)
    if dict_match:
        try:
            # ast.literal_eval handles single-quoted Python dicts
            import ast

            return ast.literal_eval(dict_match.group(0))
        except Exception:
            pass

    # Fall back to line-by-line parsing
    scores: dict = {}
    section = None
    for line in stdout.splitlines():
        line = line.strip()
        if (
            "base" in line.lower()
            and "extra" not in line.lower()
            and "plus" not in line.lower()
        ):
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
    """Pull out the pass@k numbers from evalplus result dict."""
    summary = {}
    for key in ("pass@1", "pass@10", "base", "plus"):
        if key in eval_results:
            summary[key] = eval_results[key]
    # newer evalplus format nests under "eval"
    if "eval" in eval_results:
        summary["raw"] = eval_results["eval"]
    return summary


# ---------------------------------------------------------------------------
# EvalPlus wrappers
# ---------------------------------------------------------------------------
def load_evalplus_problems(dataset: str) -> dict:
    """Load the EvalPlus problem set for the given dataset name."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    elif dataset == "mbpp":
        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_evalplus_evaluate(solutions_file: Path, dataset: str, results_dir: Path):
    """
    Call `evalplus.evaluate` as a subprocess so we get the official pass@k
    scores. Writes a JSON summary to results_dir.

    evalplus writes <stem>_eval_results.json next to the samples file, but the
    subprocess may run with a different CWD, so we search several candidate
    locations and fall back to parsing stdout.
    """

    print(f"  Running evalplus evaluate on {solutions_file.name} …")

    # evalplus caches results next to the samples file. If that file exists
    # from a previous broken/incomplete run it will be reloaded and may cause
    # a KeyError (missing 'eval' key). Delete any stale copies first.
    # evalplus uses both dot-separated and underscore-separated suffixes
    # depending on the version, so we clean up both variants.
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
        "-m",
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        str(solutions_file),
    ]
    # Run with CWD = results_dir so relative-path output lands there
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(results_dir))

    stdout = result.stdout
    stderr = result.stderr

    # Always show evalplus output so the user sees scores in their terminal
    if stdout:
        print(stdout)

    # Save raw stdout / stderr for reference
    (results_dir / "evalplus_stdout.txt").write_text(stdout)
    if stderr:
        (results_dir / "evalplus_stderr.txt").write_text(stderr)

    if result.returncode != 0:
        print(f"  [WARN] evalplus evaluate exited with code {result.returncode}")
        if stderr:
            print(stderr[-2000:])

    # -----------------------------------------------------------------------
    # 1. Try to find the eval_results JSON.
    #    evalplus writes  <samples_stem>_eval_results.json  but the CWD for
    #    the subprocess may differ from solutions_file.parent, so we check
    #    several candidate locations.
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 2. Fall back: parse pass@k numbers directly from evalplus stdout.
    #    evalplus prints lines like:
    #      pass@1: 0.7317  (base tests)
    #      pass@1: 0.6890  (base + extra tests)
    #    or in newer versions:
    #      {'pass@1': 0.73}
    # -----------------------------------------------------------------------
    print("  [INFO] eval_results JSON not found — parsing scores from stdout.")
    scores = parse_scores_from_stdout(stdout)
    if scores:
        # Persist the parsed scores so downstream summary still works
        parsed_path = results_dir / f"{stem}_eval_results.json"
        with open(parsed_path, "w") as f:
            json.dump(scores, f, indent=2)
        return scores

    print(
        "  [WARN] Could not extract scores from stdout either. "
        "Check evalplus_stdout.txt for details."
    )
    return {}

def run_benchmark(
    model_path: str,
    benchmarks: list,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    results_root: str,
    tok_pad_side:str="left"
):
    # Show model name
    ckpt = Path(model_path).resolve()
    model_name = ckpt.name
    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt}")
    print(f"Model name : {model_name}")
    print(f"Benchmarks : {benchmarks}")
    print(f"{'='*60}")

    tokenizer, model = load_model(model_path, padding_side=tok_pad_side)

    all_scores = {}

    # Benchmark
    for benchmark in benchmarks:
        print(f"\n--- Benchmark: {benchmark.upper()} ---")

        # Output directory: <results_root>/<benchmark>/<model_name>/
        if results_root:
            results_dir = Path(results_root) / benchmark / model_name
        else:
            results_dir = ckpt / "evalplus_results" / benchmark
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load problems
        problems = load_evalplus_problems(benchmark)

        # Generate solutions
        solutions = generate_solutions(
            model,
            tokenizer,
            problems,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            model_name=model_name,
        )

        # Save solutions in evalplus jsonl format
        solutions_file = results_dir / f"{model_name}_{benchmark}_samples.jsonl"
        with open(solutions_file, "w") as f:
            for task_id, completions in solutions.items():
                for completion in completions:
                    f.write(
                        json.dumps({"task_id": task_id, "solution": completion}) + "\n"
                    )
        print(f"  Solutions written to {solutions_file}")

        # Evaluate
        eval_results = run_evalplus_evaluate(solutions_file, benchmark, results_dir)
        scores = extract_pass_at_k(eval_results)
        all_scores[benchmark] = scores

        print(f"  Scores: {scores}")

    # Write a combined summary JSON next to the checkpoint
    summary_path = results_dir.parent / f"{model_name}_evalplus_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": model_name, "checkpoint": str(ckpt), "scores": all_scores}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Free VRAM before the next checkpoint
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_scores

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EvalPlus (HumanEval+ / MBPP+) on HuggingFace checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",
                        nargs="+", required=True, help="One or more paths to model checkpoint directories.")
    parser.add_argument("--benchmarks", 
                        nargs="+", default=["humaneval", "mbpp"], choices=["humaneval", "mbpp"],help="Which benchmarks to run.")
    parser.add_argument("--n_samples",
                        type=int, default=1, help="Number of solutions to sample per problem (pass@k denominator).")
    parser.add_argument("--max_new_tokens",
                        type=int, default=512,help="Maximum new tokens per generated solution.")
    parser.add_argument("--temperature",
                        type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--batch_size",
                         type=int, default=4, help="Number of problems to batch together during generation.")
    parser.add_argument("--results_dir",
                        type=str, default=None,
                        help=("Root directory to store results. "
                              "If omitted, results go inside <checkpoint>/evalplus_results/<benchmark>/. "
                              "If provided, results go in <results_dir>/<benchmark>/<model_name>/."))
    return parser.parse_args()

def main():
    args = parse_args()

    all_results = {}
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
