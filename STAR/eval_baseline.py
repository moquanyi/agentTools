"""
STAR Baseline Evaluation Script
Evaluates the scoring pipeline on provided ground-truth data.
In a real baseline run, replace `simulate_model_output` with actual model inference.
"""

import sys
import json
import argparse
import jsonlines
from typing import List, Dict, Any

import torch
sys.path.insert(0, ".")
from simrl import (
    reward_func,
    compute_function_calling_reward,
    get_rouge_score,
    parse_generation,
    get_ground_truth_from_label,
    extract_tools_from_prompt,
    check_tool_calls_valid,
)


def simulate_model_output(prompt: str, target: str) -> str:
    """
    Placeholder: returns ground-truth as model output (upper-bound oracle).
    Replace with actual vLLM/sglang inference for real baseline evaluation.
    """
    # target already starts with </think>, so just prepend the opening tag
    return "<think>\n" + target


def evaluate_dataset(data_path: str, num_samples: int = None, verbose: bool = False) -> Dict[str, Any]:
    records = []
    with jsonlines.open(data_path) as f:
        for i, obj in enumerate(f):
            if num_samples and i >= num_samples:
                break
            records.append(obj)

    queries, prompts, labels = [], [], []
    for rec in records:
        inp = rec["inputs"]
        tgt = rec["targets"]
        generation = simulate_model_output(inp, tgt)
        full_query = inp + generation
        queries.append(full_query)
        prompts.append(inp)
        labels.append(tgt)

    result = reward_func(queries, prompts, labels, print_info=verbose)
    rewards = result["rewards"]
    format_rewards = result["extra_logs"]["format_rewards"]
    answer_rewards = result["extra_logs"]["answer_rewards"]

    valid_mask = rewards >= 0
    n_total = len(rewards)
    n_valid = valid_mask.sum().item()
    n_format_err = (rewards == -1).sum().item()

    metrics = {
        "n_total": n_total,
        "n_format_errors": n_format_err,
        "format_error_rate": n_format_err / n_total,
        "mean_score": rewards[valid_mask].mean().item() if n_valid > 0 else 0.0,
        "mean_answer_score": answer_rewards[valid_mask].mean().item() if n_valid > 0 else 0.0,
        "mean_format_score": format_rewards[valid_mask].mean().item() if n_valid > 0 else 0.0,
        "perfect_score_rate": (rewards[valid_mask] == 1.0).float().mean().item() if n_valid > 0 else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="STAR Baseline Evaluation")
    parser.add_argument("--data", default="/tmp/star_eval_data.jsonl", help="Eval data in inputs/targets format")
    parser.add_argument("--n", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Evaluating on: {args.data}")
    metrics = evaluate_dataset(args.data, num_samples=args.n, verbose=args.verbose)
    print("\n=== STAR Baseline Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
