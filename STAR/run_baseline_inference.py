"""
STAR Real Baseline Inference + Evaluation
Runs Qwen3-0.6B (untrained) on the Hammer benchmark using HuggingFace transformers.
"""

import sys
import json
import argparse
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
from simrl import (
    reward_func,
    compute_function_calling_reward,
    get_rouge_score,
    get_ground_truth_from_label,
    extract_tools_from_prompt,
    check_tool_calls_valid,
    parse_generation,
)


def load_data(path, n=None):
    records = []
    with jsonlines.open(path) as f:
        for i, obj in enumerate(f):
            if n and i >= n:
                break
            records.append(obj)
    return records


SAFE_BATCH_SIZES = {
    "0.6B": 8,
    "1.7B": 8,
    "4B":   4,
    "8B":   1,
}

def safe_batch_size(model_path):
    for key, bs in SAFE_BATCH_SIZES.items():
        if key in model_path:
            return bs
    return 1  # conservative default for unknown sizes


def run_inference(model_path, records, batch_size=None):
    if batch_size is None:
        batch_size = safe_batch_size(model_path)
    print(f"Loading model: {model_path}  (batch_size={batch_size})")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {device}. Running inference on {len(records)} samples...")

    generations = []
    for i in tqdm(range(0, len(records), batch_size), desc="Inference"):
        batch = records[i : i + batch_size]
        # Each prompt already ends with `<think>` (added by messages_to_trainset.py)
        prompts = [rec["inputs"] for rec in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3584,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )

        for j, output_ids in enumerate(out):
            input_len = enc["input_ids"][j].shape[0]
            new_tokens = output_ids[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            # Strip trailing <|im_end|>
            text = text.replace("<|im_end|>", "").strip()
            generations.append(text)

    return generations


def score_outputs(records, generations, verbose=False):
    queries, prompts, labels = [], [], []
    for rec, gen in zip(records, generations):
        inp = rec["inputs"]
        tgt = rec["targets"]
        # Build full query: prompt already ends with `<think>`, generation continues from there
        full_query = inp + gen
        queries.append(full_query)
        prompts.append(inp)
        labels.append(tgt)

    result = reward_func(queries, prompts, labels, print_info=verbose)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--data", default="/tmp/star_hammer_eval.jsonl")
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=None, help="auto-selected if not set")
    parser.add_argument("--output", default="baseline_results.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    records = load_data(args.data, n=args.n)
    print(f"Loaded {len(records)} eval samples from {args.data}")

    generations = run_inference(args.model, records, batch_size=args.batch_size)

    result = score_outputs(records, generations, verbose=args.verbose)
    rewards = result["rewards"]
    format_rewards = result["extra_logs"]["format_rewards"]
    answer_rewards = result["extra_logs"]["answer_rewards"]

    valid_mask = rewards >= 0
    n_total = len(rewards)

    print("\n" + "=" * 55)
    print("STAR Baseline Results — Qwen3-0.6B (no STAR training)")
    print("=" * 55)
    print(f"  n_total              : {n_total}")
    print(f"  n_format_errors      : {(rewards == -1).sum().item()}")
    print(f"  format_error_rate    : {(rewards == -1).float().mean().item():.4f}")
    if valid_mask.sum() > 0:
        print(f"  mean_score           : {rewards[valid_mask].mean().item():.4f}")
        print(f"  mean_answer_score    : {answer_rewards[valid_mask].mean().item():.4f}")
        print(f"  mean_format_score    : {format_rewards[valid_mask].mean().item():.4f}")
        print(f"  perfect_score_rate   : {(rewards[valid_mask] == 1.0).float().mean().item():.4f}")
    print("=" * 55)

    with jsonlines.open(args.output, "w") as out:
        for i, (rec, gen, r, fr, ar) in enumerate(
            zip(records, generations, rewards.tolist(), format_rewards.tolist(), answer_rewards.tolist())
        ):
            out.write({
                "id": rec.get("id", i),
                "generation": gen,
                "score": r,
                "format_score": fr,
                "answer_score": ar,
            })
    print(f"Per-sample results saved to: {args.output}")


if __name__ == "__main__":
    main()
