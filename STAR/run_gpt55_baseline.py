"""
STAR Baseline Evaluation — GPT-5.5 via Grammarly LLM Proxy
Uses native OpenAI tool_calls format, scored with simrl.py reward logic.
"""

import sys
import json
import argparse
import jsonlines
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, ".")
from simrl import compute_function_calling_reward, get_rouge_score

CLIENT = OpenAI(
    base_url="https://us.api.openai.com/v1",
    api_key="default",
    default_headers={"X-LLM-Proxy-Calling-Service": "star-baseline-research"},
)
MODEL = "gpt-5.5"


def tools_to_openai_format(tools: list) -> list:
    """Convert hammer/xlam tool format to OpenAI function-calling format."""
    result = []
    for t in tools:
        params = t.get("parameters", {})
        # Normalize to JSON Schema object format
        if "properties" not in params:
            properties = {}
            required = []
            for k, v in params.items():
                prop = {"description": v.get("description", ""), "type": "string"}
                raw_type = v.get("type", "str")
                if "int" in raw_type:
                    prop["type"] = "integer"
                elif "float" in raw_type or "number" in raw_type:
                    prop["type"] = "number"
                elif "bool" in raw_type:
                    prop["type"] = "boolean"
                properties[k] = prop
                if "optional" not in raw_type:
                    required.append(k)
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
        else:
            schema = params

        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return result


def call_gpt55(record: dict, retries: int = 3) -> dict:
    messages = record["messages"][:-1]  # all but final assistant turn
    tools = record.get("tools", [])
    gt_msg = record["messages"][-1]    # ground truth assistant response

    openai_tools = tools_to_openai_format(tools) if tools else None

    for attempt in range(retries):
        try:
            kwargs = dict(model=MODEL, messages=messages, temperature=0, max_tokens=1024)
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"
            resp = CLIENT.chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt == retries - 1:
                return {"id": record["id"], "error": str(e), "score": -1, "format_score": -1, "answer_score": 0}
            time.sleep(2 ** attempt)

    msg = resp.choices[0].message

    # Extract predicted tool calls
    pred_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            pred_calls.append({"name": tc.function.name, "arguments": args})

    pred_text = msg.content or ""

    # Extract ground truth tool calls
    gt_calls = []
    if gt_msg.get("tool_calls"):
        for tc in gt_msg["tool_calls"]:
            gt_calls.append({"name": tc["name"], "arguments": tc.get("arguments", {})})

    gt_text = gt_msg.get("content") or ""

    # Score
    if gt_calls:
        score = compute_function_calling_reward(gt_calls, pred_calls) if pred_calls else 0.0
    else:
        score = get_rouge_score(pred_text, gt_text) if gt_text else (1.0 if not pred_calls else 0.0)

    return {
        "id": record["id"],
        "pred_calls": pred_calls,
        "pred_text": pred_text[:200],
        "gt_calls": gt_calls,
        "score": score,
        "format_score": 1,
        "answer_score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hammer_messages.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls")
    parser.add_argument("--output", default="results_gpt-5.5.jsonl")
    args = parser.parse_args()

    records = []
    with jsonlines.open(args.data) as f:
        for i, obj in enumerate(f):
            if args.n and i >= args.n:
                break
            records.append(obj)
    print(f"Loaded {len(records)} samples | model={MODEL} | workers={args.workers}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_gpt55, rec): rec for rec in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="GPT-5.5"):
            results.append(fut.result())

    scores  = [r["score"] for r in results if r.get("score", -1) >= 0]
    n_err   = sum(1 for r in results if r.get("score", -1) < 0)
    n       = len(results)
    mean    = sum(scores) / len(scores) if scores else 0
    perfect = sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0

    print(f"\n{'='*55}")
    print(f"STAR Baseline — {MODEL} (Hammer benchmark)")
    print(f"{'='*55}")
    print(f"  n_total              : {n}")
    print(f"  n_errors             : {n_err}  ({n_err/n:.1%})")
    print(f"  mean_score           : {mean:.4f}")
    print(f"  perfect_score_rate   : {perfect:.4f}")
    print(f"{'='*55}")

    with jsonlines.open(args.output, "w") as out:
        out.write_all(sorted(results, key=lambda r: r["id"]))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
