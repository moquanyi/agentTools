"""
Batched OpenAI-compatible server for BFCL evaluation.
Collects concurrent requests into batches, runs one model.generate() call per batch.
Usage: python serve_model.py --model models/Qwen3-0.6B --port 1053 --batch-size 8
"""

import argparse
import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
tokenizer = None
model = None
model_name = None
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1024
request_queue: asyncio.Queue = None


# ── request/response schemas ────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None
    extra_body: Optional[Dict] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# ── batch processing loop ────────────────────────────────────────────────────

async def batch_worker():
    """Drain the queue in batches, run model.generate once per batch."""
    loop = asyncio.get_event_loop()
    while True:
        # Collect up to BATCH_SIZE items, wait at most 50ms for a full batch
        items = []
        try:
            item = await asyncio.wait_for(request_queue.get(), timeout=0.05)
            items.append(item)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.01)
            continue

        # Drain remaining items without blocking
        while len(items) < BATCH_SIZE:
            try:
                items.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        prompts = [it["prompt"] for it in items]
        max_new = max(it["max_tokens"] for it in items)

        try:
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            ).to(model.device)

            with torch.no_grad():
                out = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    ),
                )

            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            for i, item in enumerate(items):
                new_toks = out[i][int(input_lens[i]):]
                text = tokenizer.decode(new_toks, skip_special_tokens=False)
                text = text.replace("<|im_end|>", "").strip()
                item["future"].set_result(text)

        except Exception as e:
            for item in items:
                if not item["future"].done():
                    item["future"].set_exception(e)


def _make_completion_response(text: str, n_in: int, n_out: int) -> dict:
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": model_name, "object": "model"}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put({
        "prompt": req.prompt,
        "max_tokens": req.max_tokens or MAX_NEW_TOKENS,
        "future": fut,
    })
    text = await fut
    n_in = len(tokenizer.encode(req.prompt, truncation=True, max_length=8192))
    n_out = len(tokenizer.encode(text))
    return _make_completion_response(text, n_in, n_out)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    hf_tools = None
    if req.tools:
        hf_tools = [
            {"type": "function", "function": {
                "name": (t.get("function") or t)["name"],
                "description": (t.get("function") or t).get("description", ""),
                "parameters": (t.get("function") or t).get("parameters", {}),
            }} for t in req.tools
        ]

    prompt = tokenizer.apply_chat_template(
        req.messages, tools=hf_tools, tokenize=False, add_generation_prompt=True,
    ) + "<think>\n"

    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put({
        "prompt": prompt,
        "max_tokens": req.max_tokens or MAX_NEW_TOKENS,
        "future": fut,
    })
    text = await fut

    tool_calls_raw = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    msg: Dict[str, Any] = {"role": "assistant"}
    if tool_calls_raw:
        parsed = []
        for tc in tool_calls_raw:
            try:
                obj = json.loads(tc.strip())
                parsed.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))},
                })
            except Exception:
                pass
        msg["content"] = None
        msg["tool_calls"] = parsed
    else:
        msg["content"] = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    n_in = len(tokenizer.encode(prompt, truncation=True, max_length=8192))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": len(tokenizer.encode(text)), "total_tokens": n_in + len(tokenizer.encode(text))},
    }


@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_worker())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global tokenizer, model, model_name, BATCH_SIZE, request_queue

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    request_queue = asyncio.Queue()

    model_name = args.model
    print(f"Loading {model_name}  (batch_size={BATCH_SIZE})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"Model loaded. Serving on :{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
