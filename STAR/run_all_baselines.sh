#!/bin/bash
set -e
cd /home/qmo/build/agentTools/STAR

PYTHON=./eval-env/bin/python3
DATA=/tmp/star_hammer_eval.jsonl
N=500

for MODEL in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
    OUT="results_${MODEL}.jsonl"
    echo ""
    echo "========================================="
    echo "Running baseline: $MODEL"
    echo "========================================="
    $PYTHON run_baseline_inference.py \
        --model "models/$MODEL" \
        --data "$DATA" \
        --n $N \
        --batch-size 8 \
        --output "$OUT"
done

echo ""
echo "========================================="
echo "ALL BASELINES COMPLETE — Summary"
echo "========================================="
$PYTHON - <<'EOF'
import jsonlines, glob, re

for path in sorted(glob.glob("results_Qwen3-*.jsonl")):
    model = re.search(r"results_(Qwen3-.+)\.jsonl", path).group(1)
    results = list(jsonlines.open(path))
    n = len(results)
    n_err = sum(1 for r in results if r["score"] == -1)
    valid = [r["score"] for r in results if r["score"] >= 0]
    mean = sum(valid)/len(valid) if valid else 0
    perfect = sum(1 for s in valid if s >= 1.0)/len(valid) if valid else 0
    print(f"  {model:<15}  n={n}  fmt_err={n_err:>3} ({n_err/n:.0%})  mean={mean:.4f}  perfect={perfect:.4f}")
EOF
