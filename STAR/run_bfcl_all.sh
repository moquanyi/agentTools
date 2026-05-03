#!/bin/bash
set -e
BFCL=/home/qmo/build/agentTools/STAR/eval-env/bin/bfcl
PYTHON=/home/qmo/build/agentTools/STAR/eval-env/bin/python3
BFCL_DIR=/mnt/data/gorilla/berkeley-function-call-leaderboard
PORT=1053
export LOCAL_SERVER_PORT=$PORT

MODELS=(
  "Qwen/Qwen3-1.7B-FC:/home/qmo/build/agentTools/STAR/models/Qwen3-1.7B:16"
  "Qwen/Qwen3-4B-FC:/home/qmo/build/agentTools/STAR/models/Qwen3-4B:12"
  "Qwen/Qwen3-8B-FC:/home/qmo/build/agentTools/STAR/models/Qwen3-8B:6"
)

for entry in "${MODELS[@]}"; do
  MODEL_ID="${entry%%:*}"
  rest="${entry#*:}"
  MODEL_PATH="${rest%%:*}"
  BATCH_SIZE="${rest##*:}"
  echo ""
  echo "========================================"
  echo "Model: $MODEL_ID  batch_size=$BATCH_SIZE  ($(date +%H:%M))"
  echo "========================================"

  # Start server
  $PYTHON /home/qmo/build/agentTools/STAR/serve_model.py \
    --model "$MODEL_PATH" --port $PORT --batch-size $BATCH_SIZE &
  SERVER_PID=$!

  # Wait for server to be ready
  until curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; do sleep 2; done
  echo "Server ready (PID $SERVER_PID)"

  # Run BFCL generation
  cd $BFCL_DIR
  $BFCL generate \
    --model "$MODEL_ID" \
    --test-category non_live \
    --skip-server-setup \
    --local-model-path "$MODEL_PATH" 2>&1 | grep -v "^$"

  # Kill server and free GPU memory
  kill $SERVER_PID 2>/dev/null
  wait $SERVER_PID 2>/dev/null || true
  sleep 3
  echo "Server stopped."
done

echo ""
echo "========================================"
echo "ALL GENERATION DONE — Running evaluation"
echo "========================================"
cd $BFCL_DIR
$BFCL evaluate \
  --model "Qwen/Qwen3-0.6B-FC" "Qwen/Qwen3-1.7B-FC" "Qwen/Qwen3-4B-FC" "Qwen/Qwen3-8B-FC" \
  --test-category non_live 2>&1

$BFCL scores \
  --model "Qwen/Qwen3-0.6B-FC" "Qwen/Qwen3-1.7B-FC" "Qwen/Qwen3-4B-FC" "Qwen/Qwen3-8B-FC" 2>&1
