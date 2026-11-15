#!/usr/bin/env bash
set -euo pipefail

model=Qwen/Qwen3-8B
revision=b968826d9c46dd6066d109eabc6255188de91218
export VLLM_SERVER_DEV_MODE=1

if [ -n "${MODEL_PATH:-}" ]; then
    exec vllm serve "$MODEL_PATH" --served-model-name "$model" --host 127.0.0.1 --port "${PORT:-8000}" "$@"
fi
exec vllm serve "$model" --revision "$revision" --host 127.0.0.1 --port "${PORT:-8000}" "$@"
