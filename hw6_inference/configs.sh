#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

config=${1:-baseline}
tune=${2:-$config}
args=()
case "$tune" in
    baseline) ;;
    chunk1024) args+=(--max-num-batched-tokens 1024) ;;
    chunk4096) args+=(--max-num-batched-tokens 4096) ;;
    seq32) args+=(--max-num-seqs 32) ;;
    *) echo "Unknown engine settings: $tune" >&2; exit 2 ;;
esac

case "$config" in
    baseline|chunk1024|chunk4096|seq32) ;;
    v1)
        export VLLM_USE_V2_MODEL_RUNNER=0
        args+=(--no-async-scheduling)
        ;;
    ngram|ngram_gpu)
        export VLLM_USE_V2_MODEL_RUNNER=0
        args+=(--no-async-scheduling)
        args+=(--speculative-config "{\"method\":\"$config\",\"num_speculative_tokens\":5,\"prompt_lookup_min\":3,\"prompt_lookup_max\":5}")
        ;;
    *) echo "Unknown configuration: $config" >&2; exit 2 ;;
esac

exec bash serve.sh "${args[@]}"
