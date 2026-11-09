#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source runtime.sh

python prepare.py
python reward.py
if [ ! -f results/base/metrics.json ]; then
    python evaluate.py Qwen/Qwen3.5-2B --output results/base
fi
if [ ! -f results/base/validation_metrics.json ]; then
    python validate.py Qwen/Qwen3.5-2B --name base
fi

for run in grpo_exact grpo_shaped dr_grpo_shaped; do
    if [ ! -f "runs/$run/summary.json" ]; then
        python train.py "$run" --steps 300
    fi
    if [ ! -f "checkpoints/$run/export_complete.json" ]; then
        python export.py "$run"
    fi
    if [ ! -f "results/$run/validation_metrics.json" ]; then
        python validate.py "checkpoints/$run" --name "$run"
    fi
    if [ ! -f "results/$run/metrics.json" ]; then
        python evaluate.py "checkpoints/$run" --output "results/$run"
    fi
done
