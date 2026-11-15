#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rpm=${1:-120}
output=${2:-results/load-${rpm}rpm-$(date +%Y%m%d-%H%M%S)}
exec bash benchmark.sh "$output" "$rpm" 300s
