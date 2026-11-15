#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url=${URL:-http://127.0.0.1:8000}
data=data/requests.jsonl
output=${1:-results/$(date +%Y%m%d-%H%M%S)}
rpm=${2:-120}
count=${3:-$(wc -l < "$data" | tr -d ' ')}
revision=b968826d9c46dd6066d109eabc6255188de91218
mkdir -p "$(dirname "$output")"
mkdir "$output"

reset_cache() {
    python - "$url" <<'PY'
import sys
from urllib.request import Request, urlopen

with urlopen(Request(sys.argv[1] + '/reset_prefix_cache', method='POST'), timeout=10):
    pass
PY
}

trap reset_cache EXIT
reset_cache
rps=$(awk -v rpm="$rpm" 'BEGIN { print rpm / 60 }')
if [ "$count" = 300s ]; then
    length=(--benchmark-duration 300 --benchmark-grace-period inf)
else
    length=(--request-count "$count")
fi

"${AIPERF:-aiperf}" profile --url "$url" --model Qwen/Qwen3-8B \
  --tokenizer-revision "$revision" --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file "$data" --dataset-sampling-strategy sequential --random-seed 42 \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --request-rate "$rps" --arrival-pattern constant "${length[@]}" \
  --streaming --request-timeout-seconds 120 --output-artifact-dir "$output/profile"
