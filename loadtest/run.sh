#!/usr/bin/env bash
# Load-test POST /v1/classify on the local compose stack.
#
#   docker compose -f compose.yaml -f loadtest/compose.benchmark.yaml up --build --detach
#   loadtest/run.sh [users] [seconds]
#
# Prints locust's summary and writes CSV files to loadtest/results/.
set -euo pipefail
cd "$(dirname "$0")/.."

users="${1:-16}"
seconds="${2:-30}"
base="http://localhost:8000"
email="bench-$(date +%s)-$RANDOM@example.com"
credentials="{\"email\": \"$email\", \"password\": \"benchmark password\"}"
json_field() { python3 -c "import json, sys; print(json.load(sys.stdin)['$1'])"; }

curl --silent --fail "$base/v1/auth/register" -H 'Content-Type: application/json' -d "$credentials" > /dev/null
docker compose exec -T api provenance-api set-plan "$email" pro > /dev/null
token=$(curl --silent --fail "$base/v1/auth/login" -H 'Content-Type: application/json' -d "$credentials" | json_field access_token)
PROVENANCE_API_KEY=$(curl --silent --fail "$base/v1/api-keys" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $token" -d '{"name": "benchmark"}' | json_field key)
export PROVENANCE_API_KEY

mkdir -p loadtest/results
uvx locust --locustfile loadtest/locustfile.py --host "$base" --headless \
  --users "$users" --spawn-rate "$users" --run-time "${seconds}s" \
  --csv "loadtest/results/users-$users" --only-summary
