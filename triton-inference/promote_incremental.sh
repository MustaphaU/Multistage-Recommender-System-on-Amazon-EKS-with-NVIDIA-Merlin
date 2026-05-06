#!/usr/bin/env bash
set -euo pipefail

TRITON_HOST=${1:-"triton-service.default.svc.cluster.local"}
TRITON_HTTP_PORT=${2:-8000}

TRITON_URL="http://${TRITON_HOST}:${TRITON_HTTP_PORT}"

echo "Waiting for Triton to be ready at $TRITON_URL..."
until curl -sf "${TRITON_URL}/v2/health/ready" > /dev/null; do
    sleep 5
done

echo "Promoting dlrm_ranking to latest version..."
curl -sf -X POST "${TRITON_URL}/v2/repository/models/dlrm_ranking/load"

echo "Promoting query_tower to latest version..."
curl -sf -X POST "${TRITON_URL}/v2/repository/models/query_tower/load"

echo "Promote complete. Active models:"
curl -sf -X POST "${TRITON_URL}/v2/repository/index" \
    -H "Content-Type: application/json" -d '{"ready": true}' \
    | python3 -c "import sys,json; [print(f\"  {m['name']} v{m.get('version','?')} [{m.get('state','?')}]\") for m in json.load(sys.stdin)]"
