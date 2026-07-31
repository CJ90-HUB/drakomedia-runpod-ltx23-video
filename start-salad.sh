#!/bin/bash
set -euo pipefail

uvicorn salad_server:app --host 127.0.0.1 --port 8080 &
api_pid=$!
/usr/local/bin/salad-http-job-queue-worker &
queue_pid=$!

trap 'kill -TERM "$api_pid" "$queue_pid" 2>/dev/null || true; wait "$api_pid" "$queue_pid" 2>/dev/null || true' TERM INT
wait -n "$api_pid" "$queue_pid"
