#!/bin/bash
# docker-entrypoint.sh — wait for dependencies, then run the CMD.
#
# Used as ENTRYPOINT in the Dockerfile. For api service, this is mostly
# a no-op (just exec the CMD). For trainer/drift-monitor, the wait loop
# can be expanded.
set -euo pipefail

echo "[entrypoint] $(date -Iseconds) Starting dc_real_time container..."
echo "[entrypoint] CMD: $*"

# Wait for Redis if REDIS_URL is set (api service uses this for feature store)
if [ -n "${REDIS_URL:-}" ]; then
    echo "[entrypoint] REDIS_URL=$REDIS_URL — waiting for Redis..."
    # Extract host:port from redis://host:port
    REDIS_HOST=$(echo "$REDIS_URL" | sed -E 's|^redis://||' | cut -d: -f1)
    REDIS_PORT=$(echo "$REDIS_URL" | sed -E 's|^redis://||' | cut -d: -f2)
    REDIS_PORT=${REDIS_PORT:-6379}
    for i in $(seq 1 30); do
        if curl -fsS "http://$REDIS_HOST:$REDIS_PORT/ping" >/dev/null 2>&1; then
            echo "[entrypoint] Redis is up."
            break
        fi
        # Try redis-cli if available, fall back to TCP check
        if (echo -e "PING\r\n" | timeout 1 bash -c "cat >/dev/tcp/$REDIS_HOST/$REDIS_PORT" 2>/dev/null); then
            echo "[entrypoint] Redis port is open."
            break
        fi
        echo "[entrypoint] Redis not ready, waiting... ($i/30)"
        sleep 2
    done
fi

# Wait for model file to be available (mounted volume)
if [ -n "${MODEL_PATH:-}" ]; then
    echo "[entrypoint] Waiting for model at $MODEL_PATH..."
    for i in $(seq 1 30); do
        if [ -f "$MODEL_PATH" ]; then
            echo "[entrypoint] Model found."
            break
        fi
        echo "[entrypoint] Model not found yet, waiting... ($i/30)"
        sleep 1
    done
fi

# Run the original CMD
exec "$@"
