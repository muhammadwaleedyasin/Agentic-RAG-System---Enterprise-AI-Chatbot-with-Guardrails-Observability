#!/bin/bash

# Production entrypoint script for RAG System

set -e

# Default values
WORKER_ID=${WORKER_ID:-"worker_$(hostname)"}
WORKER_PORT=${WORKER_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-INFO}
REDIS_URL=${REDIS_URL:-"redis://localhost:6379"}

echo "Starting RAG Worker: $WORKER_ID"
echo "Worker Port: $WORKER_PORT"
echo "Log Level: $LOG_LEVEL"
echo "Redis URL: $REDIS_URL"

# Wait for Redis to be available
echo "Waiting for Redis..."
until curl -f "$REDIS_URL" &>/dev/null; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done
echo "Redis is available"

# Initialize application
echo "Initializing application..."
python -c "
import sys
sys.path.insert(0, '/app')
from src.optimization.caching_layer import create_caching_system
from src.optimization.scaling_manager import create_scaling_manager
import asyncio

async def init():
    cache = create_caching_system(redis_url='$REDIS_URL')
    manager = create_scaling_manager()
    print('Application components initialized')

asyncio.run(init())
"

# Start the application
echo "Starting RAG application..."
exec python -m src.main \
    --worker-id "$WORKER_ID" \
    --port "$WORKER_PORT" \
    --log-level "$LOG_LEVEL" \
    --redis-url "$REDIS_URL"