#!/bin/sh
set -eu

if [ "${SYNC_AZURE_RELEASE_ON_STARTUP:-false}" = "true" ]; then
    market-predictor azure-sync-serving-release --root /app
fi

exec market-predictor serve-api --host 0.0.0.0 --port 8000
