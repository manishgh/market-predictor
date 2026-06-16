#!/usr/bin/env sh
set -eu

LOOKBACK_DAYS="${LOOKBACK_DAYS:-3}"
WORKERS="${WORKERS:-8}"
EXPORT_DAYS="${EXPORT_DAYS:-730}"
TIMEFRAMES="${TIMEFRAMES:-1d,1h}"
LIVE_DIR="${LIVE_DIR:-data/live}"
ARTIFACT_DIR="${ARTIFACT_DIR:-data/artifacts}"
MODELS_DIR="${MODELS_DIR:-models}"

market-predictor live-once \
  --live-dir "$LIVE_DIR" \
  --lookback-days "$LOOKBACK_DAYS" \
  --workers "$WORKERS"

market-predictor export-ohlcv-artifacts \
  --days "$EXPORT_DAYS" \
  --timeframes "$TIMEFRAMES" \
  --out-dir "$ARTIFACT_DIR/ohlcv" \
  --workers "$WORKERS"

market-predictor azure-upload-artifacts \
  --root "$ARTIFACT_DIR"

market-predictor live-train-event \
  --live-dir "$LIVE_DIR"

market-predictor azure-publish-models \
  --models-dir "$MODELS_DIR"
