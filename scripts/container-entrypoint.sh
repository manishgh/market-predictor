#!/bin/sh
set -eu

exec market-predictor-prod serve-api --host 0.0.0.0 --port 8000
