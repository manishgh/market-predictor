param(
    [string]$WorkingDir = "c:\project\market-predictor"
)

$ErrorActionPreference = "Stop"
Set-Location $WorkingDir

Write-Output "Starting Intraday Data Collection Pipeline..."
.venv\Scripts\Activate.ps1

Write-Output "Step 1: Running Universe Selection..."
python -m market_predictor.cli screen-edge-rebuild-intraday-universe `
    --canonical-dir data/canonical/edge_rebuild_intraday_5m_20260731 `
    --first-session 2026-06-01 `
    --last-session 2026-07-01 `
    --out-dir data/research/intraday_universe_selection_extended

Write-Output "Step 2: Planning Intraday History..."
python -m market_predictor.cli plan-edge-rebuild-selected-session-one-minute `
    --selection-dir data/research/intraday_universe_selection_extended `
    --policy configs/edge_rebuild_selected_session_one_minute.toml `
    --out-dir data/research/intraday_history_plan_extended

Write-Output "Step 3: Collecting Intraday History (This will fail if network is blocked)..."
python -m market_predictor.cli collect-edge-rebuild-intraday-history `
    --plan-dir data/research/intraday_history_plan_extended `
    --out-dir data/raw/edge_rebuild_selected_session_1m_extended

Write-Output "Step 4: Training Intraday Development Model..."
python -m market_predictor.cli train-edge-rebuild-intraday-development `
    --dataset-dir data/features/edge_rebuild_intraday_dataset_causal_20260802_v2 `
    --out-dir data/features/intraday_development_v3

Write-Output "Pipeline Complete."
