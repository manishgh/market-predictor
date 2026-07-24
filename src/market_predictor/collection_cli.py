from __future__ import annotations

from market_predictor.cli import app as source_app
from market_predictor.cli_surface import filtered_app

COLLECTION_COMMANDS = frozenset(
    {
        "alpaca-tickers",
        "azure-upload-artifacts",
        "collect",
        "collect-gdelt-context",
        "collect-market-context",
        "collect-seeking-alpha",
        "collect-seeking-alpha-universe",
        "collect-swing",
        "download-finviz",
        "download-finviz-screeners",
        "download-model",
        "export-ohlcv-artifacts",
        "import-finviz",
        "seeking-alpha-limits",
        "seeking-alpha-token",
        "seeking-alpha-token-status",
        "swing-universe",
    }
)

app = filtered_app(
    source_app,
    allowed_commands=COLLECTION_COMMANDS,
    help_text="Collect and export raw Market Predictor source data.",
)
