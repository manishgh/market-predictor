from __future__ import annotations

from market_predictor.cli import app as source_app
from market_predictor.cli_surface import filtered_app

COLLECTION_COMMANDS = frozenset(
    {
        "alpaca-tickers",
        "azure-upload-artifacts",
        "collect",
        "collect-alpaca-news-history",
        "collect-edge-intraday-microstructure-history",
        "collect-edge-rebuild-intraday-history",
        "collect-edge-rebuild-swing-history",
        "collect-edge-live-global-context",
        "collect-edge-prospective-broker-actions",
        "collect-edge-sec-filings",
        "collect-intraday-specialist-one-minute",
        "collect-market-context",
        "collect-sp500-official-source-archive",
        "collect-swing",
        "download-finviz",
        "download-finviz-screeners",
        "download-model",
        "export-ohlcv-artifacts",
        "import-finviz",
        "swing-universe",
    }
)

app = filtered_app(
    source_app,
    allowed_commands=COLLECTION_COMMANDS,
    help_text="Collect and export raw Market Predictor source data.",
)


if __name__ == "__main__":
    app()
