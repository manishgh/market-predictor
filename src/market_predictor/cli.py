from __future__ import annotations

import json
import re
import time as time_module
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pandas as pd
import requests
import typer
from rich.console import Console

from market_predictor.azure_store import AzureBlobStore
from market_predictor.canonical.contracts import SourceCollection
from market_predictor.commands.canonical_data import register_canonical_data_commands
from market_predictor.commands.edge_rebuild import register_edge_rebuild_commands
from market_predictor.commands.intraday_model import register_intraday_model_commands
from market_predictor.commands.intraday_specialists import (
    register_intraday_specialist_commands,
)
from market_predictor.commands.strategy_governance import (
    register_strategy_governance_commands,
)
from market_predictor.commands.swing_collection import register_swing_collection_commands
from market_predictor.commands.swing_research import register_swing_research_commands
from market_predictor.commands.v3_data import register_v3_data_commands
from market_predictor.commands.v3_evaluation import register_v3_evaluation_commands
from market_predictor.commands.v3_features import register_v3_feature_commands
from market_predictor.commands.v3_labels import register_v3_label_commands
from market_predictor.commands.v3_models import register_v3_model_commands
from market_predictor.commands.v3_readiness import register_v3_readiness_commands
from market_predictor.config import Settings, get_settings
from market_predictor.data_quality import sanitize_events_frame
from market_predictor.features import add_finbert, add_finbert_with_scorer, events_to_frame
from market_predictor.global_context import score_flashpoints
from market_predictor.intraday_confirmation import build_intraday_decision_report
from market_predictor.intraday_enrichment import build_enriched_intraday_dataset
from market_predictor.intraday_universe import build_intraday_candidate_universe
from market_predictor.price import fetch_daily_prices, fetch_intraday_prices
from market_predictor.promotion_audit import (
    ProfitabilityAuditConfig,
    build_catalyst_news_audit,
    build_market_regime_audit,
    build_walk_forward_profitability_audit,
)
from market_predictor.schemas import NewsEvent
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.finviz import FinvizSource
from market_predictor.sources.gdelt import GdeltSource
from market_predictor.sources.sec import SecSource

app = typer.Typer(help="Build and serve audited swing and intraday market predictions.")
console = Console()
DEFAULT_MARKET_CONTEXT_PATH = Path("data/external/market_context/market_context_events_scored.parquet")
register_strategy_governance_commands(app, console)
register_canonical_data_commands(app, console)
register_edge_rebuild_commands(app, console)
register_swing_collection_commands(app, console)
register_swing_research_commands(app, console)
register_intraday_model_commands(app, console)
register_intraday_specialist_commands(app, console)
register_v3_data_commands(app, console)
register_v3_feature_commands(app, console)
register_v3_evaluation_commands(app, console)
register_v3_label_commands(app, console)
register_v3_model_commands(app, console)
register_v3_readiness_commands(app, console)


def _parse_tickers(tickers: str | None, fallback: list[str]) -> list[str]:
    if tickers:
        values = [item.strip().upper() for item in tickers.replace(";", ",").split(",")]
        return [item for item in dict.fromkeys(values) if item]
    return fallback


def _parse_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def _parse_end_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.combine(date.fromisoformat(value), time(23, 59, 59), tzinfo=UTC)
    return parsed


def _filter_events_until(frame: pd.DataFrame, end: datetime | None) -> pd.DataFrame:
    if end is None or frame.empty or "timestamp" not in frame.columns:
        return frame
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    return output[output["timestamp"] <= pd.Timestamp(end)].reset_index(drop=True)


def collect_events_for_ticker(
    ticker: str,
    days: int,
    *,
    end: datetime | None = None,
    no_finviz: bool = False,
    no_sec: bool = False,
    score: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    settings = get_settings()
    end = end or datetime.now(UTC)
    start = end - timedelta(days=days)
    events: list[dict[str, object]] = []
    collections: list[dict[str, object]] = []
    errors: list[str] = []

    def record_collection(
        source_family: str,
        *,
        started_at: datetime,
        completed_at: datetime,
        status: str,
        row_count: int = 0,
        error_type: str | None = None,
    ) -> None:
        collection = SourceCollection(
            collection_id=f"{ticker.upper()}-{source_family}-{uuid4().hex}",
            ticker=ticker,
            source_family=source_family,
            requested_start_utc=start,
            requested_end_utc=end,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            status=status,
            row_count=row_count,
            error_type=error_type,
        )
        collections.append(collection.model_dump())

    def append_events(fetched: Sequence[NewsEvent], *, ingested_at: datetime) -> None:
        for event in fetched:
            record = event.to_record()
            record["ingested_at_utc"] = ingested_at
            record["availability_policy"] = "observed"
            events.append(record)

    if settings.has_alpaca:
        started_at = datetime.now(UTC)
        try:
            console.print(f"{ticker}: collecting Alpaca premium news...")
            fetched = AlpacaSource(settings).fetch_news(ticker, start, end=end, limit=50)
            completed_at = datetime.now(UTC)
            append_events(fetched, ingested_at=completed_at)
            record_collection(
                "alpaca",
                started_at=started_at,
                completed_at=completed_at,
                status="observed" if fetched else "observed_empty",
                row_count=len(fetched),
            )
        except Exception as exc:
            record_collection(
                "alpaca",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                status="failed",
                error_type=type(exc).__name__,
            )
            errors.append(f"alpaca:{exc}")
            console.print(f"[yellow]{ticker}: Alpaca collection failed: {exc}[/yellow]")
    else:
        now = datetime.now(UTC)
        record_collection("alpaca", started_at=now, completed_at=now, status="disabled")
        console.print(f"[yellow]{ticker}: skipping Alpaca because keys are not configured.[/yellow]")

    if not no_finviz:
        started_at = datetime.now(UTC)
        try:
            console.print(f"{ticker}: collecting Finviz ticker news...")
            fetched = FinvizSource().fetch_news(ticker, start, end=end, limit=100)
            completed_at = datetime.now(UTC)
            append_events(fetched, ingested_at=completed_at)
            record_collection(
                "finviz",
                started_at=started_at,
                completed_at=completed_at,
                status="observed" if fetched else "observed_empty",
                row_count=len(fetched),
            )
        except Exception as exc:
            record_collection(
                "finviz",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                status="failed",
                error_type=type(exc).__name__,
            )
            errors.append(f"finviz:{exc}")
            console.print(f"[yellow]{ticker}: Finviz news collection failed: {exc}[/yellow]")
    else:
        now = datetime.now(UTC)
        record_collection("finviz", started_at=now, completed_at=now, status="disabled")

    if not no_sec:
        started_at = datetime.now(UTC)
        try:
            console.print(f"{ticker}: collecting SEC filing events...")
            sec_forms = {
                "8-K",
                "10-Q",
                "10-K",
                "S-1",
                "S-3",
                "424B5",
                "424B3",
                "FWP",
                "DEF 14A",
                "SC 13G",
                "SC 13D",
                "4",
            }
            fetched = SecSource(settings).fetch_filings(ticker, start, end=end, forms=sec_forms)
            completed_at = datetime.now(UTC)
            append_events(fetched, ingested_at=completed_at)
            record_collection(
                "sec",
                started_at=started_at,
                completed_at=completed_at,
                status="observed" if fetched else "observed_empty",
                row_count=len(fetched),
            )
        except Exception as exc:
            record_collection(
                "sec",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                status="failed",
                error_type=type(exc).__name__,
            )
            errors.append(f"sec:{exc}")
            console.print(f"[yellow]{ticker}: SEC filing collection failed: {exc}[/yellow]")
    else:
        now = datetime.now(UTC)
        record_collection("sec", started_at=now, completed_at=now, status="disabled")

    frame = events_to_frame(events)
    frame = _filter_events_until(frame, end)
    frame, report = sanitize_events_frame(frame)
    if score and not frame.empty:
        try:
            console.print(f"{ticker}: scoring {len(frame)} events with FinBERT...")
            frame = add_finbert(frame, settings.finbert_model)
        except Exception as exc:
            errors.append(f"finbert:{exc}")
            console.print(f"[yellow]{ticker}: FinBERT scoring failed; raw events kept: {exc}[/yellow]")
    if report.missing_required_rows_removed:
        errors.append(f"sanitize:removed_missing_required={report.missing_required_rows_removed}")
    if report.duplicate_rows_removed:
        errors.append(f"sanitize:removed_duplicates={report.duplicate_rows_removed}")
    if report.future_timestamp_rows:
        errors.append(f"sanitize:removed_future_timestamps={report.future_timestamp_rows}")
    return frame, pd.DataFrame(collections), errors


def _normalize_ohlcv(ticker: str, frame: pd.DataFrame, timeframe: str, *, price_feed: str) -> pd.DataFrame:
    normalized = frame.copy()
    if timeframe == "1d":
        normalized["timestamp"] = pd.to_datetime(normalized["date"], utc=True)
    else:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["symbol"] = ticker.upper()
    normalized["timeframe"] = timeframe
    normalized["source"] = "alpaca"
    normalized["price_feed"] = price_feed.strip().lower()
    normalized["adjustment"] = "all"
    normalized["ingested_at_utc"] = pd.Timestamp.now(tz="UTC")
    columns = [
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "price_feed",
        "adjustment",
        "ingested_at_utc",
    ]
    for col in ["open", "high", "low", "close", "volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized[columns].dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def _merge_ohlcv_manifest(
    existing: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    symbols: list[str],
    timeframes: set[str],
) -> pd.DataFrame:
    if not {"ticker", "timeframe"}.issubset(existing.columns):
        return summary
    replace = existing["ticker"].astype(str).isin(symbols) & existing["timeframe"].astype(str).isin(timeframes)
    merged = pd.concat([existing.loc[~replace], summary], ignore_index=True, sort=False)
    order = [column for column in ["ticker", "timeframe", "rows", "path", "error"] if column in merged.columns]
    return merged.sort_values(["ticker", "timeframe"], na_position="last", kind="stable")[order]


def _write_artifact_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _worker_count(requested: int | None, configured: int, total: int) -> int:
    return max(1, min(int(requested or configured), max(1, total)))


@app.command("download-model")
def download_model_command() -> None:
    """Download the configured FinBERT model into the local Hugging Face cache."""
    from market_predictor.sentiment import download_model

    settings = get_settings()
    download_model(settings.finbert_model)
    console.print(f"Downloaded model cache for {settings.finbert_model}")


@app.command("alpaca-tickers")
def alpaca_tickers(
    out: Path = typer.Option(Path("data/universe/alpaca_tickers.csv"), help="Output ticker universe CSV."),
) -> None:
    """Fetch active/tradable US equity tickers from Alpaca assets."""
    settings = get_settings()
    frame = AlpacaSource(settings).fetch_ticker_universe()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote {len(frame)} Alpaca tickers to {out}")


@app.command()
def collect(
    ticker: str,
    days: int = typer.Option(90, help="Lookback window in calendar days."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD. Defaults to today/now."),
    out: Path = typer.Option(Path("data/raw/events.parquet"), help="Output parquet path."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
) -> None:
    """Collect raw events for a ticker and score them with FinBERT."""
    end = _parse_end_date(end_date)
    frame, collections, _ = collect_events_for_ticker(
        ticker,
        days,
        end=end,
        no_sec=no_sec,
        score=True,
    )
    if frame.empty:
        raise typer.BadParameter("No events collected. Configure Alpaca credentials or widen the date range.")
    frame, report = sanitize_events_frame(frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    collection_path = out.with_name(f"{out.stem}_source_collections.parquet")
    collections.to_parquet(collection_path, index=False)
    console.print({"verification": report.to_record()})
    console.print(f"Wrote {len(frame)} events to {out}")
    console.print(f"Wrote {len(collections)} source collection records to {collection_path}")


@app.command("swing-universe")
def swing_universe(
    out: Path = typer.Option(Path("data/universe/swing_candidates.csv"), help="Output CSV for configured swing symbols."),
    tickers: str | None = typer.Option(None, help="Optional comma-separated symbols to use instead of config."),
) -> None:
    """Write the configured swing watch universe."""
    settings = get_settings()
    values = _parse_tickers(tickers, settings.swing_candidate_tickers)
    frame = pd.DataFrame({"ticker": values, "is_seed": [ticker in settings.swing_seed_tickers for ticker in values]})
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote {len(frame)} swing tickers to {out}")


def _finviz_candidates_from_values(values: list[str], settings: Settings) -> pd.DataFrame:
    cleaned = []
    for value in values:
        symbol = str(value).strip().upper()
        if re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol):
            cleaned.append(symbol)
    current = set(settings.swing_candidate_tickers)
    sector_map = settings.ticker_sector_map
    rows = [
        {
            "ticker": symbol,
            "already_in_universe": symbol in current,
            "sector": sector_map.get(symbol, ""),
            "sector_benchmark": settings.sector_benchmark_for_ticker(symbol),
            "market_benchmark": settings.market_benchmark_ticker,
        }
        for symbol in dict.fromkeys(cleaned)
    ]
    return pd.DataFrame(rows)


FINVIZ_DEFAULT_SECTORS = {
    "technology": "sec_technology",
    "healthcare": "sec_healthcare",
    "financial": "sec_financial",
    "industrial": "sec_industrials",
    "consumer_cyclical": "sec_consumercyclical",
    "energy": "sec_energy",
    "communication": "sec_communicationservices",
    "materials": "sec_basicmaterials",
}

FINVIZ_DEFAULT_CAPS = {
    "mega": "cap_mega",
    "large": "cap_large",
    "mid": "cap_mid",
    "small": "cap_small",
    "micro": "cap_micro",
}


def _redact_url_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "<redacted>", parts.fragment))


def _redact_finviz_auth_text(value: object) -> str:
    return re.sub(r"auth=[^&\s]+", "auth=<redacted>", str(value))


def _finviz_export_url(base_url: str, filters: list[str], auth: str) -> str:
    request = requests.Request(
        "GET",
        base_url,
        params={"v": "111", "f": ",".join(filters), "auth": auth},
    ).prepare()
    if not request.url:
        raise ValueError("Could not build Finviz export URL.")
    return request.url


@app.command("import-finviz")
def import_finviz(
    tickers: str | None = typer.Option(None, help="Pasted symbols from Finviz, separated by commas/spaces/newlines."),
    csv: Path | None = typer.Option(None, help="Optional Finviz CSV export path."),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name for Finviz CSV exports."),
    out: Path = typer.Option(Path("data/universe/finviz_candidates.csv"), help="Cleaned output CSV."),
) -> None:
    """Clean Finviz Elite symbols and compare them with the configured universe."""
    settings = get_settings()
    values: list[str] = []
    if csv:
        frame = pd.read_csv(csv)
        column = symbol_column if symbol_column in frame.columns else frame.columns[0]
        values.extend(frame[column].dropna().astype(str).tolist())
    if tickers:
        values.extend(re.split(r"[\s,;|]+", tickers))
    result = _finviz_candidates_from_values(values, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    console.print(result.head(50))
    console.print(f"Wrote {len(result)} cleaned Finviz candidates to {out}")


@app.command("download-finviz")
def download_finviz(
    url: str = typer.Option(..., help="Finviz Elite export URL. The auth query is not saved."),
    raw_out: Path = typer.Option(Path("data/external/finviz/finviz_export.csv"), help="Raw CSV output path."),
    candidates_out: Path = typer.Option(
        Path("data/universe/finviz_candidates.csv"),
        help="Cleaned ticker candidate output CSV.",
    ),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name in the Finviz export."),
) -> None:
    """Download a Finviz Elite export and extract candidate symbols."""
    response = requests.get(
        url,
        headers={"User-Agent": "market-predictor/0.1"},
        timeout=60,
    )
    response.raise_for_status()
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_bytes(response.content)
    frame = pd.read_csv(raw_out)
    if frame.empty:
        raise typer.BadParameter("Finviz export returned no rows.")
    column = symbol_column if symbol_column in frame.columns else frame.columns[0]
    settings = get_settings()
    candidates = _finviz_candidates_from_values(frame[column].dropna().astype(str).tolist(), settings)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_out, index=False)
    console.print(
        {
            "downloaded": str(raw_out),
            "source": _redact_url_query(url),
            "rows": len(frame),
            "candidate_symbols": len(candidates),
            "candidates": str(candidates_out),
        }
    )
    console.print(candidates.head(50))


@app.command("download-finviz-screeners")
def download_finviz_screeners(
    base_url: str = typer.Option("https://elite.finviz.com/export", help="Finviz Elite export endpoint."),
    sectors: str | None = typer.Option(None, help="Comma-separated sector keys. Defaults to broad sector set."),
    caps: str | None = typer.Option(None, help="Comma-separated cap keys. Defaults to mega,large,mid,small,micro."),
    extra_filters: str = typer.Option("sh_price_o5,sh_avgvol_o500", help="Comma-separated Finviz filters added to each screen."),
    max_per_bucket: int = typer.Option(20, help="Maximum symbols to keep from each sector/cap bucket."),
    sleep_seconds: float = typer.Option(1.5, help="Delay between Finviz export requests to avoid throttling."),
    raw_dir: Path = typer.Option(Path("data/external/finviz/screeners"), help="Raw per-screen CSV directory."),
    out: Path = typer.Option(Path("data/universe/finviz_sector_cap_candidates.csv"), help="Combined candidate CSV."),
    tickers_out: Path = typer.Option(Path("data/universe/finviz_sector_cap_tickers.txt"), help="Comma-separated ticker output."),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name in Finviz exports."),
) -> None:
    """Download Finviz Elite screeners using FINVIZ_ELITE_AUTH from the environment."""
    settings = get_settings()
    token = settings.finviz_elite_auth_value
    if not token:
        raise typer.BadParameter("Set FINVIZ_ELITE_AUTH in the process environment.")
    sector_keys = [key.strip() for key in (sectors.split(",") if sectors else FINVIZ_DEFAULT_SECTORS.keys())]
    cap_keys = [key.strip() for key in (caps.split(",") if caps else FINVIZ_DEFAULT_CAPS.keys())]
    selected_sectors = {key: FINVIZ_DEFAULT_SECTORS[key] for key in sector_keys if key in FINVIZ_DEFAULT_SECTORS}
    selected_caps = {key: FINVIZ_DEFAULT_CAPS[key] for key in cap_keys if key in FINVIZ_DEFAULT_CAPS}
    extras = [item.strip() for item in extra_filters.split(",") if item.strip()]
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary: list[dict[str, object]] = []
    for sector_name, sector_filter in selected_sectors.items():
        for cap_name, cap_filter in selected_caps.items():
            filters = [sector_filter, cap_filter, *extras]
            url = _finviz_export_url(base_url, filters, token)
            raw_path = raw_dir / f"{sector_name}_{cap_name}.csv"
            try:
                response = requests.get(url, headers={"User-Agent": "market-predictor/0.1"}, timeout=60)
                response.raise_for_status()
                raw_path.write_bytes(response.content)
                frame = pd.read_csv(raw_path)
                if frame.empty:
                    summary.append({"sector": sector_name, "cap": cap_name, "rows": 0, "kept": 0})
                    continue
                column = symbol_column if symbol_column in frame.columns else frame.columns[0]
                frame = frame.head(max_per_bucket).copy()
                frame["finviz_sector_bucket"] = sector_name
                frame["finviz_cap_bucket"] = cap_name
                frame["finviz_filters"] = ",".join(filters)
                rows.append(frame)
                summary.append({"sector": sector_name, "cap": cap_name, "rows": len(pd.read_csv(raw_path)), "kept": len(frame)})
            except Exception as exc:
                summary.append(
                    {
                        "sector": sector_name,
                        "cap": cap_name,
                        "rows": 0,
                        "kept": 0,
                        "error": _redact_finviz_auth_text(exc),
                    }
                )
            if sleep_seconds > 0:
                time_module.sleep(sleep_seconds)
    if not rows:
        pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
        raise typer.BadParameter("No Finviz rows downloaded from the requested screens.")
    combined = pd.concat(rows, ignore_index=True)
    column = symbol_column if symbol_column in combined.columns else combined.columns[0]
    candidates = _finviz_candidates_from_values(combined[column].dropna().astype(str).tolist(), settings)
    metadata_cols = [column, "finviz_sector_bucket", "finviz_cap_bucket", "Sector", "Industry", "Market Cap", "Price", "Volume"]
    metadata = combined[[col for col in metadata_cols if col in combined.columns]].copy()
    metadata = metadata.rename(columns={column: "ticker"}).drop_duplicates("ticker")
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    result = candidates.merge(metadata, on="ticker", how="left").drop_duplicates("ticker")
    result = result.sort_values(["already_in_universe", "finviz_sector_bucket", "finviz_cap_bucket", "ticker"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    tickers = result.loc[~result["already_in_universe"], "ticker"].dropna().astype(str).tolist()
    tickers_out.parent.mkdir(parents=True, exist_ok=True)
    tickers_out.write_text(",".join(tickers), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
    console.print({"screens": len(summary), "unique_candidates": len(result), "new_tickers": len(tickers), "out": str(out)})
    console.print(result.head(80))


@app.command("build-intraday-universe")
def build_intraday_universe_command(
    raw: Path = typer.Option(
        Path("data/external/finviz/nasdaq200/nasdaq_liquid_raw_20260707.csv"),
        help="Raw Finviz export CSV.",
    ),
    out: Path = typer.Option(
        Path("data/universe/intraday_nasdaq_activity_latest.csv"),
        help="Ranked intraday candidate CSV.",
    ),
    tickers_out: Path = typer.Option(
        Path("data/universe/intraday_nasdaq_activity_latest_tickers.txt"),
        help="Comma-separated ticker output.",
    ),
    top_n: int = typer.Option(200, help="Number of candidates to keep."),
    min_price: float = typer.Option(2.0, help="Minimum stock price."),
    min_volume: int = typer.Option(500_000, help="Minimum current volume."),
    min_abs_change_pct: float = typer.Option(0.5, help="Minimum absolute day change percent."),
    min_market_cap_m: float = typer.Option(100.0, help="Minimum market cap in millions."),
) -> None:
    """Rank NASDAQ Finviz rows for volatile/high-volume intraday candidates."""
    if not raw.exists():
        raise typer.BadParameter(f"Missing raw Finviz CSV: {raw}")
    frame = pd.read_csv(raw)
    candidates = build_intraday_candidate_universe(
        frame,
        top_n=top_n,
        min_price=min_price,
        min_volume=min_volume,
        min_abs_change_pct=min_abs_change_pct,
        min_market_cap_m=min_market_cap_m,
    )
    if candidates.empty:
        raise typer.BadParameter("No intraday candidates matched the requested filters.")
    out.parent.mkdir(parents=True, exist_ok=True)
    tickers_out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    tickers_out.write_text(",".join(candidates["ticker"].astype(str)), encoding="utf-8")
    console.print({"raw_rows": len(frame), "candidates": len(candidates), "out": str(out)})
    console.print(candidates.head(50))


@app.command("collect-swing")
def collect_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(120, help="Lookback window in calendar days."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD. Defaults to today/now."),
    out_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory for per-ticker event parquet files."),
    no_finviz: bool = typer.Option(False, help="Disable Finviz ticker-news enrichment."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
    score: bool = typer.Option(False, help="Run FinBERT during collection. Default false keeps API download separate."),
    workers: int | None = typer.Option(None, help="Parallel API download workers. Defaults to config performance.max_workers."),
) -> None:
    """Bulk collect Alpaca, Finviz, and SEC events for swing candidates."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    end = _parse_end_date(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Collecting {len(symbols)} tickers with {max_workers} worker(s).")

    def run_symbol(symbol: str) -> dict[str, object]:
        try:
            frame, collections, errors = collect_events_for_ticker(
                symbol,
                days,
                end=end,
                no_finviz=no_finviz,
                no_sec=no_sec,
                score=score,
            )
            path = out_dir / f"{symbol}_events.parquet"
            collection_path = out_dir / f"{symbol}_source_collections.parquet"
            frame.to_parquet(path, index=False)
            collections.to_parquet(collection_path, index=False)
            _, verify = sanitize_events_frame(frame)
            return {
                "ticker": symbol,
                "events": len(frame),
                "path": str(path),
                "source_collections_path": str(collection_path),
                "errors": " | ".join(errors),
                "sources": verify.sources,
            }
        except Exception as exc:
            return {"ticker": symbol, "events": 0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            record = future.result()
            summary.append(record)
            if record.get("error"):
                console.print(f"[red]{symbol}: collection failed: {record['error']}[/red]")
            else:
                console.print(f"{symbol}: wrote {record['events']} events to {record['path']}")
    pd.DataFrame(summary).to_csv(out_dir / "_collection_summary.csv", index=False)
    collection_frames = [
        pd.read_parquet(Path(str(record["source_collections_path"]))) for record in summary if record.get("source_collections_path")
    ]
    if collection_frames:
        pd.concat(collection_frames, ignore_index=True).to_parquet(out_dir / "_source_collections.parquet", index=False)


@app.command("verify-events")
def verify_events(
    events: Path = typer.Option(..., help="Input events parquet."),
    rewrite: bool = typer.Option(False, help="Rewrite the file with sanitized rows."),
) -> None:
    """Sanitize and verify an event parquet file without calling APIs or ML."""
    frame = pd.read_parquet(events)
    clean, report = sanitize_events_frame(frame)
    if rewrite:
        clean.to_parquet(events, index=False)
    console.print(report.to_record())




@app.command("export-ohlcv-artifacts")
def export_ohlcv_artifacts(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(730, help="Calendar days of bars to export."),
    timeframes: str = typer.Option("1d,1h", help="Comma-separated timeframes: 1d,1h,5m,1m."),
    out_dir: Path = typer.Option(Path("data/artifacts/ohlcv"), help="Local OHLCV artifact output root."),
    workers: int | None = typer.Option(None, help="Parallel export workers."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date YYYY-MM-DD; freezes development exports."),
) -> None:
    """Export project-owned OHLCV parquet artifacts for this ML pipeline."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    requested = {item.strip().lower() for item in timeframes.split(",") if item.strip()}
    valid = {"1d", "1h", "5m", "1m"}
    unknown = requested - valid
    if unknown:
        raise typer.BadParameter(f"Unsupported timeframes: {sorted(unknown)}")
    if days < 1:
        raise typer.BadParameter("days must be positive")
    end = _parse_end_date(end_date) or datetime.now(UTC)
    start = end - timedelta(days=days)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))

    def run_symbol(symbol: str) -> list[dict[str, object]]:
        rows = []
        if "1d" in requested:
            daily = fetch_daily_prices(symbol, start, end, settings)
            normalized = _normalize_ohlcv(symbol, daily, "1d", price_feed=settings.alpaca_stock_feed)
            path = out_dir / "1d" / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(path, index=False)
            rows.append({"ticker": symbol, "timeframe": "1d", "rows": len(normalized), "path": str(path)})
        for timeframe in ["1h", "5m", "1m"]:
            if timeframe not in requested:
                continue
            intraday = fetch_intraday_prices(symbol, start, end, settings, timeframe=timeframe)
            normalized = _normalize_ohlcv(symbol, intraday, timeframe, price_feed=settings.alpaca_stock_feed)
            path = out_dir / timeframe / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(path, index=False)
            rows.append({"ticker": symbol, "timeframe": timeframe, "rows": len(normalized), "path": str(path)})
        return rows

    summary: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows = future.result()
                summary.extend(rows)
                console.print(f"{symbol}: exported {sum(int(str(row['rows'])) for row in rows)} OHLCV rows")
            except Exception as exc:
                summary.append({"ticker": symbol, "error": str(exc)})
                console.print(f"[red]{symbol}: OHLCV export failed: {exc}[/red]")
    summary_path = out_dir / "_ohlcv_manifest.csv"
    summary_frame = pd.DataFrame(summary)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        summary_frame = _merge_ohlcv_manifest(existing, summary_frame, symbols=symbols, timeframes=requested)
    summary_frame.to_csv(summary_path, index=False)
    contract: dict[str, object] = {
        "schema_version": "ohlcv.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "columns": [
            "symbol",
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "price_feed",
            "adjustment",
            "ingested_at_utc",
        ],
        "timeframes": sorted(requested),
        "source": "alpaca",
        "price_feed": settings.alpaca_stock_feed,
        "adjustment": "all",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "manifest": str(summary_path),
    }
    _write_artifact_manifest(out_dir / "_schema.json", contract)
    console.print(f"Wrote OHLCV manifest to {summary_path}")


@app.command("azure-upload-artifacts")
def azure_upload_artifacts(
    root: Path = typer.Option(Path("data/artifacts"), help="Local artifact root to upload."),
    blob_prefix: str = typer.Option("", help="Blob prefix under AZURE_BLOB_PREFIX. Defaults to local root name."),
    patterns: str = typer.Option("*.parquet,*.csv,*.json,*.joblib", help="Comma-separated glob patterns."),
) -> None:
    """Upload project artifacts to the configured Azure Blob container."""
    settings = get_settings()
    store = AzureBlobStore(settings)
    prefix = blob_prefix.strip("/") or root.name
    pattern_values = [item.strip() for item in patterns.split(",") if item.strip()]
    uploaded = store.upload_tree(root, blob_prefix=prefix, patterns=pattern_values)
    console.print(f"Uploaded {len(uploaded)} files to container {settings.azure_storage_container}/{settings.azure_prefix}/{prefix}")
    console.print(pd.DataFrame(uploaded).tail(20) if uploaded else "No files uploaded.")


@app.command("audit-promotion-readiness")
def audit_promotion_readiness(
    dataset: Path = typer.Option(..., help="Feature dataset used by the candidate model."),
    predictions: Path = typer.Option(..., help="Out-of-sample predictions CSV from training."),
    target_col: str | None = typer.Option(None, help="Target column. Defaults to target_entry_success_* when available."),
    alignment_audit: Path | None = typer.Option(None, help="Optional existing news/candle alignment audit CSV."),
    out_prefix: Path = typer.Option(Path("data/reports/model_promotion_audit"), help="Output prefix for audit files."),
    probability_col: str = typer.Option("oos_probability", help="OOS probability column."),
    top_fraction: float = typer.Option(0.10, help="Top probability fraction to simulate as trades."),
    min_probability: float | None = typer.Option(None, help="Optional minimum probability floor for selected trades."),
    max_trades_per_period: int | None = typer.Option(
        None,
        help="Optional cap on selected trades per session/day for drawdown-aware selection.",
    ),
) -> None:
    """Build promotion audits for catalyst alignment, regime coverage, and OOS trade economics."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing dataset: {dataset}")
    if not predictions.exists():
        raise typer.BadParameter(f"Missing predictions CSV: {predictions}")
    frame = pd.read_parquet(dataset)
    prediction_frame = pd.read_csv(predictions)
    alignment_frame = pd.read_csv(alignment_audit) if alignment_audit is not None and alignment_audit.exists() else None
    summary, trades, regime_profit = build_walk_forward_profitability_audit(
        dataset=frame,
        predictions=prediction_frame,
        target_col=target_col,
        config=ProfitabilityAuditConfig(
            probability_col=probability_col,
            top_fraction=top_fraction,
            min_probability=min_probability,
            max_trades_per_period=max_trades_per_period,
        ),
    )
    regime = build_market_regime_audit(
        dataset=frame,
        predictions=prediction_frame,
        probability_col=probability_col,
        top_fraction=top_fraction,
    )
    catalyst = build_catalyst_news_audit(dataset=frame, alignment_audit=alignment_frame)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    profitability_out = out_prefix.with_name(out_prefix.name + "_profitability.csv")
    trades_out = out_prefix.with_name(out_prefix.name + "_selected_trades.csv")
    regime_out = out_prefix.with_name(out_prefix.name + "_regime.csv")
    regime_profit_out = out_prefix.with_name(out_prefix.name + "_regime_profitability.csv")
    catalyst_out = out_prefix.with_name(out_prefix.name + "_catalyst.csv")
    summary.to_csv(profitability_out, index=False)
    trades.to_csv(trades_out, index=False)
    regime.to_csv(regime_out, index=False)
    regime_profit.to_csv(regime_profit_out, index=False)
    catalyst.to_csv(catalyst_out, index=False)
    console.print(
        {
            "profitability": str(profitability_out),
            "selected_trades": str(trades_out),
            "regime": str(regime_out),
            "catalyst": str(catalyst_out),
        }
    )
    console.print(summary.iloc[0].to_dict())
    console.print(regime.iloc[0].to_dict())
    console.print(catalyst.iloc[0].to_dict())


@app.command("score-flashpoints")
def score_flashpoints_command(
    events: Path = typer.Option(
        DEFAULT_MARKET_CONTEXT_PATH,
        help="Global/market-context events parquet or CSV.",
    ),
    out: Path = typer.Option(
        Path("data/reports/global_flashpoints_latest.csv"),
        help="Output flashpoint score CSV.",
    ),
    lookback_hours: int = typer.Option(48, help="Lookback window for flashpoint scoring."),
) -> None:
    """Score global flashpoint and commodity-channel risk from market-context news."""
    if not events.exists():
        raise typer.BadParameter(f"Missing events file: {events}")
    frame = pd.read_parquet(events) if events.suffix.lower() == ".parquet" else pd.read_csv(events)
    scored = score_flashpoints(frame, lookback_hours=lookback_hours)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    console.print(scored.head(30))
    console.print(f"Wrote flashpoint scores to {out}")


@app.command("build-intraday-enriched-dataset")
def build_intraday_enriched_dataset_command(
    input_path: Path = typer.Option(..., "--input", help="5m entry/exit dataset parquet."),
    out: Path = typer.Option(..., help="Output enriched training parquet."),
    audit_out: Path = typer.Option(..., help="Output enrichment audit CSV."),
    candidates: Path | None = typer.Option(None, help="Optional Finviz intraday candidate CSV."),
    one_minute_dir: Path | None = typer.Option(None, help="Optional 1m OHLCV parquet directory."),
    benchmark_dir: Path | None = typer.Option(None, help="Optional 5m benchmark OHLCV directory containing QQQ/SPY."),
    event_dirs: str | None = typer.Option(None, help="Comma-separated event directories containing SYMBOL_events.parquet files."),
    market_context: Path | None = typer.Option(
        DEFAULT_MARKET_CONTEXT_PATH,
        help="Optional global market-context events parquet for intraday catalyst features.",
    ),
    setup_only: bool = typer.Option(True, help="Keep only rows passing setup-candidate filters."),
    min_setup_score: float = typer.Option(2.0, help="Minimum setup-candidate score when setup-only is true."),
) -> None:
    """Create setup-filtered, market-relative, 1m-confirmed intraday training rows."""
    if not input_path.exists():
        raise typer.BadParameter(f"Missing input dataset: {input_path}")
    frame = pd.read_parquet(input_path)
    candidate_frame = pd.read_csv(candidates) if candidates is not None and candidates.exists() else None
    enriched, audit = build_enriched_intraday_dataset(
        frame,
        candidates=candidate_frame,
        one_minute_dir=one_minute_dir,
        benchmark_dir=benchmark_dir,
        event_dirs=_parse_path_list(event_dirs),
        market_context_path=market_context,
        setup_only=setup_only,
        min_setup_score=min_setup_score,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out, index=False)
    audit.to_csv(audit_out, index=False)
    summary = {
        "input_rows": len(frame),
        "input_tickers": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        "output_rows": len(enriched),
        "output_tickers": int(enriched["ticker"].nunique()) if not enriched.empty else 0,
        "setup_only": setup_only,
        "min_setup_score": min_setup_score,
        "event_dirs": event_dirs,
        "market_context": str(market_context) if market_context else None,
        "target_entry_success_rate": float(pd.to_numeric(enriched.get("target_entry_success_12b"), errors="coerce").mean())
        if not enriched.empty and "target_entry_success_12b" in enriched.columns
        else None,
        "out": str(out),
        "audit": str(audit_out),
    }
    console.print(summary)
    console.print(audit.sort_values("rows", ascending=False).head(30))


@app.command("build-intraday-decision-report")
def build_intraday_decision_report_command(
    scores: Path = typer.Option(..., help="Latest 5m entry/exit score CSV."),
    one_minute_dir: Path = typer.Option(..., help="Directory containing 1m OHLCV parquet files."),
    candidates: Path | None = typer.Option(None, help="Optional Finviz intraday candidate CSV."),
    out: Path = typer.Option(Path("data/reports/intraday_decision_latest.csv"), help="Output decision report CSV."),
) -> None:
    """Merge 5m entry model scores with latest 1m confirmation features."""
    if not scores.exists():
        raise typer.BadParameter(f"Missing scores CSV: {scores}")
    if not one_minute_dir.exists():
        raise typer.BadParameter(f"Missing 1m directory: {one_minute_dir}")
    score_frame = pd.read_csv(scores)
    candidate_frame = pd.read_csv(candidates) if candidates is not None and candidates.exists() else None
    report = build_intraday_decision_report(
        scores=score_frame,
        one_minute_dir=one_minute_dir,
        candidates=candidate_frame,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    display_cols = [
        col
        for col in [
            "ticker",
            "intraday_decision",
            "entry_model_probability",
            "entry_model_rank",
            "one_minute_confirmation_signal",
            "one_minute_dist_vwap",
            "one_minute_return_15m",
            "one_minute_volume_burst_15m",
            "above_opening_range",
            "intraday_theme",
            "intraday_candidate_score",
        ]
        if col in report.columns
    ]
    console.print(report[display_cols].head(80))
    console.print(f"Wrote intraday decision report to {out}")


@app.command("collect-market-context")
def collect_market_context(
    days: int = typer.Option(730, help="Market/global news lookback window in calendar days."),
    out: Path = typer.Option(
        Path("data/external/market_context/market_context_events.parquet"),
        help="Output market context events parquet.",
    ),
    score_sentiment: bool = typer.Option(True, help="Run FinBERT on global market/news context rows."),
    include_gdelt: bool = typer.Option(True, help="Include GDELT global flashpoint/news context."),
    gdelt_max_records_per_query: int = typer.Option(75, help="Maximum GDELT articles per configured query."),
) -> None:
    """Collect broad market/global news that can affect all tickers without being ticker-specific."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    start = datetime.now(UTC) - timedelta(days=days)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    if include_gdelt:
        try:
            gdelt_events, gdelt_errors = GdeltSource().fetch_context_events_with_errors(
                start,
                max_records_per_query=gdelt_max_records_per_query,
            )
            rows.extend([event.to_record() for event in gdelt_events])
            errors.extend(f"gdelt_context:{error}" for error in gdelt_errors)
        except Exception as exc:
            errors.append(f"gdelt_context:{exc}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "timestamp", "source", "title", "url", "summary", "text", "raw"])
    else:
        frame = sanitize_events_frame(frame)[0]
        if score_sentiment:
            scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
            frame = add_finbert_with_scorer(frame, scorer, batch_size=settings.finbert_batch_size)
        if "raw" in frame.columns:
            frame["raw"] = frame["raw"].map(
                lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, dict) else value
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    summary = {
        "rows": len(frame),
        "sources": frame["source"].value_counts().to_dict() if "source" in frame else {},
        "errors": errors,
        "out": str(out),
    }
    (out.parent / "market_context_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    console.print(summary)


@app.command("build-market-context-from-proxies")
def build_market_context_from_proxies(
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Raw event directory containing proxy ETF/index event files.",
    ),
    symbols: str = typer.Option(
        "SPY,QQQ,DIA,IWM,RSP,XLK,SMH,XBI,IBB,XAR,ITA,ARKF,ARKK,TLT,HYG,LQD,GLD,USO,UUP,KWEB,BITO",
        help="Comma-separated proxy symbols used as historical market/global context.",
    ),
    out: Path = typer.Option(
        Path("data/external/market_context/market_context_events.parquet"),
        help="Output market context parquet.",
    ),
) -> None:
    """Build historical broad-market context from proxy ETF/index event stores."""
    proxy_symbols = _parse_tickers(symbols, [])
    frames = []
    summary = []
    for symbol in proxy_symbols:
        path = raw_dir / f"{symbol}_events.parquet"
        if not path.exists():
            summary.append({"proxy": symbol, "rows": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            if frame.empty:
                summary.append({"proxy": symbol, "rows": 0, "path": str(path)})
                continue
            frame = frame.copy()
            frame["market_proxy_symbol"] = symbol
            frame["ticker"] = "MARKET"
            frame["source"] = "market_proxy:" + symbol + ":" + frame["source"].astype(str)
            frames.append(frame)
            summary.append({"proxy": symbol, "rows": len(frame), "path": str(path)})
        except Exception as exc:
            summary.append({"proxy": symbol, "rows": 0, "error": str(exc)})
    if not frames:
        raise typer.BadParameter("No proxy event files found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = sanitize_events_frame(combined)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    summary_path = out.parent / "market_context_proxy_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    console.print({"rows": len(combined), "out": str(out), "summary": str(summary_path)})
