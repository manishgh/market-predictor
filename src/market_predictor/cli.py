from __future__ import annotations

import json
import re
import time as time_module
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pandas as pd
import requests
import typer
from rich.console import Console

from market_predictor.azure_store import AzureBlobStore
from market_predictor.canonical.contracts import SourceCollection
from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.commands.canonical_data import register_canonical_data_commands
from market_predictor.commands.intraday_model import register_intraday_model_commands
from market_predictor.commands.ranking import register_ranking_commands
from market_predictor.commands.swing_model import register_swing_model_commands
from market_predictor.commands.v3_data import register_v3_data_commands
from market_predictor.commands.v3_evaluation import register_v3_evaluation_commands
from market_predictor.commands.v3_features import register_v3_feature_commands
from market_predictor.commands.v3_labels import register_v3_label_commands
from market_predictor.commands.v3_models import register_v3_model_commands
from market_predictor.commands.v3_readiness import register_v3_readiness_commands
from market_predictor.config import Settings, get_settings
from market_predictor.data_quality import sanitize_events_frame
from market_predictor.deployment import (
    DEFAULT_ACTIVE_POINTER,
    DEFAULT_RELEASE_PREFIX,
    publish_serving_release,
    rollback_serving_release,
    sync_active_serving_release,
)
from market_predictor.entry_exit import (
    ENTRY_EXIT_SCHEMA_VERSION,
    EntryExitLabelConfig,
    build_entry_exit_dataset,
    merge_entry_exit_context,
    score_entry_exit_frame,
    train_entry_exit_model,
)
from market_predictor.feature_store import LiveFeatureStore, LiveFeatureStoreConfig
from market_predictor.features import (
    add_event_taxonomy,
    add_finbert,
    add_finbert_with_scorer,
    align_events_to_trading_dates,
    build_daily_dataset,
    events_to_frame,
    source_family_for_source,
)
from market_predictor.global_context import score_flashpoints
from market_predictor.intraday_confirmation import build_intraday_decision_report
from market_predictor.intraday_enrichment import build_enriched_intraday_dataset
from market_predictor.intraday_universe import build_intraday_candidate_universe
from market_predictor.live_features import LIVE_ARTIFACT_TYPES, LIVE_SCHEMA_VERSIONS, LiveMode
from market_predictor.model import DEFAULT_FEATURES
from market_predictor.prediction_service import serving_routes_from_config
from market_predictor.price import fetch_daily_prices, fetch_intraday_prices
from market_predictor.promotion_audit import (
    ProfitabilityAuditConfig,
    build_catalyst_news_audit,
    build_market_regime_audit,
    build_walk_forward_profitability_audit,
)
from market_predictor.schemas import NewsEvent
from market_predictor.sources import AlpacaSource, FinvizSource, GdeltSource, RedditSource, SeekingAlphaRapidApiSource
from market_predictor.sources.gdelt import DEFAULT_GDELT_CONTEXT_QUERIES
from market_predictor.sources.sec import SecSource

app = typer.Typer(help="Build and serve audited swing and intraday market predictions.")
console = Console()
DEFAULT_MARKET_CONTEXT_PATH = Path("data/external/market_context/market_context_events_scored.parquet")
register_ranking_commands(app, console)
register_canonical_data_commands(app, console)
register_swing_model_commands(app, console)
register_intraday_model_commands(app, console)
register_v3_data_commands(app, console)
register_v3_feature_commands(app, console)
register_v3_evaluation_commands(app, console)
register_v3_label_commands(app, console)
register_v3_model_commands(app, console)
register_v3_readiness_commands(app, console)


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", help="API bind host."),
    port: int = typer.Option(8000, help="API bind port."),
    reload: bool = typer.Option(False, help="Enable uvicorn reload for local development."),
) -> None:
    """Serve the typed prediction API for swing, intraday, and unified views."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("uvicorn is not installed. Run `python -m pip install -e .` first.") from exc
    uvicorn.run("market_predictor.api:app", host=host, port=port, reload=reload)


@app.command("publish-live-features")
def publish_live_features(
    mode: str = typer.Option(..., help="Feature mode: swing or intraday."),
    input_path: Path = typer.Option(..., help="Canonical inference feature artifact to publish."),
    live_dir: Path = typer.Option(Path("data/live"), help="Managed live feature root."),
) -> None:
    """Atomically publish an integrity-checked feature snapshot for API serving."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"swing", "intraday"}:
        raise typer.BadParameter("mode must be swing or intraday")
    live_mode = cast(LiveMode, normalized_mode)
    expected_type = LIVE_ARTIFACT_TYPES[live_mode]
    frame, canonical_manifest = load_canonical_artifact(
        input_path,
        expected_type=expected_type,
        allow_research=False,
    )
    schema_column = "swing_feature_schema_version" if normalized_mode == "swing" else "intraday_feature_schema_version"
    schemas = set(frame[schema_column].astype(str).unique()) if schema_column in frame else set()
    expected_schema = LIVE_SCHEMA_VERSIONS[live_mode]
    if schemas != {expected_schema}:
        raise typer.BadParameter(f"canonical {normalized_mode} features do not match schema {expected_schema}")
    feeds = set(frame["price_feed"].astype(str).str.lower().str.strip().unique())
    if len(feeds) != 1:
        raise typer.BadParameter("canonical inference features must contain exactly one price feed")
    price_feed = next(iter(feeds))
    store = LiveFeatureStore(
        Path("."),
        LiveFeatureStoreConfig(
            swing_path=live_dir / "features/swing.parquet",
            intraday_path=live_dir / "features/intraday.parquet",
        ),
    )
    manifest = store.publish(
        live_mode,
        frame,
        price_feed=price_feed,
        feature_schema_version=expected_schema,
        source_artifact_sha256=str(canonical_manifest["artifact_sha256"]),
        source_artifact_type=expected_type,
    )
    console.print(manifest)


def _daily_training_columns(frame: pd.DataFrame, horizon_days: int) -> list[str]:
    keep = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        f"entry_next_open_{horizon_days}d",
        f"future_return_{horizon_days}d",
        f"target_up_{horizon_days}d",
        f"target_bucket_{horizon_days}d",
        f"target_favorable_before_stop_{horizon_days}d",
    }
    keep.update(DEFAULT_FEATURES)
    return [column for column in frame.columns if column in keep]


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
    no_reddit: bool = False,
    no_finviz: bool = False,
    no_seeking_alpha: bool = False,
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

    if not no_reddit and settings.has_reddit:
        started_at = datetime.now(UTC)
        try:
            console.print(f"{ticker}: collecting Reddit mentions and comments...")
            fetched = RedditSource(settings).fetch_mentions(ticker, start)
            completed_at = datetime.now(UTC)
            append_events(fetched, ingested_at=completed_at)
            record_collection(
                "reddit",
                started_at=started_at,
                completed_at=completed_at,
                status="observed" if fetched else "observed_empty",
                row_count=len(fetched),
            )
        except Exception as exc:
            record_collection(
                "reddit",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                status="failed",
                error_type=type(exc).__name__,
            )
            errors.append(f"reddit:{exc}")
            console.print(f"[yellow]{ticker}: Reddit collection failed: {exc}[/yellow]")
    else:
        now = datetime.now(UTC)
        record_collection("reddit", started_at=now, completed_at=now, status="disabled")
        if not no_reddit:
            console.print(f"[yellow]{ticker}: skipping Reddit because credentials are not configured.[/yellow]")

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

    if not no_seeking_alpha and settings.has_seeking_alpha_rapidapi:
        started_at = datetime.now(UTC)
        try:
            console.print(f"{ticker}: collecting Seeking Alpha news/analysis via RapidAPI...")
            sa_events, sa_errors = SeekingAlphaRapidApiSource(settings).fetch_events_with_errors(ticker, start)
            completed_at = datetime.now(UTC)
            append_events(sa_events, ingested_at=completed_at)
            status = "partial" if sa_errors and sa_events else "failed" if sa_errors else "observed" if sa_events else "observed_empty"
            record_collection(
                "seeking_alpha",
                started_at=started_at,
                completed_at=completed_at,
                status=status,
                row_count=len(sa_events),
                error_type="ProviderFeedError" if sa_errors else None,
            )
            errors.extend(f"seeking_alpha:{error}" for error in sa_errors)
        except Exception as exc:
            record_collection(
                "seeking_alpha",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                status="failed",
                error_type=type(exc).__name__,
            )
            errors.append(f"seeking_alpha:{exc}")
            console.print(f"[yellow]{ticker}: Seeking Alpha collection failed: {exc}[/yellow]")
    else:
        now = datetime.now(UTC)
        record_collection("seeking_alpha", started_at=now, completed_at=now, status="disabled")

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


def collect_events_frame(
    ticker: str,
    days: int,
    *,
    end: datetime | None = None,
    no_reddit: bool = False,
    score: bool = True,
) -> pd.DataFrame:
    frame, _, _ = collect_events_for_ticker(ticker, days, end=end, no_reddit=no_reddit, score=score)
    return frame


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


def write_seeking_alpha_snapshot(ticker: str, out: Path) -> Path:
    settings = get_settings()
    snapshot = SeekingAlphaRapidApiSource(settings).fetch_quant_snapshot(ticker)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([snapshot])
    if out.exists():
        existing = pd.read_csv(out)
        frame = pd.concat([existing, row], ignore_index=True)
    else:
        frame = row
    frame.to_csv(out, index=False)
    return out


def _worker_count(requested: int | None, configured: int, total: int) -> int:
    return max(1, min(int(requested or configured), max(1, total)))


def _count_nonzero_feature_days(
    events: pd.DataFrame,
    daily_features: pd.DataFrame,
    column: str,
    bucket: str | None = None,
) -> int:
    dates = set(events["date"] if bucket is None else events.loc[events["event_time_bucket"] == bucket, "date"])
    if column not in daily_features.columns or not dates:
        return 0
    matching_dates = daily_features.index.intersection(dates)
    series = pd.to_numeric(daily_features.loc[matching_dates, column], errors="coerce")
    return int((series.fillna(0).abs() > 1e-12).sum())


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
    no_reddit: bool = typer.Option(False, help="Disable Reddit enrichment."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha enrichment."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
) -> None:
    """Collect raw events for a ticker and score them with FinBERT."""
    end = _parse_end_date(end_date)
    frame, collections, _ = collect_events_for_ticker(
        ticker,
        days,
        end=end,
        no_reddit=no_reddit,
        no_seeking_alpha=no_seeking_alpha,
        no_sec=no_sec,
        score=True,
    )
    if frame.empty:
        raise typer.BadParameter("No events collected. Configure Alpaca/Reddit credentials or widen the date range.")
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
    auth: str | None = typer.Option(None, help="Finviz Elite auth token. Defaults to FINVIZ_ELITE_AUTH."),
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
    """Download multiple Finviz Elite screener exports across sector and cap buckets."""
    settings = get_settings()
    token = auth or settings.finviz_elite_auth
    if not token:
        raise typer.BadParameter("Provide --auth or set FINVIZ_ELITE_AUTH in .env.")
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
    no_reddit: bool = typer.Option(False, help="Disable Reddit enrichment."),
    no_finviz: bool = typer.Option(False, help="Disable Finviz ticker-news enrichment."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha enrichment."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
    score: bool = typer.Option(False, help="Run FinBERT during collection. Default false keeps API download separate."),
    workers: int | None = typer.Option(None, help="Parallel API download workers. Defaults to config performance.max_workers."),
) -> None:
    """Bulk collect Alpaca news, Reddit chatter, and Seeking Alpha events for swing candidates."""
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
                no_reddit=no_reddit,
                no_finviz=no_finviz,
                no_seeking_alpha=no_seeking_alpha,
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


@app.command("verify-swing")
def verify_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing per-ticker event parquet files."),
    rewrite: bool = typer.Option(False, help="Rewrite each file with sanitized rows."),
    out: Path = typer.Option(Path("data/raw/swing/_verification_summary.csv"), help="Output verification summary CSV."),
) -> None:
    """Bulk verify swing event files with per-ticker isolation."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        path = raw_dir / f"{symbol}_events.parquet"
        if not path.exists():
            rows.append({"ticker": symbol, "rows_out": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            clean, report = sanitize_events_frame(frame)
            if rewrite:
                clean.to_parquet(path, index=False)
            record = report.to_record()
            record["ticker"] = symbol
            rows.append(record)
        except Exception as exc:
            rows.append({"ticker": symbol, "rows_out": 0, "error": str(exc)})
            console.print(f"[red]{symbol}: verification failed: {exc}[/red]")
    summary = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    console.print(summary[["ticker", "rows_out", "duplicate_rows_removed", "missing_required_rows_removed"]].head(40))
    console.print(f"Wrote verification summary to {out}")


@app.command("score-swing-events")
def score_swing_events(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Directory containing raw per-ticker event parquet files.",
    ),
    out_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean_scored"),
        help="Directory for FinBERT-scored per-ticker event parquet files.",
    ),
    text_mode: str = typer.Option(
        "title_summary",
        help="FinBERT input: title, title_summary, or text. Title-summary is recommended for catalyst scoring.",
    ),
    max_length: int = typer.Option(128, min=1, max=512, help="Maximum FinBERT input tokens."),
    batch_size: int | None = typer.Option(None, min=1, help="Inference batch size. Defaults to configuration."),
    force: bool = typer.Option(False, help="Rescore even if an output file already exists."),
) -> None:
    """Run FinBERT on existing downloaded events without calling news APIs."""
    from market_predictor.sentiment import (
        SENTIMENT_INPUT_MODES,
        FinbertScorer,
        build_sentiment_inputs,
    )

    settings = get_settings()
    text_mode = text_mode.strip().lower()
    if text_mode not in SENTIMENT_INPUT_MODES:
        raise typer.BadParameter(f"text-mode must be one of: {', '.join(SENTIMENT_INPUT_MODES)}")
    effective_batch_size = int(batch_size or settings.finbert_batch_size)
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = FinbertScorer(
        settings.finbert_model,
        torch_num_threads=settings.torch_num_threads,
        max_length=max_length,
    )
    summary: list[dict[str, object]] = []
    for symbol in symbols:
        source = raw_dir / f"{symbol}_events.parquet"
        target = out_dir / f"{symbol}_events.parquet"
        if target.exists() and not force:
            frame = pd.read_parquet(target)
            provenance_columns = {
                "sentiment_label",
                "sentiment_score",
                "sentiment_numeric",
                "sentiment_input_mode",
                "sentiment_model",
                "sentiment_max_length",
            }
            compatible = (
                provenance_columns <= set(frame.columns)
                and frame["sentiment_input_mode"].eq(text_mode).all()
                and frame["sentiment_model"].eq(settings.finbert_model).all()
                and pd.to_numeric(frame["sentiment_max_length"], errors="coerce").eq(max_length).all()
            )
            if compatible:
                summary.append(
                    {
                        "ticker": symbol,
                        "events": len(frame),
                        "path": str(target),
                        "skipped": True,
                        "text_mode": text_mode,
                        "max_length": max_length,
                    }
                )
                console.print(f"{symbol}: compatible scored file exists, skipped")
                continue
            console.print(f"[yellow]{symbol}: existing scored file has incompatible provenance; rescoring.[/yellow]")
        if not source.exists():
            summary.append({"ticker": symbol, "events": 0, "error": f"missing {source}"})
            console.print(f"[red]{symbol}: missing {source}[/red]")
            continue
        try:
            frame = pd.read_parquet(source)
            frame, report = sanitize_events_frame(frame)
            for col in ["title", "summary", "text"]:
                if col not in frame.columns:
                    frame[col] = ""
            frame["text"] = frame["text"].fillna(frame["summary"]).fillna(frame["title"]).fillna("")
            frame = frame.drop(
                columns=["sentiment_label", "sentiment_score", "sentiment_numeric"],
                errors="ignore",
            )
            frame["_sentiment_input"] = build_sentiment_inputs(frame, mode=text_mode)
            frame = add_finbert_with_scorer(
                frame,
                scorer,
                batch_size=effective_batch_size,
                text_column="_sentiment_input",
            ).drop(columns="_sentiment_input")
            frame["sentiment_input_mode"] = text_mode
            frame["sentiment_model"] = settings.finbert_model
            frame["sentiment_max_length"] = max_length
            frame.to_parquet(target, index=False)
            summary.append(
                {
                    "ticker": symbol,
                    "events": len(frame),
                    "path": str(target),
                    "missing_required_rows_removed": report.missing_required_rows_removed,
                    "duplicate_rows_removed": report.duplicate_rows_removed,
                    "text_mode": text_mode,
                    "max_length": max_length,
                    "batch_size": effective_batch_size,
                }
            )
            console.print(f"{symbol}: scored {len(frame)} events to {target}")
        except Exception as exc:
            summary.append({"ticker": symbol, "events": 0, "error": str(exc)})
            console.print(f"[red]{symbol}: sentiment scoring failed: {exc}[/red]")
    pd.DataFrame(summary).to_csv(out_dir / "_sentiment_summary.csv", index=False)


@app.command("audit-swing-alignment")
def audit_swing_alignment(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing per-ticker event parquet files."),
    feature_dir: Path = typer.Option(Path("data/features/swing"), help="Directory containing per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Feature dataset horizon to audit."),
    out: Path = typer.Option(Path("data/reports/swing_alignment_audit.csv"), help="Output audit CSV."),
) -> None:
    """Audit news timing assignment and daily/hourly candle feature matching."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        events_path = raw_dir / f"{symbol}_events.parquet"
        features_path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not events_path.exists() or not features_path.exists():
            rows.append(
                {
                    "ticker": symbol,
                    "error": f"missing events={events_path.exists()} features={features_path.exists()}",
                }
            )
            continue
        try:
            events, verify = sanitize_events_frame(pd.read_parquet(events_path))
            dataset = pd.read_parquet(features_path)
            if events.empty or dataset.empty:
                rows.append({"ticker": symbol, "events": len(events), "feature_rows": len(dataset), "error": "empty"})
                continue
            events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
            dataset_dates = set(pd.to_datetime(dataset["date"]).dt.date)
            events = align_events_to_trading_dates(events, list(dataset_dates))
            events = add_event_taxonomy(events)
            latest_feature_date = max(dataset_dates)
            events["has_feature_row"] = events["date"].isin(dataset_dates)
            events["pending_after_latest_feature_date"] = events["date"] > latest_feature_date
            events["missing_historical_feature_row"] = (~events["has_feature_row"]) & (~events["pending_after_latest_feature_date"])
            source_counts = events["source"].map(source_family_for_source).value_counts().to_dict()
            grouped_events = events.groupby("date").size().rename("event_count_raw")
            grouped_dataset = dataset.copy()
            grouped_dataset["date"] = pd.to_datetime(grouped_dataset["date"]).dt.date
            grouped_dataset = grouped_dataset.set_index("date")
            joined = grouped_events.to_frame().join(grouped_dataset[["news_count"]], how="left")
            joined["news_count_diff"] = joined["event_count_raw"] - joined["news_count"].fillna(0)
            event_dates = events.groupby("event_time_bucket")["date"].nunique().to_dict()

            rows.append(
                {
                    "ticker": symbol,
                    "events": len(events),
                    "feature_rows": len(dataset),
                    "alpaca_events": int(source_counts.get("alpaca", 0)),
                    "seeking_alpha_events": int(source_counts.get("seeking_alpha", 0)),
                    "reddit_events": int(source_counts.get("reddit", 0)),
                    "finviz_events": int(source_counts.get("finviz", 0)),
                    "events_without_feature_row": int((~events["has_feature_row"]).sum()),
                    "pending_after_latest_feature_date": int(events["pending_after_latest_feature_date"].sum()),
                    "missing_historical_feature_rows": int(events["missing_historical_feature_row"].sum()),
                    "dates_with_news_count_mismatch": int((joined["news_count_diff"].abs() > 0).sum()),
                    "max_abs_news_count_diff": float(joined["news_count_diff"].abs().max() or 0),
                    "pre_market_event_dates": int(event_dates.get("pre_market", 0)),
                    "intraday_event_dates": int(event_dates.get("intraday", 0)),
                    "after_hours_event_dates": int(event_dates.get("after_hours", 0)),
                    "premarket_gap_matched_days": _count_nonzero_feature_days(events, grouped_dataset, "premarket_gap_mean", "pre_market"),
                    "intraday_2h_reaction_matched_days": _count_nonzero_feature_days(
                        events, grouped_dataset, "intraday_reaction_2h_mean", "intraday"
                    ),
                    "afterhours_gap_matched_days": _count_nonzero_feature_days(
                        events,
                        grouped_dataset,
                        "afterhours_next_open_gap_mean",
                        "after_hours",
                    ),
                    "sanitized_rows_out": verify.rows_out,
                }
            )
        except Exception as exc:
            rows.append({"ticker": symbol, "error": str(exc)})
            console.print(f"[red]{symbol}: alignment audit failed: {exc}[/red]")
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(frame.head(60))
    console.print(f"Wrote alignment audit to {out}")


@app.command("build-swing-datasets")
def build_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing *_events.parquet files."),
    out_dir: Path = typer.Option(Path("data/features/swing"), help="Directory for per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Forward trading-day target horizon."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD for price/features."),
    workers: int | None = typer.Option(None, help="Parallel dataset build workers. Defaults to config performance.max_workers."),
) -> None:
    """Build daily feature/label datasets for the swing universe."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    end = _parse_end_date(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Building {len(symbols)} datasets with {max_workers} worker(s).")

    def run_symbol(symbol: str) -> dict[str, object]:
        events_path = raw_dir / f"{symbol}_events.parquet"
        if not events_path.exists():
            return {"ticker": symbol, "rows": 0, "error": f"missing {events_path}"}
        try:
            dataset = build_daily_dataset(
                symbol,
                events_path,
                settings,
                horizon_days=horizon_days,
                market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                end=end,
            )
            out = out_dir / f"{symbol}_daily_{horizon_days}d.parquet"
            dataset.to_parquet(out, index=False)
            return {"ticker": symbol, "rows": len(dataset), "path": str(out)}
        except Exception as exc:
            return {"ticker": symbol, "rows": 0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            record = future.result()
            summary.append(record)
            if record.get("error"):
                console.print(f"[red]{symbol}: dataset build failed: {record['error']}[/red]")
            else:
                console.print(f"{symbol}: wrote {record['rows']} rows to {record['path']}")
    pd.DataFrame(summary).to_csv(out_dir / "_dataset_summary.csv", index=False)


@app.command("combine-swing-datasets")
def combine_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(Path("data/features/swing"), help="Directory containing per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Dataset horizon to combine."),
    out: Path = typer.Option(Path("data/features/swing_combined_1d.parquet"), help="Combined output parquet."),
) -> None:
    """Combine per-ticker swing datasets for cross-sectional model training."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    frames = []
    summary: list[dict[str, object]] = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not path.exists():
            summary.append({"ticker": symbol, "rows": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            frame["ticker"] = symbol
            frame = frame[_daily_training_columns(frame, horizon_days)]
            frames.append(frame)
            summary.append({"ticker": symbol, "rows": len(frame), "path": str(path)})
        except Exception as exc:
            summary.append({"ticker": symbol, "rows": 0, "error": str(exc)})
    if not frames:
        raise typer.BadParameter("No datasets found to combine.")
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
    console.print(f"Wrote {len(combined)} rows across {len(frames)} tickers to {out}")


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


@app.command("azure-publish-serving-release")
def azure_publish_serving_release(
    root: Path = typer.Option(Path("."), help="Deployment root containing configured models and live features."),
    release_prefix: str = typer.Option(DEFAULT_RELEASE_PREFIX, help="Immutable release prefix under AZURE_BLOB_PREFIX."),
    active_pointer: str = typer.Option(DEFAULT_ACTIVE_POINTER, help="Mutable active-release pointer blob."),
) -> None:
    """Publish one immutable, complete model-plus-feature serving release."""

    settings = get_settings()
    resolved_root = root.resolve()
    pointer = publish_serving_release(
        AzureBlobStore(settings),
        root=resolved_root,
        routes=serving_routes_from_config(settings.app_config),
        live_feature_store=LiveFeatureStore(resolved_root),
        release_prefix=release_prefix,
        active_pointer_blob=active_pointer,
    )
    console.print(pointer)


@app.command("azure-rollback-serving-release")
def azure_rollback_serving_release(
    release_id: str = typer.Option(..., help="Previously published 64-character serving release id."),
    release_prefix: str = typer.Option(DEFAULT_RELEASE_PREFIX, help="Immutable release prefix under AZURE_BLOB_PREFIX."),
    active_pointer: str = typer.Option(DEFAULT_ACTIVE_POINTER, help="Mutable active-release pointer blob."),
) -> None:
    """Atomically move the active pointer to a verified earlier release."""

    pointer = rollback_serving_release(
        AzureBlobStore(get_settings()),
        release_id=release_id,
        release_prefix=release_prefix,
        active_pointer_blob=active_pointer,
    )
    console.print(pointer)


@app.command("azure-sync-serving-release")
def azure_sync_serving_release(
    root: Path = typer.Option(Path("."), help="Local deployment root to hydrate before API startup."),
    active_pointer: str = typer.Option(DEFAULT_ACTIVE_POINTER, help="Active-release pointer blob."),
) -> None:
    """Hydrate and integrity-check the active Azure release into the local root."""

    pointer = sync_active_serving_release(
        AzureBlobStore(get_settings()),
        root=root,
        active_pointer_blob=active_pointer,
    )
    console.print(pointer)


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


@app.command("build-entry-exit-dataset")
def build_entry_exit_dataset_command(
    input_path: Path = typer.Option(
        Path("data/features/volatile_mover_daily_20260704.parquet"),
        "--input",
        help="Input feature parquet/CSV with ticker, date, OHLCV, and optional news/model features.",
    ),
    out: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Output entry/exit labeled dataset.",
    ),
    audit_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_5b_audit_20260704.csv"),
        help="Per-ticker entry/exit readiness audit CSV.",
    ),
    horizon_bars: int = typer.Option(5, help="Number of bars after next open to evaluate target/stop path."),
    take_profit_atr: float = typer.Option(1.5, help="ATR multiple for target from next open."),
    stop_loss_atr: float = typer.Option(1.0, help="ATR multiple for stop from next open."),
    min_rows_per_ticker: int = typer.Option(120, help="Minimum rows per ticker."),
    min_labeled_rows_per_ticker: int = typer.Option(40, help="Minimum non-null path labels per ticker."),
    bar_kind: str = typer.Option("swing", help="Human label for bar type: swing, daily, hourly, 5min, etc."),
    allow_overnight: bool = typer.Option(False, help="Allow intraday path labels to cross session boundaries."),
    session_scope: str = typer.Option(
        "all",
        help="Intraday setup window: all, premarket, opening (09:30-11:30 ET), or regular.",
    ),
    setup_cooldown_bars: int = typer.Option(
        0,
        min=0,
        help="Minimum bars between setup events per ticker/session; horizon+1 is enforced when enabled.",
    ),
    round_trip_cost_bps: float = typer.Option(
        0.0,
        min=0.0,
        help="Round-trip fees and slippage in basis points included in labels and realized returns.",
    ),
    min_setup_score: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional minimum point-in-time setup score, applied after path labels are built.",
    ),
    context_path: Path | None = typer.Option(
        None,
        "--context",
        help="Optional point-in-time enriched parquet; only approved missing model features are joined.",
    ),
) -> None:
    """Build leak-safe entry/exit path labels from OHLCV feature rows."""
    if not input_path.exists():
        raise typer.BadParameter(f"Missing input dataset: {input_path}")
    if input_path.is_dir():
        files = sorted(path for path in input_path.rglob("*.parquet") if not path.name.startswith("_"))
        if not files:
            raise typer.BadParameter(f"No parquet files found under input directory: {input_path}")
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    else:
        frame = pd.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path)
    config = EntryExitLabelConfig(
        horizon_bars=horizon_bars,
        take_profit_atr=take_profit_atr,
        stop_loss_atr=stop_loss_atr,
        min_rows_per_ticker=min_rows_per_ticker,
        min_labeled_rows_per_ticker=min_labeled_rows_per_ticker,
        bar_kind=bar_kind,
        allow_overnight=allow_overnight,
        session_scope=session_scope.strip().lower(),
        setup_cooldown_bars=setup_cooldown_bars,
        round_trip_cost_bps=round_trip_cost_bps,
        min_setup_score=min_setup_score,
    )
    dataset, audit = build_entry_exit_dataset(frame, config=config)
    if context_path is not None:
        if not context_path.exists():
            raise typer.BadParameter(f"Missing context dataset: {context_path}")
        context = pd.read_parquet(context_path) if context_path.suffix.lower() == ".parquet" else pd.read_csv(context_path)
        dataset = merge_entry_exit_context(dataset, context)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    audit.to_csv(audit_out, index=False)
    suffix = f"{horizon_bars}b"
    summary = {
        "schema": ENTRY_EXIT_SCHEMA_VERSION,
        "bar_kind": bar_kind,
        "allow_overnight": allow_overnight,
        "session_scope": session_scope,
        "setup_cooldown_bars": setup_cooldown_bars,
        "round_trip_cost_bps": round_trip_cost_bps,
        "min_setup_score": min_setup_score,
        "context": str(context_path) if context_path else None,
        "rows": len(dataset),
        "tickers": int(dataset["ticker"].nunique()) if not dataset.empty else 0,
        "first_date": str(dataset["date"].min()) if not dataset.empty else None,
        "last_date": str(dataset["date"].max()) if not dataset.empty else None,
        "target_columns": [
            f"target_entry_success_{suffix}",
            f"target_exit_risk_{suffix}",
            f"target_timeout_positive_{suffix}",
            f"target_net_positive_{suffix}",
        ],
        "out": str(out),
        "audit": str(audit_out),
    }
    console.print(summary)
    console.print(audit.sort_values(["model_eligible", "rows"], ascending=[True, False]).head(30))


@app.command("train-entry-exit-model")
def train_entry_exit_model_command(
    dataset: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Entry/exit dataset produced by build-entry-exit-dataset.",
    ),
    target_col: str = typer.Option("target_entry_success_5b", help="Target column to train."),
    model_out: Path = typer.Option(
        Path("models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib"),
        help="Output model artifact.",
    ),
    predictions_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_entry_success_5b_oos_predictions_20260704.csv"),
        help="Out-of-sample prediction audit CSV.",
    ),
    metrics_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_entry_success_5b_metrics_20260704.csv"),
        help="Model metrics/model-card CSV.",
    ),
    max_iter: int = typer.Option(350, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.04, help="Boosting learning rate."),
    embargo_rows: int | None = typer.Option(None, help="Purged walk-forward embargo rows. Defaults from target horizon."),
    feature_set: str = typer.Option("all", help="Feature ablation set: all, technical, or catalyst."),
    estimator: str = typer.Option(
        "hist_gradient_boosting",
        help="Estimator family: hist_gradient_boosting, extra_trees, or logistic.",
    ),
) -> None:
    """Train an entry/exit path classifier."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing entry/exit dataset: {dataset}")
    frame = pd.read_parquet(dataset)
    report, metrics, _ = train_entry_exit_model(
        frame,
        target_col=target_col,
        model_out=model_out,
        predictions_out=predictions_out,
        metrics_out=metrics_out,
        max_iter=max_iter,
        learning_rate=learning_rate,
        embargo_rows=embargo_rows,
        feature_set=feature_set,
        estimator=estimator,
    )
    console.print(report)
    console.print(metrics)


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


@app.command("score-entry-exit-latest")
def score_entry_exit_latest(
    dataset: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Entry/exit feature dataset.",
    ),
    model: Path = typer.Option(
        Path("models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib"),
        help="Entry/exit model artifact.",
    ),
    tickers: str | None = typer.Option(None, help="Optional comma-separated ticker subset."),
    out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_latest_scores_20260704.csv"),
        help="Latest per-ticker entry/exit model scores.",
    ),
) -> None:
    """Score latest rows with an entry/exit model."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing entry/exit dataset: {dataset}")
    if not model.exists():
        raise typer.BadParameter(f"Missing entry/exit model: {model}")
    frame = pd.read_parquet(dataset)
    symbols = _parse_tickers(tickers, []) if tickers else []
    if symbols:
        frame = frame[frame["ticker"].astype(str).str.upper().isin(symbols)].copy()
    latest = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    scored = score_entry_exit_frame(latest, model)
    probability_cols = [col for col in scored.columns if col.endswith("_probability")]
    sort_col = probability_cols[-1] if probability_cols else "ticker"
    scored = scored.sort_values(sort_col, ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    keep = [
        col
        for col in [
            "ticker",
            "date",
            "close",
            "return_1d",
            "volume_z20",
            "news_count",
            "event_count",
            "rsi_14",
            "macd_signal_diff",
            "entry_stop_pct",
            "entry_target_pct",
            sort_col,
            sort_col.replace("probability", "prediction"),
            "entry_exit_model_target",
            "entry_exit_model_schema",
        ]
        if col in scored.columns
    ]
    console.print(scored[keep].head(50))
    console.print(f"Wrote entry/exit latest scores to {out}")


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


@app.command("collect-seeking-alpha")
def collect_seeking_alpha(
    ticker: str,
    out: Path = typer.Option(
        Path("data/external/seeking_alpha_quant.csv"),
        help="Output CSV used by build-dataset --seeking-alpha.",
    ),
) -> None:
    """Collect a Seeking Alpha quant, earnings, and rating snapshot via RapidAPI."""
    write_seeking_alpha_snapshot(ticker, out)
    console.print(f"Wrote Seeking Alpha quant snapshot for {ticker.upper()} to {out}")


@app.command("collect-seeking-alpha-universe")
def collect_seeking_alpha_universe(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(730, help="Event lookback window in calendar days."),
    out_dir: Path = typer.Option(Path("data/external/seeking_alpha_full"), help="Per-ticker output directory."),
    include_events: bool = typer.Option(True, help="Collect configured Seeking Alpha event feeds."),
    include_snapshots: bool = typer.Option(True, help="Collect configured Seeking Alpha snapshot feeds."),
) -> None:
    """Collect all configured Seeking Alpha event/snapshot feeds with per-ticker isolation."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    if not symbols:
        raise typer.BadParameter("No tickers configured or supplied.")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(UTC) - timedelta(days=days)
    source = SeekingAlphaRapidApiSource(settings)
    summary_rows = []
    snapshot_rows = []
    for index, symbol in enumerate(symbols, start=1):
        row: dict[str, object] = {"ticker": symbol, "status": "ok", "events": 0, "snapshot_fields": 0}
        try:
            if include_events:
                events, event_errors = source.fetch_events_with_errors(symbol, start)
                event_frame = pd.DataFrame([event.to_record() for event in events])
                if "raw" in event_frame.columns:
                    event_frame["raw"] = event_frame["raw"].map(
                        lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, dict) else value
                    )
                event_path = out_dir / f"{symbol}_events.parquet"
                event_frame.to_parquet(event_path, index=False)
                row["events"] = len(event_frame)
                row["events_path"] = str(event_path)
                row["event_errors"] = " | ".join(event_errors)
            if include_snapshots:
                snapshot = source.fetch_quant_snapshot(symbol)
                snapshot_frame = pd.DataFrame([snapshot])
                snapshot_path = out_dir / f"{symbol}_snapshot.csv"
                snapshot_frame.to_csv(snapshot_path, index=False)
                snapshot_rows.append(snapshot)
                row["snapshot_fields"] = len(snapshot)
                row["snapshot_path"] = str(snapshot_path)
                row["snapshot_errors"] = " | ".join(f"{key}={value}" for key, value in snapshot.items() if str(key).endswith("_error"))
                row["snapshot_skips"] = " | ".join(f"{key}={value}" for key, value in snapshot.items() if str(key).endswith("_skipped"))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        summary_rows.append(row)
        console.print(f"{index}/{len(symbols)} {symbol}: {row['status']} events={row.get('events')} fields={row.get('snapshot_fields')}")
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    if snapshot_rows:
        combined_path = out_dir / "_snapshots_combined.csv"
        pd.DataFrame(snapshot_rows).to_csv(combined_path, index=False)
        console.print(f"Wrote combined snapshots to {combined_path}")
    console.print(f"Wrote Seeking Alpha collection summary to {summary_path}")


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
    rows = []
    errors = []
    if settings.has_seeking_alpha_rapidapi:
        try:
            rows.extend([event.to_record() for event in SeekingAlphaRapidApiSource(settings).fetch_market_context_events(start)])
        except Exception as exc:
            errors.append(f"seeking_alpha_market_context:{exc}")
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


@app.command("collect-gdelt-context")
def collect_gdelt_context(
    days: int = typer.Option(2, help="Global flashpoint lookback window in calendar days."),
    out: Path = typer.Option(
        Path("data/external/market_context/gdelt_context_events.parquet"),
        help="Output GDELT context events parquet.",
    ),
    max_records_per_query: int = typer.Option(75, help="Maximum GDELT articles per configured query."),
    query: str | None = typer.Option(None, help="Optional single GDELT DOC query for one flashpoint family."),
    request_pause_seconds: float = typer.Option(5.5, help="Pause/retry backoff for GDELT public rate limits."),
    request_retries: int = typer.Option(2, help="HTTP retries per GDELT query."),
    score_sentiment: bool = typer.Option(False, help="Run FinBERT on GDELT context rows."),
) -> None:
    """Collect real global flashpoint news from GDELT into the market-context schema."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    start = datetime.now(UTC) - timedelta(days=days)
    queries = (query,) if query else None
    source = GdeltSource(request_pause_seconds=request_pause_seconds, request_retries=request_retries)
    events, errors = source.fetch_context_events_with_errors(
        start,
        queries=queries or DEFAULT_GDELT_CONTEXT_QUERIES,
        max_records_per_query=max_records_per_query,
    )
    frame = pd.DataFrame([event.to_record() for event in events])
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
        "out": str(out),
        "days": days,
        "max_records_per_query": max_records_per_query,
        "query": query,
        "errors": errors,
    }
    (out.parent / "gdelt_context_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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


@app.command("seeking-alpha-limits")
def seeking_alpha_limits() -> None:
    """Show local RapidAPI usage and the last rate-limit headers seen."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    status = source.quota_status()
    console.print(
        {
            "month": status.month,
            "local_used": status.used,
            "configured_monthly_limit": status.limit,
            "local_remaining": status.remaining,
            "last_rapidapi_headers": status.last_headers,
            "cache_dir": str(settings.seeking_alpha_cache_dir),
        }
    )


@app.command("seeking-alpha-token")
def seeking_alpha_token(
    refresh: bool = typer.Option(False, help="Force a new token request instead of using the local cache."),
) -> None:
    """Fetch/cache a Seeking Alpha account access token without printing the token."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    token = source.get_account_access_token(force_refresh=refresh)
    console.print(
        {
            "token_cached": bool(token),
            "cache_file": str(settings.seeking_alpha_access_token_cache_file),
        }
    )


@app.command("seeking-alpha-token-status")
def seeking_alpha_token_status() -> None:
    """Show whether Seeking Alpha account credentials/token cache are configured."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    console.print(source.account_token_status())
