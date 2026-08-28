"""ER3 deterministic setup population for ``SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1``.

This module builds the population that
:mod:`market_predictor.edge_rebuild.setup_economics` admits or rejects. No
estimator is involved and none may be: V1 and V2 both fitted models on
populations that had no gross edge before any model existed, and the V2 intraday
setup averaged a negative gross return *before* costs. ER3 inverts that order.

Nothing here is reimplemented. The causal feature history, the exact next-open to
horizon-close label path, the point-in-time membership join, and the deterministic
ticker holdout all come from the existing canonical implementations
(:mod:`market_predictor.swing.dataset`, :mod:`market_predictor.swing.labels`,
:mod:`market_predictor.canonical.joins`, :mod:`market_predictor.modeling.validation`).
This module contributes exactly one new thing: the deterministic setup rule.

The rule, evaluated after the completed daily bar of session ``t`` at the frozen
18:00 New York cutoff, using only bars at or before ``t``:

``warm-up``
    ``daily_bar_count(t) >= minimum_warmup_sessions`` from the frozen contract.
    Zero-volume provider placeholders are dropped before any feature is computed,
    so they can neither warm a security up nor enter a moving average.

``residual strength``
    ``return_20d`` and ``return_60d`` must exceed *both* the SPY return and the
    point-in-time sector-ETF return over the identical window. Removing both
    benchmarks at unit beta is what "stock-specific strength remaining after
    market and sector effects are removed" means here; see ``Residual definition``
    below for why no beta is estimated.

``long-term trend``
    ``dist_sma_200(t) > 0`` and ``sma_200_slope_20d(t) > 0``. Price above a rising
    200-session average, with the full 250-session warm-up behind it.

``bounded pullback that does not break the trend``
    Measured on ``t-1``: ``dist_ema_10(t-1) < 0`` (price had pulled back below its
    own short-term trend) while ``dist_sma_200(t-1) > 0`` (the pullback never broke
    the long-term trend). The pullback is bounded by the long-term trend itself
    rather than by an invented depth threshold.

``price/volume reclaim confirmation``
    On ``t``: ``dist_ema_10(t) > 0``, ``intraday_return(t) > 0`` and
    ``volume_ratio_20(t) > 1``. Together with the pullback condition this is a
    strict upward crossing of the 10-session EMA on an up bar with above-average
    volume. **This is what makes the setup one-shot.** A standing condition such as
    "price is above its EMA" is true for weeks and would sample the same state
    repeatedly; a crossing fires at most once per pullback leg.

``liquidity and capacity``
    Swing eligibility is index membership (plan section 8A), so the screens here
    are the ones membership does not supply: the decision bar must have traded
    (zero-volume bars are already gone) on above-average volume, the exact
    executable path must exist in the immutable bars, and the contract's
    ``maximum_trades_per_decision`` cap keeps the most liquid qualifying names per
    decision date by decision-bar dollar volume.

``No threshold outside the frozen contract.``
    Every numeric constant used for selection comes from
    ``configs/edge_rebuild_strategy_contract.toml``: the warm-up length, the
    horizon, the round-trip cost, the trade cap and the holdout fraction. Every
    other condition is a sign test or an ordering against the security's own
    trailing statistic. There is nothing to tune, which is the point: a setup with
    free parameters searched against the outcome is how a random result becomes a
    "discovery".

``Residual definition.``
    Residual strength is the unit-beta excess over SPY *and* over the
    point-in-time sector ETF, not the residual of a fitted market/sector
    regression. A regression residual would require a beta estimation window,
    and no such parameter exists in the frozen contract. Introducing one would be
    exactly the free parameter the contract exists to forbid. This is a faithful
    approximation, not the literal decomposition, and it is stated so a reviewer
    can weigh it.

``Point-in-time discipline.``
    Features are produced by the shared builder, whose rolling and exponential
    windows are backward-looking with full ``min_periods``, and the pullback terms
    are explicit within-security lags. Labels come only from
    :func:`market_predictor.swing.labels.add_exact_swing_labels`, which reads the
    entry open at ``t+1`` and the exit close at ``t+horizon``. No column is ever
    computed across the whole series and sliced afterwards.

``Phases.``
    Consecutive daily decisions share nine of their ten outcome sessions, so row
    counts are not sample size. ``phase = session_ordinal % horizon_sessions``
    over the ordered SPY session calendar. Within one phase,
    consecutive decision sessions are a full horizon apart and their labels share
    no bar.

``Scopes.``
    ``unseen_ticker`` is the deterministic-hash 20% ticker holdout required by the
    contract, applied through the shared
    :func:`market_predictor.modeling.validation.deterministic_ticker_holdout`. A security
    that was renamed carries several tickers; it is held out when *any* of them is,
    so no held-out ticker string can appear in ``walk_forward`` and no security is
    ever split across scopes. The trade cap is applied per scope, because each
    scope is an independent replay of the same strategy over its own universe
    rather than a thinned residue of a single global replay.

``session_segment``
    A daily post-close decision has exactly one segment, so the swing baseline
    config declares only the three mandatory concentration dimensions. The column
    is still emitted, and its single value is honest evidence rather than a
    fabricated split.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.canonical.joins import (
    decisions_from_completed_bars,
    join_universe_membership,
)
from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.setup_economics import (
    MANDATORY_CONCENTRATION_DIMENSIONS,
    UNSEEN_TICKER_SCOPE,
    WALK_FORWARD_SCOPE,
    SetupEconomicsConfig,
)
from market_predictor.edge_rebuild.swing_features import swing_dataset_config
from market_predictor.edge_rebuild.swing_pipeline_steps import SetupComponentsStep
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.modeling.validation import deterministic_ticker_holdout
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.dataset import build_swing_feature_history
from market_predictor.swing.labels import add_exact_swing_labels

SWING_SETUP_SCHEMA = "edge_rebuild.swing_setups.v1"
SWING_SETUP_FEATURE_PROFILE = "technical_market"
SWING_SESSION_SEGMENT = "post_close"
SWING_HOLDOUT_SEED = 42

#: A daily decision has one session segment, so only the mandatory dimensions are
#: declared. Every baseline threshold stays at its frozen default.
SWING_SETUP_ECONOMICS_CONFIG = SetupEconomicsConfig(
    concentration_dimensions=MANDATORY_CONCENTRATION_DIMENSIONS,
)

SWING_SETUP_COLUMNS = (
    "strategy_id",
    "security_id",
    "ticker",
    "session",
    "decision_time_utc",
    "entry_session_date_et",
    "exit_session_date_et",
    "entry_time_utc",
    "exit_time_utc",
    "gross_return",
    "cost",
    "net_return",
    "spy_excess_return",
    "sector_excess_return",
    "scope",
    "phase",
    "sector",
    "market_regime",
    "session_segment",
    "residual_return_20d_vs_spy",
    "residual_return_20d_vs_sector",
    "residual_return_60d_vs_spy",
    "residual_return_60d_vs_sector",
    "dist_sma_200",
    "sma_200_slope_20d",
    "prior_dist_ema_10",
    "prior_dist_sma_200",
    "dist_ema_10",
    "intraday_return",
    "volume_ratio_20",
    "dollar_volume",
    "daily_bar_count",
)

_RESIDUAL_COLUMNS = (
    "residual_return_20d_vs_spy",
    "residual_return_20d_vs_sector",
    "residual_return_60d_vs_spy",
    "residual_return_60d_vs_sector",
)
_COMPONENT_COLUMNS = (
    *_RESIDUAL_COLUMNS,
    "dist_sma_200",
    "sma_200_slope_20d",
    "prior_dist_ema_10",
    "prior_dist_sma_200",
    "dist_ema_10",
    "intraday_return",
    "volume_ratio_20",
    "dollar_volume",
)
_MEMORY_BUDGET_GIB = 4.0
_MEMORY_HEADROOM_GIB = 0.75


def drop_placeholder_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove zero-volume daily bars, which are placeholders rather than observations.

    A provider emits flat zero-volume daily bars for a dead symbol. They already
    reached the model input layer once. Dropped here, before any feature exists,
    they can neither satisfy the warm-up count nor enter a moving average, a
    volume baseline, or an executable path.
    """

    traded = bars.loc[pd.to_numeric(bars["volume"], errors="coerce").gt(0)]
    return traded, int(len(bars) - len(traded))


# --------------------------------------------------------------------------- #
# The deterministic setup rule
# --------------------------------------------------------------------------- #
def swing_setup_mask(components: pd.DataFrame, *, contract: StrategyContract) -> pd.Series:
    """The deterministic one-shot qualification test. Missing evidence never qualifies."""

    swing = contract.swing
    finite = components.loc[:, list(_COMPONENT_COLUMNS)].notna().all(axis=1)
    warm = components["daily_bar_count"].ge(swing.minimum_warmup_sessions)
    residual = (
        components["residual_return_20d_vs_spy"].gt(0.0)
        & components["residual_return_20d_vs_sector"].gt(0.0)
        & components["residual_return_60d_vs_spy"].gt(0.0)
        & components["residual_return_60d_vs_sector"].gt(0.0)
    )
    trend = components["dist_sma_200"].gt(0.0) & components["sma_200_slope_20d"].gt(0.0)
    pullback = components["prior_dist_ema_10"].lt(0.0) & components["prior_dist_sma_200"].gt(0.0)
    reclaim = components["dist_ema_10"].gt(0.0) & components["intraday_return"].gt(0.0) & components["volume_ratio_20"].gt(1.0)
    horizon = swing.horizon_sessions
    labels = [
        f"future_gross_return_{horizon}d",
        f"future_net_return_{horizon}d",
        f"future_excess_return_{horizon}d_vs_spy",
        f"future_excess_return_{horizon}d_vs_sector",
    ]
    executable = (
        components["label_window_expected"].fillna(False).astype(bool)
        & components["label_path_exact"].fillna(False).astype(bool)
        & components.loc[:, labels].notna().all(axis=1)
    )
    mask = finite & warm & residual & trend & pullback & reclaim & executable
    return pd.Series(mask.to_numpy(dtype=bool), index=components.index, name="swing_setup")


# --------------------------------------------------------------------------- #
# One batch of securities: bars in, qualifying uncapped candidates out
# --------------------------------------------------------------------------- #
def build_swing_setup_candidates(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
) -> pd.DataFrame:
    """Replay one batch of securities into qualifying, uncapped setup candidates.

    ``stock_bars`` must carry every bar of every security in the batch, because the
    warm-up count and every trailing window are computed per ``security_id``.
    """

    config = swing_dataset_config(contract)
    horizon = contract.swing.horizon_sessions
    traded, _ = drop_placeholder_bars(stock_bars)
    decisions = decisions_from_completed_bars(traded, mode="swing-nightly")
    decisions = join_universe_membership(decisions, memberships)
    decisions["feature_profile"] = SWING_SETUP_FEATURE_PROFILE
    features, benchmark_features = build_swing_feature_history(
        decisions,
        benchmark_bars,
        config=config,
    )
    del decisions
    labelled = add_exact_swing_labels(features, benchmark_features, config, inplace=True)
    del features
    step = SetupComponentsStep(benchmark_features)
    components = step.transform(labelled)
    del labelled
    selected = components.loc[swing_setup_mask(components, contract=contract)]
    return _candidate_frame(selected, contract=contract, horizon=horizon)


def _candidate_frame(
    selected: pd.DataFrame,
    *,
    contract: StrategyContract,
    horizon: int,
) -> pd.DataFrame:
    """Project candidate rows onto the economics contract and verify cost identity."""

    cost = contract.swing.round_trip_cost_bps / 10_000.0
    frame = pd.DataFrame(
        {
            "strategy_id": contract.swing.strategy_id,
            "security_id": selected["security_id"].astype(str),
            "ticker": selected["ticker"].astype(str),
            "session": pd.to_datetime(selected["session_date_et"]),
            "decision_time_utc": selected["decision_time_utc"],
            "entry_session_date_et": pd.to_datetime(selected["entry_session_date_et"]),
            "exit_session_date_et": pd.to_datetime(selected["exit_session_date_et"]),
            "entry_time_utc": selected["entry_time_utc"],
            "exit_time_utc": selected["exit_time_utc"],
            "gross_return": selected[f"future_gross_return_{horizon}d"].astype(float),
            "cost": float(cost),
            "net_return": selected[f"future_net_return_{horizon}d"].astype(float),
            "spy_excess_return": selected[f"future_excess_return_{horizon}d_vs_spy"].astype(float),
            "sector_excess_return": selected[f"future_excess_return_{horizon}d_vs_sector"].astype(float),
            "sector": selected["sector"].astype(str),
            "market_regime": selected["market_regime"].astype(str),
            "session_segment": SWING_SESSION_SEGMENT,
            **{column: selected[column].astype(float) for column in _COMPONENT_COLUMNS},
            "daily_bar_count": selected["daily_bar_count"].astype(int),
        }
    ).reset_index(drop=True)
    _verify_label_economics(frame)
    return frame


def _verify_label_economics(frame: pd.DataFrame) -> None:
    """Costs are applied exactly once, and benchmark excess is never re-deducted.

    ``add_exact_swing_labels`` stamps ``net = gross - cost`` and
    ``excess = net - benchmark_return`` over the identical executable interval, so
    the excess columns already carry that single deduction. The identity is
    verified here and never recomputed, which is the only way a cost cannot be
    applied twice.
    """

    if frame.empty:
        return
    numeric = frame.loc[
        :,
        ["gross_return", "cost", "net_return", "spy_excess_return", "sector_excess_return"],
    ]
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise DataReadinessError("swing setup labels contain non-finite economics")
    residual = (frame["net_return"] - (frame["gross_return"] - frame["cost"])).abs()
    if float(residual.max()) > 1e-9:
        raise DataReadinessError("swing setup labels violate the single-cost identity net = gross - cost")


# --------------------------------------------------------------------------- #
# Whole-population finish: scope, phase, and the contract trade cap
# --------------------------------------------------------------------------- #
def finalize_swing_setup_population(
    candidates: pd.DataFrame,
    *,
    contract: StrategyContract,
    sessions: Sequence[pd.Timestamp],
    universe_tickers: pd.Series,
) -> pd.DataFrame:
    """Assign the holdout scope and the non-overlapping phase, then apply the cap."""

    if candidates.empty:
        raise DataReadinessError("swing setup population is empty before scope assignment")
    swing = contract.swing
    horizon = swing.horizon_sessions
    ordinals = {pd.Timestamp(session): index for index, session in enumerate(sessions)}
    frame = candidates.copy()
    ordinal = frame["session"].map(ordinals)
    if bool(ordinal.isna().any()):
        raise DataReadinessError("swing setup decisions contain sessions absent from SPY")
    frame["phase"] = ordinal.astype(int).mod(horizon)

    holdout = deterministic_ticker_holdout(
        universe_tickers,
        fraction=contract.validation.unseen_ticker_holdout_fraction,
        seed=SWING_HOLDOUT_SEED,
    )
    # A renamed security carries several tickers. Holding out the security whenever
    # any of its tickers is held out keeps every held-out ticker string out of
    # walk_forward and never splits one security across two scopes.
    held_out_security = frame.assign(_held=frame["ticker"].isin(holdout)).groupby("security_id", sort=False)["_held"].transform("any")
    frame["scope"] = np.where(held_out_security, UNSEEN_TICKER_SCOPE, WALK_FORWARD_SCOPE)

    # The contract's trade cap, applied per scope because each scope is an
    # independent replay of the same strategy over its own universe. Ranked by
    # decision-bar dollar volume so the cap doubles as the capacity screen; the
    # security identity breaks ties so the selection is reproducible.
    ordered = frame.sort_values(
        ["scope", "session", "dollar_volume", "security_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    capped = ordered.groupby(["scope", "session"], sort=False).head(swing.maximum_trades_per_decision)
    population = capped.sort_values(["decision_time_utc", "security_id"], kind="stable").reset_index(drop=True)
    duplicates = population.duplicated(subset=["security_id", "decision_time_utc"])
    if bool(duplicates.any()):
        raise DataReadinessError(f"swing setup population repeats a security/decision {int(duplicates.sum())} time(s)")
    return population.loc[:, list(SWING_SETUP_COLUMNS)]


# --------------------------------------------------------------------------- #
# Corpus collection: stream securities, never hold the whole feature panel
# --------------------------------------------------------------------------- #
def collect_swing_setup_population(
    *,
    collection_dir: Path,
    memberships_path: Path,
    contract: StrategyContract,
    securities_per_batch: int = 48,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the whole ER3 swing setup population from the immutable daily corpus.

    Securities are replayed in bounded batches. Only qualifying candidate rows
    survive a batch, so the feature panel for the full universe is never resident.
    """

    if securities_per_batch < 1:
        raise ValueError("securities per batch must be positive")
    memberships, _ = load_canonical_artifact(
        memberships_path,
        expected_type="memberships",
        allow_research=True,
    )
    artifacts = load_collection_artifact_index(collection_dir)
    benchmark_tickers = tuple(
        sorted(
            {
                contract.labels.benchmark_market.upper(),
                "QQQ",
                *memberships["primary_benchmark"].astype(str).str.upper(),
            }
        )
    )
    benchmark_bars = pd.concat(
        [load_daily_bars(ticker, artifacts) for ticker in benchmark_tickers],
        ignore_index=True,
    )
    zero_volume_benchmark = int(benchmark_bars["volume"].le(0).sum())
    if zero_volume_benchmark:
        raise DataReadinessError(f"benchmark bars contain {zero_volume_benchmark} zero-volume placeholder(s)")
    sessions = _session_calendar(benchmark_bars, contract.labels.benchmark_market.upper())

    batches: list[pd.DataFrame] = []
    zero_volume_dropped = 0
    securities_replayed = 0
    for group in iter_security_batches(memberships, securities_per_batch):
        stock_bars, dropped = load_security_batch_bars(group, artifacts)
        zero_volume_dropped += dropped
        securities_replayed += int(group["security_id"].nunique())
        if stock_bars.empty:
            continue
        batches.append(
            build_swing_setup_candidates(
                stock_bars,
                benchmark_bars,
                group,
                contract=contract,
            )
        )
        del stock_bars
        assert_memory_budget(
            hard_budget_gib=_MEMORY_BUDGET_GIB,
            headroom_gib=_MEMORY_HEADROOM_GIB,
            stage="swing setup candidate batch",
        )
    if not batches:
        raise DataReadinessError("no swing setup candidates were produced")
    candidates = pd.concat(batches, ignore_index=True)
    del batches
    population = finalize_swing_setup_population(
        candidates,
        contract=contract,
        sessions=sessions,
        universe_tickers=memberships["ticker"],
    )
    audit: dict[str, Any] = {
        "schema_version": SWING_SETUP_SCHEMA,
        "strategy_id": contract.swing.strategy_id,
        "strategy_contract_sha256": contract.sha256(),
        "memberships_path": str(memberships_path.resolve()),
        "memberships_sha256": file_sha256(memberships_path),
        "collection_dir": str(collection_dir.resolve()),
        "benchmark_tickers": list(benchmark_tickers),
        "securities_replayed": securities_replayed,
        "zero_volume_bars_dropped": zero_volume_dropped,
        "candidate_rows": int(len(candidates)),
        "population_rows": int(len(population)),
        "sessions_in_calendar": len(sessions),
    }
    return population, audit


def load_collection_artifact_index(
    collection_dir: Path,
) -> dict[str, tuple[Path, str]]:
    manifest_path = collection_dir / "_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"collection manifest must be an object: {manifest_path}")
    records = payload.get("artifacts")
    if not isinstance(records, list):
        raise DataReadinessError(f"collection manifest has no artifacts: {manifest_path}")
    root = collection_dir.resolve().parents[2]
    artifacts: dict[str, tuple[Path, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DataReadinessError("collection manifest artifact record is invalid")
        ticker = str(record["ticker"]).strip().upper()
        artifacts[ticker] = (root / str(record["path"]).replace("\\", "/"), str(record["sha256"]))
    return artifacts


def load_daily_bars(
    ticker: str,
    artifacts: dict[str, tuple[Path, str]],
) -> pd.DataFrame:
    normalized = ticker.strip().upper()
    if normalized not in artifacts:
        raise DataReadinessError(f"daily-history artifact is absent for {normalized}")
    path, expected = artifacts[normalized]
    if not path.exists() or file_sha256(path) != expected:
        raise DataReadinessError(f"daily-history artifact hash does not match for {normalized}")
    bars, _ = load_canonical_artifact(path, expected_type="bars")
    if set(bars["ticker"].astype(str).str.upper()) != {normalized}:
        raise DataReadinessError(f"daily-history ticker identity does not match for {normalized}")
    return bars


def iter_security_batches(
    memberships: pd.DataFrame,
    size: int,
) -> Iterator[pd.DataFrame]:
    identities = sorted(memberships["security_id"].astype(str).unique())
    for start in range(0, len(identities), size):
        selected = set(identities[start : start + size])
        yield memberships.loc[memberships["security_id"].astype(str).isin(selected)]


def load_security_batch_bars(
    group: pd.DataFrame,
    artifacts: dict[str, tuple[Path, str]],
) -> tuple[pd.DataFrame, int]:
    """Load every membership interval in the batch, dropping zero-volume placeholders.

    A zero-volume daily bar is a provider placeholder for a dead symbol, not an
    observation. It is dropped before any feature is computed, so it can neither
    satisfy the warm-up count nor enter a moving average or a volume baseline.
    """

    parts: list[pd.DataFrame] = []
    dropped = 0
    cache: dict[str, pd.DataFrame] = {}
    for ticker, intervals in group.groupby("ticker", sort=True):
        name = str(ticker).strip().upper()
        if name not in cache:
            traded, placeholders = drop_placeholder_bars(load_daily_bars(name, artifacts))
            dropped += placeholders
            cache[name] = traded
        bars = cache[name]
        starts = bars["bar_start_utc"]
        selected = pd.Series(False, index=bars.index)
        for interval in intervals.itertuples(index=False):
            window = starts.ge(interval.effective_from_utc)
            if pd.notna(interval.effective_to_utc):
                window &= starts.lt(interval.effective_to_utc)
            selected |= window
        part = bars.loc[selected]
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(), dropped
    return pd.concat(parts, ignore_index=True), dropped


def _session_calendar(benchmark_bars: pd.DataFrame, benchmark: str) -> list[pd.Timestamp]:
    rows = benchmark_bars.loc[benchmark_bars["ticker"].astype(str).str.upper().eq(benchmark)]
    if rows.empty:
        raise DataReadinessError(f"benchmark bars do not contain {benchmark}")
    sessions = (
        pd.to_datetime(rows["bar_start_utc"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
        .sort_values()
        .unique()
    )
    return [pd.Timestamp(session) for session in sessions]
