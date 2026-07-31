"""Diagnostic failure attribution for ``SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1``.

This is a one-off research study of a retired strategy, not a production path. It
answers one question: was the ER3 rejection a property of the strategy, or of the way
the strategy was evaluated?

Discipline
----------
Every cohort, ranking, split, and threshold is fixed in ``declaration.json`` and hashed
*before* any outcome is computed; the hash is recorded in the published request. Cohorts
are strictly one-dimensional, because crossing six dimensions produces thousands of
cells of which one passes by luck.

Every admission gate reported here comes from
:func:`market_predictor.edge_rebuild.setup_economics.evaluate_setup_economics` with the
frozen :data:`~market_predictor.edge_rebuild.swing_setups.SWING_SETUP_ECONOMICS_CONFIG`.
No threshold, aggregation, bootstrap, or drawdown rule is reimplemented here. Nothing is
tuned to make anything pass: a reproducible rejection is a valid outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.setup_economics import (
    SetupEconomicsReport,
    evaluate_setup_economics,
)
from market_predictor.edge_rebuild.swing_setups import SWING_SETUP_ECONOMICS_CONFIG
from market_predictor.regime_evidence import (
    REGIME_BOOTSTRAP_ITERATIONS,
    REGIME_BOOTSTRAP_SEED,
    maximum_drawdown_from_returns,
    session_block_mean_interval,
)
from market_predictor.v3.errors import DataReadinessError

SCHEMA: Final = "edge_rebuild.swing_failure_attribution.run.v1"
STRATEGY_ID: Final = "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1"
HORIZON_SESSIONS: Final = 10
SESSIONS_PER_YEAR: Final = 252.0
RESIDUAL_COLUMNS: Final = (
    "residual_return_20d_vs_spy",
    "residual_return_20d_vs_sector",
    "residual_return_60d_vs_spy",
    "residual_return_60d_vs_sector",
)
PSEUDO_RANDOM_SEED: Final = "swing-failure-attribution-20260731"
TERCILE_BOUNDS: Final = (1.0 / 3.0, 2.0 / 3.0)
MINIMUM_SECURITY_TRADES: Final = 3
GENERALISATION_SPLIT: Final = pd.Timestamp("2023-07-09")


# --------------------------------------------------------------------------- #
# Declared features. None of these reads an outcome column.
# --------------------------------------------------------------------------- #
def residual_strength(frame: pd.DataFrame) -> pd.Series:
    """The binding margin of the residual-momentum condition.

    All four residuals are already positive for every qualifying row, so the weakest
    one is how far the setup clears its own thesis. No parameter is fitted.
    """

    return frame.loc[:, list(RESIDUAL_COLUMNS)].min(axis=1)


def pseudo_random_score(frame: pd.DataFrame) -> pd.Series:
    """A fixed deterministic permutation used as a selection control."""

    keys = (
        PSEUDO_RANDOM_SEED
        + "|"
        + frame["security_id"].astype(str)
        + "|"
        + pd.to_datetime(frame["session"]).dt.strftime("%Y-%m-%d")
    )
    digests = [
        int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16) / 2.0**64
        for key in keys
    ]
    return pd.Series(digests, index=frame.index, dtype=float)


def tercile_label(values: pd.Series, labels: Sequence[str]) -> pd.Series:
    """Label a feature by its own 33.3/66.7 percentiles. Outcomes are never consulted."""

    numeric = pd.to_numeric(values, errors="coerce")
    low, high = (float(numeric.quantile(bound)) for bound in TERCILE_BOUNDS)
    result = pd.Series(labels[1], index=values.index, dtype=object)
    result = result.mask(numeric.le(low), labels[0])
    result = result.mask(numeric.gt(high), labels[2])
    return result.mask(numeric.isna(), "unknown")


def with_declared_features(population: pd.DataFrame) -> pd.DataFrame:
    """Attach every declared cohort key and ranking score to the population."""

    frame = population.copy()
    frame["session"] = pd.to_datetime(frame["session"])
    frame["residual_strength"] = residual_strength(frame)
    frame["pullback_depth"] = -frame["prior_dist_ema_10"]
    frame["pseudo_random"] = pseudo_random_score(frame)
    frame["calendar_year"] = frame["session"].dt.year.astype(str)
    frame["catalyst_3d"] = np.where(
        frame["catalyst_event_count_3d"].gt(0), "present", "absent"
    )
    frame["dollar_volume_tercile"] = tercile_label(
        frame["dollar_volume"], ("t1_low", "t2_mid", "t3_high")
    )
    frame["trailing_volatility_tercile"] = tercile_label(
        frame["trailing_volatility_20d"], ("t1_low", "t2_mid", "t3_high")
    )
    frame["residual_strength_tercile"] = tercile_label(
        frame["residual_strength"], ("t1_low", "t2_mid", "t3_high")
    )
    frame["pullback_depth_tercile"] = tercile_label(
        frame["pullback_depth"], ("t1_shallow", "t2_medium", "t3_deep")
    )
    # SPY's own return over the identical executable interval of each held position.
    frame["spy_interval_return"] = frame["net_return"] - frame["spy_excess_return"]
    return frame


# --------------------------------------------------------------------------- #
# Gate evaluation. One code path, always both scopes.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Evaluation:
    """One evaluated population and the reason it is or is not admissible."""

    label: str
    rows: int
    report: SetupEconomicsReport | None
    error: str

    @property
    def admitted(self) -> bool:
        return self.report is not None and self.report.admitted


def evaluate(frame: pd.DataFrame, *, label: str) -> Evaluation:
    """Run the full ER3 battery, failing closed on anything unevaluable."""

    if frame.empty:
        return Evaluation(label=label, rows=0, report=None, error="empty population")
    try:
        report = evaluate_setup_economics(
            frame,
            strategy_id=STRATEGY_ID,
            config=SWING_SETUP_ECONOMICS_CONFIG,
        )
    except DataReadinessError as exc:
        return Evaluation(label=label, rows=int(len(frame)), report=None, error=str(exc))
    return Evaluation(label=label, rows=int(len(frame)), report=report, error="")


def gate_records(evaluation: Evaluation, **context: object) -> Iterator[dict[str, object]]:
    """Flatten one evaluation into one row per scope per gate."""

    if evaluation.report is None:
        for scope in ("walk_forward", "unseen_ticker"):
            yield {
                **context,
                "label": evaluation.label,
                "scope": scope,
                "scope_rows": 0,
                "scope_admitted": False,
                "gate": "population_evaluable",
                "passed": False,
                "measured": float("nan"),
                "threshold": float("nan"),
                "margin": float("nan"),
                "direction": "minimum",
                "detail": evaluation.error,
            }
        return
    for scope_report in evaluation.report.scopes:
        scope_rows = scope_report.baseline.rows
        for gate in scope_report.gates:
            yield {
                **context,
                "label": evaluation.label,
                "scope": scope_report.scope,
                "scope_rows": scope_rows,
                "scope_admitted": scope_report.admitted,
                **gate.as_dict(),
            }


def descriptive(frame: pd.DataFrame) -> dict[str, float]:
    """Plain measured economics for a subset, reported alongside the gates."""

    if frame.empty:
        return {
            "rows": 0.0,
            "securities": 0.0,
            "sessions": 0.0,
            "gross_return": float("nan"),
            "net_return": float("nan"),
            "spy_excess_return": float("nan"),
            "sector_excess_return": float("nan"),
        }
    return {
        "rows": float(len(frame)),
        "securities": float(frame["security_id"].nunique()),
        "sessions": float(frame["session"].nunique()),
        "gross_return": float(frame["gross_return"].mean()),
        "net_return": float(frame["net_return"].mean()),
        "spy_excess_return": float(frame["spy_excess_return"].mean()),
        "sector_excess_return": float(frame["sector_excess_return"].mean()),
    }


# --------------------------------------------------------------------------- #
# Experiment A: deterministic top-k selection
# --------------------------------------------------------------------------- #
def top_k(frame: pd.DataFrame, *, ranking: str, k: int) -> pd.DataFrame:
    """Keep the k best rows of each (scope, session) group under a declared ranking."""

    ordered = frame.sort_values(
        ["scope", "session", ranking, "security_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return ordered.groupby(["scope", "session"], sort=False).head(k)


def sleeve_periods(frame: pd.DataFrame, *, calendar: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """Reduce a selection to one non-overlapping holding period per (phase, session).

    ``phase = session_ordinal % 10``, so consecutive sessions inside one phase are a
    full horizon apart and their holding windows share no bar. Each phase is therefore
    an independently chainable capital sleeve.
    """

    grouped = frame.groupby(["scope", "phase", "session"], sort=True)
    periods = grouped.agg(
        names=("security_id", "size"),
        net_return=("net_return", "mean"),
        spy_interval_return=("spy_interval_return", "mean"),
    ).reset_index()
    periods["excess"] = periods["net_return"] - periods["spy_interval_return"]
    ordinal = {session: index for index, session in enumerate(calendar)}
    periods["ordinal"] = periods["session"].map(ordinal)
    return periods.sort_values(["scope", "phase", "ordinal"], kind="stable")


def portfolio_economics(
    periods: pd.DataFrame,
    *,
    scope: str,
    calendar: Sequence[pd.Timestamp],
    spy_forward: Mapping[pd.Timestamp, float],
) -> dict[str, float]:
    """Compound the ten capital sleeves and compare against holding SPY.

    Each sleeve is a phase. It enters the selected names in equal weight at the next
    open and exits at the tenth session close; when nothing qualifies on its session it
    holds cash and earns zero. Capital is split evenly across the ten sleeves at the
    start and never rebalanced between them, so the portfolio value is the mean of the
    sleeve equities. The stamped cost is already inside ``net_return`` and is never
    re-applied.
    """

    scoped = periods.loc[periods["scope"].eq(scope)]
    if scoped.empty:
        return {"evaluable": 0.0}
    first = int(scoped["ordinal"].min())
    last = int(scoped["ordinal"].max())
    sleeve_equity: list[float] = []
    sleeve_spy_equity: list[float] = []
    sleeve_spy_continuous: list[float] = []
    invested = 0
    slots = 0
    curve: dict[int, float] = {}
    for phase in range(HORIZON_SESSIONS):
        sleeve = scoped.loc[scoped["phase"].eq(phase)].set_index("ordinal")
        equity = 1.0
        spy_equity = 1.0
        spy_continuous = 1.0
        for ordinal in range(first, last + 1):
            if ordinal % HORIZON_SESSIONS != phase % HORIZON_SESSIONS:
                continue
            slots += 1
            session = calendar[ordinal]
            benchmark = spy_forward.get(session)
            if benchmark is not None:
                spy_continuous *= 1.0 + benchmark
            if ordinal in sleeve.index:
                row = sleeve.loc[ordinal]
                invested += 1
                equity *= 1.0 + float(row["net_return"])
                spy_equity *= 1.0 + float(row["spy_interval_return"])
            curve[ordinal] = curve.get(ordinal, 0.0) + equity
        sleeve_equity.append(equity)
        sleeve_spy_equity.append(spy_equity)
        sleeve_spy_continuous.append(spy_continuous)

    sessions = float(last - first + 1)
    years = sessions / SESSIONS_PER_YEAR
    portfolio = float(np.mean(sleeve_equity))
    spy_matched = float(np.mean(sleeve_spy_equity))
    spy_held = float(np.mean(sleeve_spy_continuous))
    marks = pd.Series({key: value / HORIZON_SESSIONS for key, value in sorted(curve.items())})
    interval = session_block_mean_interval(
        scoped["session"],
        scoped["excess"],
        iterations=REGIME_BOOTSTRAP_ITERATIONS,
        seed=REGIME_BOOTSTRAP_SEED,
    )
    return {
        "evaluable": 1.0,
        "sessions": sessions,
        "years": years,
        "holding_periods": float(len(scoped)),
        "sleeve_slots": float(slots),
        "invested_slots": float(invested),
        "participation": float(invested) / float(slots) if slots else float("nan"),
        "total_return": portfolio - 1.0,
        "annualised_return": portfolio ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "spy_matched_total_return": spy_matched - 1.0,
        "spy_matched_annualised": (
            spy_matched ** (1.0 / years) - 1.0 if years > 0 else float("nan")
        ),
        "spy_held_total_return": spy_held - 1.0,
        "spy_held_annualised": spy_held ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "annualised_excess_vs_spy_matched": (
            portfolio ** (1.0 / years) - spy_matched ** (1.0 / years) if years > 0 else float("nan")
        ),
        "annualised_excess_vs_spy_held": (
            portfolio ** (1.0 / years) - spy_held ** (1.0 / years) if years > 0 else float("nan")
        ),
        "maximum_drawdown": maximum_drawdown_from_returns(marks.pct_change().dropna()),
        "per_period_excess_mean": float(scoped["excess"].mean()),
        "per_period_excess_block_ci_low": float(interval["low"]),
        "per_period_excess_block_ci_high": float(interval["high"]),
    }


# --------------------------------------------------------------------------- #
# Experiment E: concentration
# --------------------------------------------------------------------------- #
def concentration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Profit share of the largest contributors, and leave-one-out on the top ten."""

    contribution = (
        frame.groupby("security_id", sort=False)
        .agg(
            ticker=("ticker", "first"),
            trades=("net_return", "size"),
            total_net=("net_return", "sum"),
            mean_net=("net_return", "mean"),
            mean_spy_excess=("spy_excess_return", "mean"),
        )
        .sort_values("total_net", ascending=False)
    )
    total = float(contribution["total_net"].sum())
    positive = float(contribution.loc[contribution["total_net"].gt(0), "total_net"].sum())
    shares = {
        f"top_{count}_share_of_total_net": float(
            contribution["total_net"].head(count).sum() / total
        )
        for count in (1, 5, 20)
    }
    shares.update(
        {
            f"top_{count}_share_of_gross_profit": float(
                contribution["total_net"].head(count).sum() / positive
            )
            for count in (1, 5, 20)
        }
    )
    records: list[dict[str, object]] = []
    for security_id in contribution.head(10).index:
        remainder = frame.loc[~frame["security_id"].eq(security_id)]
        row = contribution.loc[security_id]
        records.append(
            {
                "security_id": security_id,
                "ticker": row["ticker"],
                "trades": int(row["trades"]),
                "total_net_contribution": float(row["total_net"]),
                "share_of_total_net": float(row["total_net"]) / total,
                **{f"remaining_{key}": value for key, value in descriptive(remainder).items()},
            }
        )
    return pd.DataFrame.from_records(records), shares


# --------------------------------------------------------------------------- #
# Experiment G: the drawdown episode
# --------------------------------------------------------------------------- #
def drawdown_episodes(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Peak-to-trough episodes of the session-mean net-return equity curve."""

    session_returns = (
        frame.groupby("session", sort=True)["net_return"].mean().sort_index()
    )
    equity = (1.0 + session_returns.clip(lower=-0.999999)).cumprod()
    peak = equity.cummax()
    underwater = 1.0 - equity / peak
    records: list[dict[str, object]] = []
    in_episode = False
    peak_session = equity.index[0]
    trough_session = equity.index[0]
    depth = 0.0
    for session, value in underwater.items():
        if value > 0.0 and not in_episode:
            in_episode = True
            peak_session = session
            trough_session = session
            depth = value
        elif value > 0.0:
            if value > depth:
                depth = float(value)
                trough_session = session
        elif in_episode:
            if depth >= threshold:
                records.append(
                    {
                        "peak_session": str(pd.Timestamp(peak_session).date()),
                        "trough_session": str(pd.Timestamp(trough_session).date()),
                        "recovered_session": str(pd.Timestamp(session).date()),
                        "depth": depth,
                    }
                )
            in_episode = False
            depth = 0.0
    if in_episode and depth >= threshold:
        records.append(
            {
                "peak_session": str(pd.Timestamp(peak_session).date()),
                "trough_session": str(pd.Timestamp(trough_session).date()),
                "recovered_session": "",
                "depth": depth,
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["peak_session", "trough_session", "recovered_session", "depth"]
    )


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(directory: Path, *, request: Mapping[str, object], status: str) -> str:
    """Write the immutable manifest and authority record over the emitted files."""

    request_path = directory / "_request.json"
    _write_json(request_path, dict(request))
    request_sha = _sha256(request_path)
    artifacts = sorted(
        (
            {
                "bytes": path.stat().st_size,
                "path": path.name,
                "sha256": _sha256(path),
            }
            for path in directory.iterdir()
            if path.is_file() and path.name not in {"_manifest.json", "_authority.json"}
        ),
        key=lambda record: str(record["path"]),
    )
    manifest = {
        "artifacts": artifacts,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha,
        "schema": SCHEMA,
        "status": status,
        "strategy_id": STRATEGY_ID,
    }
    manifest_path = directory / "_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        directory / "_authority.json",
        {
            "artifact": "_manifest.json",
            "artifact_sha256": _sha256(manifest_path),
            "request_sha256": request_sha,
            "schema": "edge_rebuild.swing_failure_attribution.authority.v1",
            "state": "complete",
        },
    )
    return request_sha


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
def _cohort_dimensions(declaration: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    experiment = declaration["experiment_b_cohorts"]
    return [
        (str(dimension["id"]), [str(value) for value in dimension["cohorts"]])
        for dimension in experiment["dimensions"]
    ]


def run_cohorts(
    frozen: pd.DataFrame, declaration: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    gates: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    baseline = evaluate(frozen, label="baseline:whole_population")
    gates.extend(gate_records(baseline, dimension="baseline", cohort="whole_population"))
    summary.append(
        {
            "dimension": "baseline",
            "cohort": "whole_population",
            "admitted_walk_forward": _scope_admitted(baseline, "walk_forward"),
            "admitted_unseen_ticker": _scope_admitted(baseline, "unseen_ticker"),
            "admitted_both": baseline.admitted,
            **descriptive(frozen),
        }
    )
    evaluated = 0
    for dimension, cohorts in _cohort_dimensions(declaration):
        for cohort in cohorts:
            subset = frozen.loc[frozen[dimension].astype(str).eq(cohort)]
            evaluation = evaluate(subset, label=f"{dimension}={cohort}")
            evaluated += 1
            gates.extend(gate_records(evaluation, dimension=dimension, cohort=cohort))
            summary.append(
                {
                    "dimension": dimension,
                    "cohort": cohort,
                    "admitted_walk_forward": _scope_admitted(evaluation, "walk_forward"),
                    "admitted_unseen_ticker": _scope_admitted(evaluation, "unseen_ticker"),
                    "admitted_both": evaluation.admitted,
                    **descriptive(subset),
                }
            )
    return pd.DataFrame.from_records(gates), pd.DataFrame.from_records(summary), evaluated


def _scope_admitted(evaluation: Evaluation, scope: str) -> bool:
    if evaluation.report is None:
        return False
    return evaluation.report.scope(scope).admitted


def run_topk(
    uncapped: pd.DataFrame,
    declaration: Mapping[str, Any],
    *,
    calendar: Sequence[pd.Timestamp],
    spy_forward: Mapping[pd.Timestamp, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment = declaration["experiment_a_topk_selection"]
    rankings = [str(entry["id"]) for entry in experiment["rankings"]]
    k_values = [int(value) for value in experiment["k_values"]]
    gates: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    portfolio: list[dict[str, object]] = []
    for ranking in rankings:
        for k in k_values:
            selected = top_k(uncapped, ranking=ranking, k=k)
            evaluation = evaluate(selected, label=f"{ranking}:top{k}")
            gates.extend(gate_records(evaluation, ranking=ranking, k=k))
            periods = sleeve_periods(selected, calendar=calendar)
            for scope in ("walk_forward", "unseen_ticker"):
                scoped = selected.loc[selected["scope"].eq(scope)]
                summary.append(
                    {
                        "ranking": ranking,
                        "k": k,
                        "scope": scope,
                        "admitted": _scope_admitted(evaluation, scope),
                        **descriptive(scoped),
                    }
                )
                portfolio.append(
                    {
                        "ranking": ranking,
                        "k": k,
                        "scope": scope,
                        **portfolio_economics(
                            periods,
                            scope=scope,
                            calendar=calendar,
                            spy_forward=spy_forward,
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(gates),
        pd.DataFrame.from_records(summary),
        pd.DataFrame.from_records(portfolio),
    )


def run_per_security(frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    per_security = (
        frozen.groupby("security_id", sort=True)
        .agg(
            ticker=("ticker", "first"),
            sector=("sector", "first"),
            scope=("scope", "first"),
            trades=("net_return", "size"),
            mean_net=("net_return", "mean"),
            mean_spy_excess=("spy_excess_return", "mean"),
            total_net=("net_return", "sum"),
        )
        .reset_index()
    )
    reported = per_security.loc[per_security["trades"].ge(MINIMUM_SECURITY_TRADES)]
    deciles = [round(0.1 * step, 1) for step in range(1, 10)]
    stats: dict[str, object] = {
        "securities_total": int(len(per_security)),
        "securities_reported": int(len(reported)),
        "minimum_trades_for_reporting": MINIMUM_SECURITY_TRADES,
        "trades_per_security_mean": float(per_security["trades"].mean()),
        "trades_per_security_median": float(per_security["trades"].median()),
        "trades_per_security_max": int(per_security["trades"].max()),
        "securities_with_at_least_20_trades": int(per_security["trades"].ge(20).sum()),
        "securities_with_at_least_30_trades": int(per_security["trades"].ge(30).sum()),
        "securities_with_at_least_50_trades": int(per_security["trades"].ge(50).sum()),
        "net_positive_securities": int(reported["mean_net"].gt(0).sum()),
        "spy_excess_positive_securities": int(reported["mean_spy_excess"].gt(0).sum()),
        "mean_net_deciles": {
            str(q): float(reported["mean_net"].quantile(q)) for q in deciles
        },
        "mean_spy_excess_deciles": {
            str(q): float(reported["mean_spy_excess"].quantile(q)) for q in deciles
        },
        "trades_per_security_deciles": {
            str(q): float(per_security["trades"].quantile(q)) for q in deciles
        },
    }
    return per_security, stats


def run_generalisation(frozen: pd.DataFrame) -> dict[str, object]:
    """Do in-sample per-security winners stay winners in later, non-overlapping time?"""

    walk = frozen.loc[frozen["scope"].eq("walk_forward")].copy()
    walk["exit_session_date_et"] = pd.to_datetime(walk["exit_session_date_et"])
    rank_window = walk.loc[walk["exit_session_date_et"].lt(GENERALISATION_SPLIT)]
    test_window = walk.loc[walk["session"].ge(GENERALISATION_SPLIT)]
    ranked = (
        rank_window.groupby("security_id", sort=True)
        .agg(trades=("spy_excess_return", "size"), mean_spy_excess=("spy_excess_return", "mean"))
        .query(f"trades >= {MINIMUM_SECURITY_TRADES}")
        .sort_values("mean_spy_excess", ascending=False)
    )
    later = test_window.groupby("security_id", sort=True)["spy_excess_return"].agg(
        ["size", "mean"]
    )
    tercile = max(int(len(ranked) / 3), 1)
    winners = list(ranked.head(tercile).index)
    losers = list(ranked.tail(tercile).index)
    overlap = ranked.join(later, how="inner")
    correlation = (
        float(overlap["mean_spy_excess"].corr(overlap["mean"], method="spearman"))
        if len(overlap) > 2
        else float("nan")
    )
    winner_rows = test_window.loc[test_window["security_id"].isin(winners)]
    loser_rows = test_window.loc[test_window["security_id"].isin(losers)]
    return {
        "split_session": str(GENERALISATION_SPLIT.date()),
        "rank_window_rows": int(len(rank_window)),
        "rank_window_securities_ranked": int(len(ranked)),
        "test_window_rows": int(len(test_window)),
        "top_tercile_size": int(len(winners)),
        "top_tercile_rank_window_mean_spy_excess": float(
            ranked.head(tercile)["mean_spy_excess"].mean()
        ),
        "top_tercile_test_window_rows": int(len(winner_rows)),
        "top_tercile_test_window_mean_spy_excess": (
            float(winner_rows["spy_excess_return"].mean()) if len(winner_rows) else float("nan")
        ),
        "top_tercile_test_window_mean_net": (
            float(winner_rows["net_return"].mean()) if len(winner_rows) else float("nan")
        ),
        "bottom_tercile_rank_window_mean_spy_excess": float(
            ranked.tail(tercile)["mean_spy_excess"].mean()
        ),
        "bottom_tercile_test_window_mean_spy_excess": (
            float(loser_rows["spy_excess_return"].mean()) if len(loser_rows) else float("nan")
        ),
        "test_window_all_securities_mean_spy_excess": float(
            test_window["spy_excess_return"].mean()
        ),
        "securities_in_both_windows": int(len(overlap)),
        "spearman_rank_correlation": correlation,
    }


def run_characteristic_generalisation(
    frozen: pd.DataFrame, dimensions: Iterable[str]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scope_means = frozen.groupby("scope", sort=True)["spy_excess_return"].mean()
    for dimension in dimensions:
        for cohort, subset in frozen.groupby(dimension, sort=True):
            record: dict[str, object] = {"dimension": dimension, "cohort": str(cohort)}
            for scope in ("walk_forward", "unseen_ticker"):
                scoped = subset.loc[subset["scope"].eq(scope)]
                advantage = (
                    float(scoped["spy_excess_return"].mean()) - float(scope_means.get(scope, np.nan))
                    if len(scoped)
                    else float("nan")
                )
                record[f"{scope}_rows"] = int(len(scoped))
                record[f"{scope}_mean_spy_excess"] = (
                    float(scoped["spy_excess_return"].mean()) if len(scoped) else float("nan")
                )
                record[f"{scope}_advantage"] = advantage
            walk = record["walk_forward_advantage"]
            unseen = record["unseen_ticker_advantage"]
            record["sign_agreement"] = bool(
                isinstance(walk, float)
                and isinstance(unseen, float)
                and np.isfinite(walk)
                and np.isfinite(unseen)
                and walk * unseen > 0
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def run_benchmark_decomposition(
    frozen: pd.DataFrame, unconditional: pd.DataFrame
) -> dict[str, object]:
    span = unconditional.loc[
        unconditional["session"].between(frozen["session"].min(), frozen["session"].max())
    ]
    deciles = [round(0.1 * step, 1) for step in range(1, 10)]
    conditional = frozen["spy_interval_return"]
    reference = span["spy_forward_return"]
    return {
        "conditional_rows": int(len(conditional)),
        "conditional_mean": float(conditional.mean()),
        "conditional_median": float(conditional.median()),
        "conditional_deciles": {str(q): float(conditional.quantile(q)) for q in deciles},
        "unconditional_rows": int(len(reference)),
        "unconditional_mean": float(reference.mean()),
        "unconditional_median": float(reference.median()),
        "unconditional_deciles": {str(q): float(reference.quantile(q)) for q in deciles},
        "conditional_minus_unconditional_mean": float(conditional.mean() - reference.mean()),
        "setup_mean_net_return": float(frozen["net_return"].mean()),
        "setup_mean_gross_return": float(frozen["gross_return"].mean()),
        "mean_spy_excess_return": float(frozen["spy_excess_return"].mean()),
        "identity_check_net_minus_conditional_spy": float(
            frozen["net_return"].mean() - conditional.mean()
        ),
        "distinct_decision_sessions": int(frozen["session"].nunique()),
        "unconditional_sessions_in_span": int(len(span)),
        "session_coverage_fraction": float(frozen["session"].nunique()) / float(len(span)),
    }


def run_drawdown(frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    records: list[dict[str, object]] = []
    worst: dict[str, object] = {"depth": -1.0}
    for (scope, phase), subset in frozen.groupby(["scope", "phase"], sort=True):
        session_returns = subset.groupby("session", sort=True)["net_return"].mean().sort_index()
        depth = maximum_drawdown_from_returns(session_returns)
        episodes = drawdown_episodes(subset, threshold=0.20)
        records.append(
            {
                "scope": scope,
                "phase": int(phase),
                "rows": int(len(subset)),
                "sessions": int(subset["session"].nunique()),
                "maximum_drawdown": depth,
                "episodes_over_20pct": int(len(episodes)),
            }
        )
        if depth > float(worst["depth"]):
            deepest = (
                episodes.sort_values("depth", ascending=False).iloc[0].to_dict()
                if len(episodes)
                else {}
            )
            worst = {
                "scope": scope,
                "phase": int(phase),
                "depth": depth,
                "episodes_over_20pct": int(len(episodes)),
                **{f"deepest_{key}": value for key, value in deepest.items()},
            }
    return pd.DataFrame.from_records(records), worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Directory holding the inputs.")
    parser.add_argument("--out", type=Path, required=True, help="Artifact directory to publish.")
    arguments = parser.parse_args()
    work: Path = arguments.work
    out: Path = arguments.out
    if out.exists() and (out / "_authority.json").exists():
        raise SystemExit(f"artifact directory is already published: {out}")
    out.mkdir(parents=True, exist_ok=True)

    declaration_path = work / "declaration.json"
    declaration_sha = _sha256(declaration_path)
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))

    frozen = with_declared_features(pd.read_parquet(work / "population_enriched.parquet"))
    uncapped = with_declared_features(pd.read_parquet(work / "uncapped_enriched.parquet"))
    unconditional = pd.read_parquet(work / "spy_unconditional.parquet")
    unconditional["session"] = pd.to_datetime(unconditional["session"])
    calendar = list(unconditional["session"])
    spy_forward = dict(zip(unconditional["session"], unconditional["spy_forward_return"], strict=True))

    topk_gates, topk_summary, topk_portfolio = run_topk(
        uncapped, declaration, calendar=calendar, spy_forward=spy_forward
    )
    cohort_gates, cohort_summary, cohorts_evaluated = run_cohorts(frozen, declaration)
    leave_one_out, shares = concentration(frozen)
    per_security, per_security_stats = run_per_security(frozen)
    generalisation = run_generalisation(frozen)
    characteristics = run_characteristic_generalisation(
        frozen,
        [str(value) for value in declaration["experiment_d_characteristic_generalisation"]["dimensions"]],
    )
    benchmark = run_benchmark_decomposition(frozen, unconditional)
    drawdown, worst_drawdown = run_drawdown(frozen)

    topk_gates.to_csv(out / "topk_gates.csv", index=False)
    topk_summary.to_csv(out / "topk_summary.csv", index=False)
    topk_portfolio.to_csv(out / "topk_portfolio.csv", index=False)
    cohort_gates.to_csv(out / "cohort_gates.csv", index=False)
    cohort_summary.to_csv(out / "cohort_summary.csv", index=False)
    leave_one_out.to_csv(out / "concentration_leave_one_out.csv", index=False)
    per_security.to_csv(out / "per_security.csv", index=False)
    characteristics.to_csv(out / "characteristic_generalisation.csv", index=False)
    drawdown.to_csv(out / "drawdown_by_phase.csv", index=False)
    (out / "declaration.json").write_text(
        declaration_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    admitted_cohorts = cohort_summary.loc[
        cohort_summary["admitted_both"] & cohort_summary["dimension"].ne("baseline")
    ]
    admitted_topk = topk_summary.groupby(["ranking", "k"], sort=True)["admitted"].all()
    summary = {
        "schema": SCHEMA,
        "strategy_id": STRATEGY_ID,
        "declaration_sha256": declaration_sha,
        "cohorts_evaluated": cohorts_evaluated,
        "cohort_scope_records": int(len(cohort_summary) * 2),
        "cohort_gate_records": int(len(cohort_gates)),
        "cohorts_admitted_in_both_scopes": int(len(admitted_cohorts)),
        "admitted_cohorts": sorted(
            f"{row.dimension}={row.cohort}" for row in admitted_cohorts.itertuples()
        ),
        "topk_configurations": int(len(admitted_topk)),
        "topk_admitted_in_both_scopes": int(admitted_topk.sum()),
        "whole_population": descriptive(frozen),
        "uncapped_population": descriptive(uncapped),
        "concentration": shares,
        "per_security": per_security_stats,
        "out_of_time_generalisation": generalisation,
        "characteristic_sign_agreement": {
            str(dimension): {
                "cohorts": int(len(group)),
                "sign_agreeing": int(group["sign_agreement"].sum()),
            }
            for dimension, group in characteristics.groupby("dimension", sort=True)
        },
        "benchmark_decomposition": benchmark,
        "worst_drawdown": worst_drawdown,
        "bootstrap_iterations": REGIME_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": REGIME_BOOTSTRAP_SEED,
        "setup_economics_config_sha256": SWING_SETUP_ECONOMICS_CONFIG.sha256(),
    }
    _write_json(out / "summary.json", summary)

    status = (
        "no_admitted_subset"
        if not len(admitted_cohorts) and not int(admitted_topk.sum())
        else "admitted_subset_present"
    )
    request_sha = publish(
        out,
        request={
            "declaration_sha256": declaration_sha,
            "frozen_population": {
                "rows": int(len(frozen)),
                "securities": int(frozen["security_id"].nunique()),
                "sessions": int(frozen["session"].nunique()),
                "strategy_contract_sha256": declaration["populations"]["frozen"][
                    "strategy_contract_sha256"
                ],
            },
            "uncapped_population": {
                "rows": int(len(uncapped)),
                "securities": int(uncapped["security_id"].nunique()),
                "strategy_contract_sha256": declaration["populations"]["uncapped"][
                    "strategy_contract_sha256"
                ],
            },
            "implementation": {
                "path": "scripts/swing_failure_attribution.py",
                "sha256": _sha256(Path(__file__)),
            },
            "memberships_sha256": (
                "0e222a234dbfbc600dc12e25bec2eb4b75782b46e720d1df4d6a2ae541cdd656"
            ),
            "schema": SCHEMA,
            "setup_economics_config_sha256": SWING_SETUP_ECONOMICS_CONFIG.sha256(),
            "source": {
                "bars": "data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3",
                "memberships": (
                    "data/canonical/swing_memberships_verified_20190709_20260708_v2.parquet"
                ),
                "news": [
                    "data/raw/alpaca_news_20190709_20210708_v1",
                    "data/raw/alpaca_news_20210709_20260708_v1",
                ],
            },
            "strategy_id": STRATEGY_ID,
        },
        status=status,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("request_sha256", request_sha)
    print("status", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
