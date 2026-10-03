"""One decision-to-query binding shared by adjusted predictor consumers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.adjusted_history_source import (
    AdjustedHistorySource,
    AdjustedHistoryUnit,
    load_adjusted_history_source,
    read_adjusted_history_unit,
)
from market_predictor.swing.datasets.history_plan_publication import UNIT_COLUMNS
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.adjusted_source import expected_adjusted_history_sessions


@dataclass(frozen=True)
class AdjustedHistoryBindings:
    source: AdjustedHistorySource
    windows: Mapping[str, Mapping[str, Mapping[str, str]]]
    source_files: Mapping[str, str]


def load_adjusted_history_bindings(*, root: Path, plan_authority: SourcePin,
    archive_authority: SourcePin,
) -> AdjustedHistoryBindings:
    source = load_adjusted_history_source(root=root, plan_authority=plan_authority, archive_authority=archive_authority)
    plan = (source.root / plan_authority.path).parent
    relative = (plan / "_request.json").relative_to(source.root).as_posix()
    request = pinned_object(plan / "_request.json", source.source_files[relative])
    by_query = {tuple(str(record[column]) for column in UNIT_COLUMNS): unit_id
        for unit_id, record in source.records.items()}
    if len(by_query) != len(source.records):
        raise DataReadinessError("adjusted source repeats a query window")
    rows = request.get("parent_window_mapping")
    if not isinstance(rows, list) or len(rows) != len(by_query):
        raise DataReadinessError("adjusted source lacks complete parent-window bindings")
    windows: dict[str, Mapping[str, Mapping[str, str]]] = {}
    for entry in rows:
        if (not isinstance(entry, dict) or set(entry) != {"parent", "query"}
                or any(not isinstance(entry[name], dict) or set(entry[name]) != set(UNIT_COLUMNS)
                    for name in ("parent", "query"))):
            raise DataReadinessError("adjusted source parent-window binding is malformed")
        parent, query = entry["parent"], entry["query"]
        unit_id = by_query.get(tuple(query[column] for column in UNIT_COLUMNS))
        if (unit_id is None or unit_id in windows or parent["security_id"] != query["security_id"]
                or parent["role"] != query["role"]):
            raise DataReadinessError("adjusted source parent/query inventory differs")
        windows[unit_id] = MappingProxyType({"parent": MappingProxyType(dict(parent)), "query": MappingProxyType(dict(query))})
    return AdjustedHistoryBindings(source, MappingProxyType(windows), source.source_files)


def bind_adjusted_history_decisions(decisions: pd.DataFrame, bindings: AdjustedHistoryBindings) -> pd.DataFrame:
    """Keep corrected decision tickers; select exactly one original parent window."""
    required = {"decision_id", "security_id", "parent_ticker", "session_date_et"}
    if not required.issubset(decisions) or decisions.decision_id.duplicated().any() or decisions.loc[:, list(required)].isna().any().any():
        raise DataReadinessError("adjusted binding requires unique complete decision identities")
    result = decisions.reset_index(drop=True).copy()
    result["source_group"] = ""
    candidates: dict[tuple[str, str], list[tuple[str, date, date]]] = {}
    for unit_id, window in bindings.windows.items():
        parent = window["parent"]
        if parent["role"] == "stock":
            candidates.setdefault((parent["security_id"], parent["ticker"]), []).append((unit_id,
                date.fromisoformat(parent["start_date"]), date.fromisoformat(parent["end_date"])))
    for key, group in result.groupby(["security_id", "parent_ticker"], sort=False):
        hits = pd.Series(0, index=group.index)
        for unit_id, first, last in candidates.get((str(key[0]), str(key[1])), ()):
            selected = group.session_date_et.between(first, last)
            hits.loc[selected] += 1
            result.loc[group.index[selected], "source_group"] = unit_id
        if not hits.eq(1).all():
            raise DataReadinessError("adjusted decision must bind exactly one original parent query window")
    return result


def expected_bound_history_sessions(bindings: AdjustedHistoryBindings, unit_id: str,
    memberships: pd.DataFrame,
) -> tuple[date, ...]:
    """Membership determines usable history independently of returned or missing bars."""
    window = bindings.windows[unit_id]
    parent, query = window["parent"], window["query"]
    if parent["role"] != "stock":
        raise DataReadinessError("stock history requirements cannot use a benchmark window")
    owned = memberships.loc[memberships.ticker.eq(parent["ticker"])]
    required = expected_adjusted_history_sessions(security_id=parent["security_id"], memberships=owned,
        sparse_missing_sessions_by_ticker={})
    first, last = date.fromisoformat(query["start_date"]), date.fromisoformat(query["end_date"])
    expected = tuple(day for day in required if first <= day <= last)
    if not expected:
        raise DataReadinessError("adjusted query has no independently required membership history")
    return expected


def read_bound_adjusted_history(bindings: AdjustedHistoryBindings, unit_id: str,
    expected_sessions: tuple[date, ...],
) -> AdjustedHistoryUnit:
    if unit_id not in bindings.windows or not expected_sessions or tuple(sorted(set(expected_sessions))) != expected_sessions:
        raise DataReadinessError("adjusted bound history requires ordered independent sessions")
    query = bindings.windows[unit_id]["query"]
    first, last = date.fromisoformat(query["start_date"]), date.fromisoformat(query["end_date"])
    if any(not first <= day <= last for day in expected_sessions):
        raise DataReadinessError("adjusted expected history escapes the selected query window")
    unit = read_adjusted_history_unit(bindings.source, unit_id)
    bars = unit.bars
    dates = pd.to_datetime(bars.bar_start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    expected = set(expected_sessions)
    clipped = bars.loc[dates.isin(expected)].reset_index(drop=True)
    return AdjustedHistoryUnit(clipped, tuple(day for day in unit.missing_sessions if day in expected),
        tuple(day for day in unit.invalid_sessions if day in expected), unit.record)


def read_bound_benchmarks(bindings: AdjustedHistoryBindings, symbols: set[str]) -> pd.DataFrame:
    """Read exactly one strict benchmark unit per required historical symbol."""
    units: dict[str, str] = {}
    for unit_id, record in bindings.source.records.items():
        if record["role"] == "benchmark" and record["ticker"] in symbols:
            if record["ticker"] in units:
                raise DataReadinessError("adjusted benchmark has ambiguous query units")
            units[record["ticker"]] = unit_id
    if not symbols or set(units) != symbols:
        raise DataReadinessError("adjusted source lacks required benchmark query units")
    return pd.concat([read_adjusted_history_unit(bindings.source, units[symbol]).bars
        for symbol in sorted(symbols)], ignore_index=True)
