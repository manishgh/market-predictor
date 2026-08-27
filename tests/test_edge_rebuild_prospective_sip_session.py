from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pandas as pd
import pytest

import market_predictor.edge_rebuild.prospective_sip_session as module
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.history_contracts import (
    load_intraday_history_config,
    load_selected_session_benchmark_config,
)
from market_predictor.sources.alpaca import AlpacaBarsPage
from market_predictor.universe.sp500.observed_membership_authority import (
    ObservedMembershipAuthority,
)

ROOT = Path(__file__).resolve().parents[1]
FIVE_MINUTE_POLICY = ROOT / "configs" / "edge_rebuild_intraday_history.toml"
BENCHMARK_POLICY = (
    ROOT / "configs" / "edge_rebuild_selected_session_benchmarks.toml"
)
REGULAR_SESSION = date(2024, 7, 8)
COLLECTION_TIME = datetime(2024, 7, 8, 20, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    ("session_date", "open_utc", "close_utc", "five_minute_bars", "one_minute_bars"),
    [
        (date(2024, 7, 8), "2024-07-08T13:30:00Z", "2024-07-08T20:00:00Z", 78, 390),
        (date(2024, 7, 3), "2024-07-03T13:30:00Z", "2024-07-03T17:00:00Z", 42, 210),
    ],
)
def test_closed_session_bounds_follow_regular_and_early_close_calendars(
    session_date: date,
    open_utc: str,
    close_utc: str,
    five_minute_bars: int,
    one_minute_bars: int,
) -> None:
    _, opened, closed, _ = module._closed_session_bounds(
        session_date,
        calendar_name="XNYS",
        finalization_delay_seconds=60,
        now_utc=pd.Timestamp(close_utc).to_pydatetime()
        + pd.Timedelta(minutes=2).to_pytimedelta(),
    )

    assert opened == pd.Timestamp(open_utc)
    assert closed == pd.Timestamp(close_utc)
    assert int((closed - opened) / pd.Timedelta(minutes=5)) == five_minute_bars
    assert int((closed - opened) / pd.Timedelta(minutes=1)) == one_minute_bars


@pytest.mark.parametrize(
    ("now_utc", "message"),
    [
        (datetime(2024, 7, 8, 20, 0, 59, tzinfo=UTC), "have not finalized"),
        (datetime(2024, 7, 9, 13, 30, tzinfo=UTC), "next XNYS open"),
    ],
)
def test_closed_session_bounds_reject_outside_post_close_window(
    now_utc: datetime,
    message: str,
) -> None:
    with pytest.raises(DataReadinessError, match=message):
        module._closed_session_bounds(
            REGULAR_SESSION,
            calendar_name="XNYS",
            finalization_delay_seconds=60,
            now_utc=now_utc,
        )


def test_session_membership_filters_by_causal_availability_and_effective_window(
    tmp_path: Path,
) -> None:
    open_at = pd.Timestamp("2024-07-08T13:30:00Z")
    memberships = pd.DataFrame(
        [
            _membership_row("ACTIVE", "sec-active", available="2024-07-08T13:29:59Z"),
            _membership_row("LATE", "sec-late", available="2024-07-08T13:30:01Z"),
            _membership_row(
                "FUTURE",
                "sec-future",
                available="2024-07-08T13:00:00Z",
                effective_from="2024-07-08T13:30:01Z",
            ),
            _membership_row(
                "EXPIRED",
                "sec-expired",
                available="2024-07-08T13:00:00Z",
                effective_to="2024-07-08T13:30:00Z",
            ),
            _membership_row(
                "ENDING_LATER",
                "sec-ending",
                available="2024-07-08T13:00:00Z",
                effective_to="2024-07-08T13:30:01Z",
            ),
        ]
    )
    observed = ObservedMembershipAuthority(
        directory=tmp_path,
        memberships=memberships,
        manifest={},
        parent={},
    )

    active = module._session_membership(
        observed,
        open_at=open_at,
        minimum_cross_section=2,
    )

    assert active["ticker"].tolist() == ["ACTIVE", "ENDING_LATER"]


def test_collection_plans_full_cohort_and_exact_benchmarks_from_exact_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, observed, calls = _publish_authority(tmp_path, monkeypatch)

    manifest = module.load_complete_prospective_sip_session(output)
    request = _read_json(output / "_request.json")
    full_units = _read_plan_units(output / "plans" / "full_cohort_5m")
    benchmark_units = _read_plan_units(output / "plans" / "benchmarks_1m")
    planned_stocks = {
        symbol
        for value in full_units["canonical_symbols_json"]
        for symbol in json.loads(value)
    }
    planned_benchmarks = {
        symbol
        for value in benchmark_units["canonical_symbols_json"]
        for symbol in json.loads(value)
    }

    assert planned_stocks == set(observed.memberships["ticker"])
    assert full_units["symbol_count"].max() <= 50
    assert len(full_units) == 9
    assert planned_benchmarks == module.REQUIRED_BENCHMARKS
    assert len(planned_benchmarks) == 13
    assert request["benchmark_symbols"] == list(
        load_selected_session_benchmark_config(BENCHMARK_POLICY).normalized_benchmarks()
    )
    assert {call[0] for call in calls} == {"5Min", "1Min"}
    assert all(call[1] == REGULAR_SESSION for call in calls)
    assert manifest["status"] == "source_complete_warmup_ineligible"
    assert manifest["selection_status"] == "warmup_incomplete"
    assert manifest["required_prior_five_minute_sessions"] == 20
    assert manifest["training_eligible"] is False
    assert manifest["serving_eligible"] is False
    assert manifest["selection_eligible"] is False
    assert manifest["coverage_status"] == "complete"
    assert manifest["coverage"]["benchmark_incomplete_symbols"] == []


def test_complete_authority_rejects_parent_and_child_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _ = _publish_authority(tmp_path, monkeypatch)
    child_tampered = tmp_path / "child-tampered"
    shutil.copytree(output, child_tampered)
    child_manifest = (
        child_tampered / "collections" / "full_cohort_5m" / "_manifest.json"
    )
    child_manifest.write_bytes(child_manifest.read_bytes() + b"\n")

    with pytest.raises(DataReadinessError):
        module.load_complete_prospective_sip_session(child_tampered)

    membership_path = tmp_path / "membership-authority" / "memberships.parquet"
    membership_path.write_bytes(membership_path.read_bytes() + b"tampered")

    with pytest.raises(DataReadinessError, match="membership parent changed"):
        module.load_complete_prospective_sip_session(output)


def test_incomplete_bar_coverage_publishes_no_source_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _ = _publish_authority(
        tmp_path,
        monkeypatch,
        full_grid=False,
        expected_status="source_incomplete_coverage",
    )

    assert not (output / "_authority.json").exists()
    status = _read_json(output / "_status.json")
    assert status["coverage_status"] == "incomplete"


def test_child_plan_identity_cannot_be_reused_for_another_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _ = _publish_authority(tmp_path, monkeypatch)
    parent = _read_json(output / "_request.json")
    plan = output / "plans" / "full_cohort_5m"
    request_path = plan / "_request.json"
    child = _read_json(request_path)
    child["symbols"] = ["WRONG"]
    request_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="child plan identity changed"):
        module._verify_child_plan_identity(
            plan,
            parent_request=parent,
            expected_schema=module.INTRADAY_HISTORY_PLAN_SCHEMA,
            expected_timeframe="5Min",
            expected_symbols=parent["full_cohort_symbols"],
        )


def test_loader_recomputes_exchange_calendar_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _ = _publish_authority(tmp_path, monkeypatch)
    request_path = output / "_request.json"
    request = _read_json(request_path)
    request["session_close_utc"] = "2024-07-08T19:59:00+00:00"
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request["request_sha256"] = module._json_sha256(payload)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    for name in ("_manifest.json", "_status.json"):
        path = output / name
        value = _read_json(path)
        value["request_sha256"] = request["request_sha256"]
        path.write_text(json.dumps(value), encoding="utf-8")
    authority_path = output / "_authority.json"
    authority = _read_json(authority_path)
    authority["request_sha256"] = request["request_sha256"]
    authority["artifact_sha256"] = module.file_sha256(
        output / "_manifest.json"
    )
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="calendar bounds changed"):
        module.load_complete_prospective_sip_session(output)


def test_collection_rejects_resource_policy_above_frozen_limits(
    tmp_path: Path,
) -> None:
    config = load_intraday_history_config(FIVE_MINUTE_POLICY).model_copy(
        update={"collection_workers": 3}
    )

    with pytest.raises(DataReadinessError, match="workers exceed two"):
        module.collect_prospective_sip_session(
            session_date=REGULAR_SESSION,
            membership_authority_directory=tmp_path / "membership",
            five_minute_policy_path=FIVE_MINUTE_POLICY,
            benchmark_policy_path=BENCHMARK_POLICY,
            output_directory=tmp_path / "output",
            five_minute_config=config,
            benchmark_config=load_selected_session_benchmark_config(
                BENCHMARK_POLICY
            ),
            source_factory=lambda: pytest.fail("network must not be called"),
            now_utc=COLLECTION_TIME,
        )


def test_parent_resume_loads_completed_children_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _ = _publish_authority(tmp_path, monkeypatch)
    (output / "_authority.json").unlink()
    (output / "_manifest.json").unlink()

    result = module.collect_prospective_sip_session(
        session_date=REGULAR_SESSION,
        membership_authority_directory=tmp_path / "membership-authority",
        five_minute_policy_path=FIVE_MINUTE_POLICY,
        benchmark_policy_path=BENCHMARK_POLICY,
        output_directory=output,
        five_minute_config=load_intraday_history_config(FIVE_MINUTE_POLICY),
        benchmark_config=load_selected_session_benchmark_config(BENCHMARK_POLICY),
        source_factory=lambda: pytest.fail("verified child resume must not call Alpaca"),
        now_utc=COLLECTION_TIME,
    )

    assert result["status"] == "source_complete_warmup_ineligible"


def test_five_percent_stock_exclusion_rule_is_explicit() -> None:
    full_coverage = {
        f"T{index:03d}": {
            "expected_rows": 78,
            "observed_rows": 0 if index < 5 else 78,
            "status": "unavailable" if index < 5 else "complete",
        }
        for index in range(100)
    }
    benchmark_coverage = {
        ticker: {
            "expected_rows": 390,
            "observed_rows": 390,
            "status": "complete",
        }
        for ticker in module.REQUIRED_BENCHMARKS
    }

    summary = module._source_coverage_summary(
        {"artifacts": [{"symbol_coverage": full_coverage}]},
        {"artifacts": [{"symbol_coverage": benchmark_coverage}]},
    )

    assert summary["status"] == "acceptable_with_exclusions"
    assert summary["full_cohort_incomplete_fraction"] == 0.05


def _publish_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_grid: bool = True,
    expected_status: str = "source_complete_warmup_ineligible",
) -> tuple[Path, ObservedMembershipAuthority, list[tuple[str, date]]]:
    membership_directory = tmp_path / "membership-authority"
    membership_directory.mkdir()
    membership_path = membership_directory / "memberships.parquet"
    membership_path.write_bytes(b"immutable membership evidence")
    (membership_directory / "_manifest.json").write_text("{}", encoding="utf-8")
    (membership_directory / "_authority.json").write_text("{}", encoding="utf-8")
    memberships = pd.DataFrame(
        [
            _membership_row(
                f"T{index:03d}",
                f"sec-{index:03d}",
                available="2024-07-08T12:00:00Z",
            )
            for index in range(450)
        ]
    )
    observed = ObservedMembershipAuthority(
        directory=membership_directory,
        memberships=memberships,
        manifest={
            "membership_artifact": {"path": membership_path.name},
            "observed_at_utc": "2024-07-08T12:00:00+00:00",
            "effective_horizon_date": REGULAR_SESSION.isoformat(),
        },
        parent={},
    )
    monkeypatch.setattr(
        module,
        "load_observed_sp500_membership_authority",
        lambda _directory: observed,
    )
    calls: list[tuple[str, date]] = []

    def source_factory() -> _ExactPageSource:
        return _ExactPageSource(calls, full_grid=full_grid)

    output = tmp_path / "prospective-session"
    result = module.collect_prospective_sip_session(
        session_date=REGULAR_SESSION,
        membership_authority_directory=membership_directory,
        five_minute_policy_path=FIVE_MINUTE_POLICY,
        benchmark_policy_path=BENCHMARK_POLICY,
        output_directory=output,
        five_minute_config=load_intraday_history_config(FIVE_MINUTE_POLICY),
        benchmark_config=load_selected_session_benchmark_config(BENCHMARK_POLICY),
        source_factory=source_factory,
        now_utc=COLLECTION_TIME,
    )
    assert result["status"] == expected_status
    return output, observed, calls


class _ExactPageSource:
    def __init__(
        self,
        calls: list[tuple[str, date]],
        *,
        full_grid: bool,
    ) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=30)
        self._calls = calls
        self._full_grid = full_grid

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        timeframe = str(kwargs["timeframe"])
        asof = kwargs["asof"]
        assert isinstance(asof, date)
        self._calls.append((timeframe, asof))
        interval = 5 if timeframe == "5Min" else 1
        timestamps = pd.date_range(
            pd.Timestamp(start),
            pd.Timestamp(end),
            freq=f"{interval}min",
        )
        if not self._full_grid:
            timestamps = timestamps[:2]
        bars = {
            symbol: tuple(
                {
                    "t": timestamp.isoformat(),
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1_000,
                }
                for timestamp in timestamps
            )
            for symbol in symbols
        }
        payload = {
            "bars": {symbol: list(rows) for symbol, rows in bars.items()},
            "next_page_token": None,
        }
        query = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": "sip",
            "limit": str(kwargs["limit"]),
            "adjustment": "all",
            "sort": "asc",
            "asof": asof.isoformat(),
        }
        requested_url = (
            "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(query)
        )
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars=bars,
            response_headers={"Content-Type": "application/json"},
            raw_payload=payload,
            raw_body=json.dumps(payload, separators=(",", ":")).encode(),
            requested_url=requested_url,
            status_code=200,
            retrieved_at_utc=COLLECTION_TIME,
            final_url=requested_url,
        )


def _membership_row(
    ticker: str,
    security_id: str,
    *,
    available: str,
    effective_from: str = "2024-01-01T00:00:00Z",
    effective_to: str | None = None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "security_id": security_id,
        "effective_from_utc": effective_from,
        "effective_to_utc": effective_to,
        "available_at_utc": available,
        "universe_snapshot_id": "observed-2024-07-08",
    }


def _read_plan_units(plan: Path) -> pd.DataFrame:
    files = sorted((plan / "units").rglob("*.parquet"))
    assert files
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
