from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.datasets.microstructure_history import (
    COLLECTION_AUTHORITY_SCHEMA,
    MicrostructureCollectionConfig,
    build_intraday_microstructure_plan,
    collect_intraday_microstructure_history,
    load_complete_intraday_microstructure_collection,
    load_complete_intraday_microstructure_plan,
    load_microstructure_collection_config,
)
from market_predictor.edge_rebuild.one_minute_coverage import (
    COVERAGE_AUTHORITY_SCHEMA,
    COVERAGE_SCHEMA,
)
from market_predictor.sources.alpaca import AlpacaQuotesPage, AlpacaTradesPage
from market_predictor.core.errors import DataReadinessError


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")


def _coverage(tmp_path: Path, *, sessions: int = 1, incomplete: bool = False) -> Path:
    root = tmp_path / "coverage"
    root.mkdir()
    rows: list[dict[str, Any]] = []
    for index in range(sessions):
        session = pd.Timestamp("2025-01-02") + pd.offsets.BDay(index)
        rows.append(
            {
                "session_date_et": session.date().isoformat(),
                "session_open_utc": pd.Timestamp(
                    f"{session.date().isoformat()} 14:30:00+00:00"
                ),
                "session_close_utc": pd.Timestamp(
                    f"{session.date().isoformat()} 21:00:00+00:00"
                ),
                "ticker": "AAPL",
                "coverage_status": "complete",
            }
        )
    if incomplete:
        rows.append(
            {
                "session_date_et": "2025-01-06",
                "session_open_utc": pd.Timestamp("2025-01-06 14:30:00+00:00"),
                "session_close_utc": pd.Timestamp("2025-01-06 21:00:00+00:00"),
                "ticker": "MSFT",
                "coverage_status": "incomplete",
            }
        )
    coverage_path = root / "stock_session_coverage.parquet"
    pd.DataFrame(rows).to_parquet(coverage_path, index=False)
    manifest = {
        "schema": COVERAGE_SCHEMA,
        "status": "ready",
        "ready_for_feature_build": True,
        "files": [
            {
                "path": "stock_session_coverage.parquet",
                "sha256": file_sha256(coverage_path),
                "bytes": coverage_path.stat().st_size,
                "rows": len(rows),
            }
        ],
    }
    _write_json(root / "_manifest.json", manifest)
    _write_json(
        root / "_authority.json",
        {
            "schema": COVERAGE_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(root / "_manifest.json"),
            "ready_for_feature_build": True,
        },
    )
    return root


class _FakeAlpacaSource:
    def __init__(
        self,
        *,
        fail_once: set[tuple[str, str]] | None = None,
        fail_pages_once: set[tuple[str, str, str | None]] | None = None,
    ) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=0)
        self.calls: list[tuple[str, str, str | None]] = []
        self.bounds: list[tuple[str, datetime, datetime]] = []
        self.fail_once = set(fail_once or set())
        self.fail_pages_once = set(fail_pages_once or set())

    def fetch_trades_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        page_token: str | None = None,
        asof: date | None = None,
        limit: int = 10_000,
        retries: int = 5,
    ) -> AlpacaTradesPage:
        del asof, limit, retries
        symbol = symbols[0]
        self._record("trades", symbol, page_token)
        self.bounds.append(("trades", start, end))
        suffix = "00" if page_token is None else "01"
        next_token = "trade-page-2" if page_token is None else None
        row = {
            "t": f"{start.date().isoformat()}T14:30:{suffix}Z",
            "p": 101.25,
            "s": 7,
            "x": "V",
            "c": ["@"],
            "i": 100 + int(suffix),
            "z": "C",
        }
        raw = {"trades": {symbol: [row]}, "next_page_token": next_token}
        return AlpacaTradesPage(
            request_page_token=page_token,
            next_page_token=next_token,
            trades={symbol: (row,)},
            response_headers={
                "X-RateLimit-Remaining": "9999",
                "Authorization": "must-not-persist",
            },
            raw_payload=raw,
        )

    def fetch_quotes_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        page_token: str | None = None,
        asof: date | None = None,
        limit: int = 10_000,
        retries: int = 5,
    ) -> AlpacaQuotesPage:
        del asof, limit, retries
        symbol = symbols[0]
        self._record("quotes", symbol, page_token)
        self.bounds.append(("quotes", start, end))
        row = {
            "t": f"{start.date().isoformat()}T14:30:00Z",
            "ap": 101.30,
            "as": 4,
            "ax": "Q",
            "bp": 101.20,
            "bs": 6,
            "bx": "P",
            "c": ["R"],
            "z": "C",
        }
        raw = {"quotes": {symbol: [row]}, "next_page_token": None}
        return AlpacaQuotesPage(
            request_page_token=page_token,
            next_page_token=None,
            quotes={symbol: (row,)},
            response_headers={"Retry-After": "1", "Set-Cookie": "secret"},
            raw_payload=raw,
        )

    def _record(self, event: str, symbol: str, page_token: str | None) -> None:
        key = (event, symbol)
        page_key = (event, symbol, page_token)
        self.calls.append((event, symbol, page_token))
        if page_key in self.fail_pages_once:
            self.fail_pages_once.remove(page_key)
            raise RuntimeError(f"planned {event} page failure")
        if key in self.fail_once:
            self.fail_once.remove(key)
            raise RuntimeError(f"planned {event} failure")


def _plan(tmp_path: Path, *, sessions: int = 1) -> Path:
    coverage = _coverage(tmp_path, sessions=sessions)
    plan = tmp_path / "plan"
    build_intraday_microstructure_plan(
        one_minute_coverage_directory=coverage,
        output_directory=plan,
    )
    return plan


def _config() -> MicrostructureCollectionConfig:
    return MicrostructureCollectionConfig(workers=1, maximum_pages_per_job=10)


def test_plan_keeps_all_selected_sessions_and_coverage_status_metadata(
    tmp_path: Path,
) -> None:
    coverage = _coverage(tmp_path, sessions=2, incomplete=True)
    plan = tmp_path / "plan"

    manifest = build_intraday_microstructure_plan(
        one_minute_coverage_directory=coverage,
        output_directory=plan,
    )

    assert manifest["units"] == 3
    assert manifest["jobs"] == 6
    assert manifest["symbols"] == 2
    assert manifest["included_stock_sessions_by_status"] == {
        "complete": 2,
        "incomplete": 1,
    }
    assert load_complete_intraday_microstructure_plan(plan)["plan_fingerprint"] == manifest["plan_fingerprint"]
    units = pd.read_parquet(plan / "units.parquet")
    assert set(units["ticker"]) == {"AAPL", "MSFT"}
    assert set(units["source_bar_coverage_status"]) == {
        "complete",
        "incomplete",
    }


def test_plan_and_collection_reject_overlapping_input_output_trees(
    tmp_path: Path,
) -> None:
    coverage = _coverage(tmp_path)
    with pytest.raises(DataReadinessError, match="trees overlap"):
        build_intraday_microstructure_plan(
            one_minute_coverage_directory=coverage,
            output_directory=coverage / "nested-plan",
        )

    plan = tmp_path / "plan"
    build_intraday_microstructure_plan(
        one_minute_coverage_directory=coverage,
        output_directory=plan,
    )
    with pytest.raises(DataReadinessError, match="trees overlap"):
        collect_intraday_microstructure_history(
            plan_directory=plan,
            output_directory=plan / "nested-collection",
            source_factory=_FakeAlpacaSource,  # type: ignore[arg-type]
            config=_config(),
            maximum_jobs_this_run=1,
        )


def test_plan_rejects_tampered_unit_artifact(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    units = pd.read_parquet(plan / "units.parquet")
    units.loc[0, "ticker"] = "MSFT"
    units.to_parquet(plan / "units.parquet", index=False)

    with pytest.raises(DataReadinessError, match="failed integrity"):
        load_complete_intraday_microstructure_plan(plan)


def test_collection_is_bounded_resumable_and_preserves_raw_pages(tmp_path: Path) -> None:
    plan = _plan(tmp_path, sessions=2)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()

    first = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )

    assert first["status"] == "transport_incomplete"
    assert first["completed_jobs"] == 1
    assert first["ready_for_materialization"] is False
    assert not (output / "_authority.json").exists()
    first_calls = list(source.calls)

    final = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=3,
    )

    assert final["status"] == "transport_complete"
    assert final["completed_jobs"] == 4
    assert final["ready_for_materialization"] is True
    assert source.calls[: len(first_calls)] == first_calls
    assert source.calls.count(("trades", "AAPL", None)) == 2
    verified = load_complete_intraday_microstructure_collection(output)
    assert verified["rows_by_event"] == {"trades": 4, "quotes": 2}
    first_trade_wrapper = next(
        wrapper
        for wrapper in verified["jobs"]
        if wrapper["event_type"] == "trades"
    )
    assert "job" not in first_trade_wrapper
    first_trade = json.loads(
        (output / first_trade_wrapper["path"]).read_text(encoding="utf-8")
    )
    assert [page["request_page_token"] for page in first_trade["pages"]] == [
        None,
        "trade-page-2",
    ]
    assert first_trade["observed_fields"] == ["c", "i", "p", "s", "t", "x", "z"]
    assert first_trade["regular_session_rows"] == first_trade["rows"]
    assert first_trade["pages"][0]["rate_headers"] == {
        "x-ratelimit-remaining": "9999"
    }
    raw_path = output / first_trade["pages"][0]["raw_page_path"]
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["trades"]["AAPL"][0] == {
        "c": ["@"],
        "i": 100,
        "p": 101.25,
        "s": 7,
        "t": "2025-01-02T14:30:00Z",
        "x": "V",
        "z": "C",
    }


def test_provider_request_uses_exact_session_close_boundary(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeAlpacaSource()

    collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=tmp_path / "collection",
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )

    expected_close = datetime.fromisoformat("2025-01-02T21:00:00+00:00")
    assert source.bounds
    assert all(end == expected_close for _, _, end in source.bounds)


def test_authority_is_not_published_when_prepublication_replay_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import market_predictor.intraday.datasets.microstructure_history as intraday_microstructure_history

    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()
    original = intraday_microstructure_history._verify_collection_content

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise DataReadinessError("planned replay failure")

    monkeypatch.setattr(
        intraday_microstructure_history,
        "_verify_collection_content",
        reject,
    )
    with pytest.raises(DataReadinessError, match="planned replay failure"):
        collect_intraday_microstructure_history(
            plan_directory=plan,
            output_directory=output,
            source_factory=lambda: source,  # type: ignore[arg-type]
            config=_config(),
            maximum_jobs_this_run=2,
        )
    assert not (output / "_authority.json").exists()

    monkeypatch.setattr(
        intraday_microstructure_history,
        "_verify_collection_content",
        original,
    )
    result = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )
    assert result["status"] == "transport_complete"
    assert (output / "_authority.json").is_file()


def test_event_failure_is_isolated_and_only_failed_job_retries(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource(fail_once={("trades", "AAPL")})

    first = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )

    assert first["status"] == "transport_incomplete"
    assert first["completed_jobs"] == 1
    assert len(first["failed_jobs"]) == 1
    assert source.calls.count(("quotes", "AAPL", None)) == 1

    final = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )

    assert final["status"] == "transport_complete"
    assert source.calls.count(("quotes", "AAPL", None)) == 1
    assert source.calls.count(("trades", "AAPL", None)) == 2
    assert final["attempt_count"] == 3


def test_failed_second_page_resumes_from_verified_page_checkpoint(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource(
        fail_pages_once={("trades", "AAPL", "trade-page-2")}
    )

    first = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )
    assert first["completed_jobs"] == 0

    second = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )
    assert second["completed_jobs"] == 1
    assert source.calls.count(("trades", "AAPL", None)) == 1
    assert source.calls.count(("trades", "AAPL", "trade-page-2")) == 2

    final = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )
    assert final["status"] == "transport_complete"
    assert final["attempt_count"] == 3


def test_collection_rejects_noncanonical_job_path(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()
    collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )
    manifest_path = output / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["path"] = "../outside.manifest.json"
    _write_json(manifest_path, manifest)
    authority = json.loads((output / "_authority.json").read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority["schema"] = COLLECTION_AUTHORITY_SCHEMA
    _write_json(output / "_authority.json", authority)

    with pytest.raises(DataReadinessError, match="job path differs"):
        load_complete_intraday_microstructure_collection(output)


def test_raw_page_content_must_match_recorded_response_hash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()
    manifest = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )
    first_job = json.loads(
        (output / manifest["jobs"][0]["path"]).read_text(encoding="utf-8")
    )
    page = first_job["pages"][0]
    raw_path = output / page["raw_page_path"]
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["trades"]["AAPL"][0]["p"] = 999.0
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)
    changed_page = dict(page)
    changed_page["raw_page_sha256"] = file_sha256(raw_path)
    changed_page["raw_page_bytes"] = raw_path.stat().st_size

    with pytest.raises(DataReadinessError, match="raw page content differs"):
        import market_predictor.intraday.datasets.microstructure_history as intraday_microstructure_history

        intraday_microstructure_history._verify_raw_page(
            output,
            changed_page,
            event_type="trades",
            ticker="AAPL",
            requested_start=datetime.fromisoformat("2025-01-02T14:30:00+00:00"),
            requested_end=datetime.fromisoformat("2025-01-02T21:00:00+00:00"),
        )


def test_failed_attempt_is_bound_by_final_collection_inventory(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource(fail_once={("trades", "AAPL")})
    collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )
    collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=1,
    )
    failed_attempt = next(
        path
        for path in (output / "attempts" / "trades").rglob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"
    )
    payload = json.loads(failed_attempt.read_text(encoding="utf-8"))
    payload["error"] = "tampered"
    _write_json(failed_attempt, payload)

    with pytest.raises(DataReadinessError, match="summary differs"):
        load_complete_intraday_microstructure_collection(output)


def test_required_trade_and_quote_market_identity_fails_closed(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()
    original = source.fetch_quotes_page

    def missing_tape(*args: Any, **kwargs: Any) -> AlpacaQuotesPage:
        page = original(*args, **kwargs)
        row = dict(page.quotes["AAPL"][0])
        row.pop("z")
        raw = {"quotes": {"AAPL": [row]}, "next_page_token": None}
        return AlpacaQuotesPage(
            request_page_token=page.request_page_token,
            next_page_token=None,
            quotes={"AAPL": (row,)},
            response_headers={},
            raw_payload=raw,
        )

    source.fetch_quotes_page = missing_tape  # type: ignore[method-assign]
    result = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )

    assert result["status"] == "transport_incomplete"
    assert result["ready_for_materialization"] is False
    assert any("required market identity" in error for error in result["failed_jobs"].values())


def test_zero_sided_quote_is_preserved_by_raw_transport(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeAlpacaSource()
    original = source.fetch_quotes_page

    def zero_bid(*args: Any, **kwargs: Any) -> AlpacaQuotesPage:
        page = original(*args, **kwargs)
        row = dict(page.quotes["AAPL"][0])
        row["bp"] = 0.0
        row["bs"] = 0
        raw = {"quotes": {"AAPL": [row]}, "next_page_token": None}
        return AlpacaQuotesPage(
            request_page_token=page.request_page_token,
            next_page_token=None,
            quotes={"AAPL": (row,)},
            response_headers={},
            raw_payload=raw,
        )

    source.fetch_quotes_page = zero_bid  # type: ignore[method-assign]
    result = collect_intraday_microstructure_history(
        plan_directory=plan,
        output_directory=tmp_path / "collection",
        source_factory=lambda: source,  # type: ignore[arg-type]
        config=_config(),
        maximum_jobs_this_run=2,
    )

    assert result["status"] == "transport_complete"


def test_memory_configuration_cannot_exceed_four_gib() -> None:
    with pytest.raises(ValueError, match=r"\[1, 4\] GiB"):
        MicrostructureCollectionConfig(maximum_process_memory_gib=4.1)


def test_repository_collection_config_is_complete() -> None:
    config = load_microstructure_collection_config(
        Path("configs/edge_rebuild_intraday_microstructure_history.toml")
    )

    assert config.workers == 1
    assert config.page_size == 10_000
    assert config.maximum_process_memory_gib == 4.0
