from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditReport, audit_canonical_bars
from market_predictor.canonical.store import file_sha256, write_canonical_artifact
from market_predictor.edge_rebuild import swing_daily_combination as module
from market_predictor.v3.errors import DataReadinessError


def test_modeled_population_uses_exact_cutoffs_and_retains_warmup_intervals() -> None:
    memberships = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW", "ENDED", "LATE", "UNKNOWN"],
            "security_id": [
                "sec:active",
                "sec:active",
                "sec:ended",
                "sec:late",
                "sec:unknown",
            ],
            "effective_from_utc": pd.to_datetime(
                [
                    "2018-05-29T04:00:00Z",
                    "2020-01-02T05:00:00Z",
                    "2018-05-29T04:00:00Z",
                    "2026-07-09T00:00:00Z",
                    "2019-07-09T00:00:00Z",
                ],
                utc=True,
            ),
            "effective_to_utc": pd.to_datetime(
                [
                    "2020-01-02T05:00:00Z",
                    None,
                    "2019-07-09T22:01:00Z",
                    None,
                    None,
                ],
                utc=True,
            ),
            "available_at_utc": pd.to_datetime(
                [
                    "2018-05-29T04:00:00Z",
                    "2020-01-02T05:00:00Z",
                    "2018-05-29T04:00:00Z",
                    "2026-07-09T00:00:00Z",
                    "2026-07-09T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    retained, modeled, warmup_only = module._modeled_security_population(
        memberships,
        decision_start=module.POST_START_DATE,
        decision_cutoff=module.CUTOFF_DATE,
    )

    assert modeled == ("sec:active", "sec:ended")
    assert warmup_only == ("sec:late", "sec:unknown")
    assert retained["ticker"].tolist() == ["OLD", "NEW", "ENDED"]


def test_verifier_combines_lineage_and_excludes_unavailable_securities_in_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _authority_inputs(tmp_path, monkeypatch)

    verified = module.verify_combined_swing_inputs(**inputs)

    assert verified.excluded_security_ids == ("sec:bbb", "sec:ccc")
    assert not set(verified.excluded_security_ids).intersection(
        verified.memberships["security_id"].astype(str)
    )
    assert verified.request_payload["excluded_security_fraction"] == 0.05
    assert verified.request_payload["pre_collection"]["authority_sha256"]
    assert verified.request_payload["post_collection"]["manifest_sha256"]
    assert verified.request_payload["membership_authority"]["authority_sha256"]
    assert len(verified.request_payload["security_exclusions"]) == 2
    assert verified.request_payload["coverage_audit_sha256"] == module._json_sha256(
        verified.coverage_audit
    )


def test_verifier_refuses_a_corrupted_post_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _authority_inputs(tmp_path, monkeypatch)
    post = inputs["post_collection_directory"]
    artifact = post / "bars" / "AAA.parquet"
    with artifact.open("ab") as handle:
        handle.write(b"poison")

    with pytest.raises(DataReadinessError, match="partition hash mismatch"):
        module.verify_combined_swing_inputs(**inputs)


def test_verifier_refuses_benchmark_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _authority_inputs(tmp_path, monkeypatch, unavailable="SPY")

    with pytest.raises(DataReadinessError, match="benchmark is unavailable"):
        module.verify_combined_swing_inputs(**inputs)


def test_verifier_coverage_gate_receives_only_modeled_securities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _authority_inputs(tmp_path, monkeypatch)
    memberships = _memberships()
    memberships["effective_to_utc"] = pd.to_datetime(
        memberships["effective_to_utc"], utc=True
    )
    memberships.loc[memberships.index[-1], "effective_to_utc"] = pd.Timestamp(
        "2019-07-09T22:00:00Z"
    )
    monkeypatch.setattr(
        module,
        "require_sp500_membership_authority",
        lambda *_args, **_kwargs: memberships.copy(),
    )
    observed: dict[str, int] = {}

    def preflight(**kwargs: Any) -> module._CoveragePreflight:
        population = kwargs["memberships"]
        observed["security_count"] = int(population["security_id"].nunique())
        audit = _coverage_audit(excluded=0, security_count=39)
        return module._CoveragePreflight(
            audit=audit,
            excluded_security_ids=(),
            exclusion_records=(),
        )

    monkeypatch.setattr(module, "_preflight_exact_coverage", preflight)

    verified = module.verify_combined_swing_inputs(
        **inputs,
        model_decision_start=module.POST_START_DATE,
        model_decision_cutoff=module.CUTOFF_DATE,
    )

    assert observed["security_count"] == 39
    assert verified.request_payload["modeled_security_count"] == 39
    assert verified.request_payload["warmup_only_security_count"] == 1
    assert verified.warmup_only_security_ids == ("sec:36",)


def test_combination_refuses_gap_overlap_and_invalid_ohlcv(tmp_path: Path) -> None:
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(calendar.session_open("2019-07-08")).tz_convert("UTC")
    second = pd.Timestamp(calendar.session_open("2019-07-09")).tz_convert("UTC")
    post = _canonical_bars("AAA", [second])
    post_path = tmp_path / "post.parquet"
    _write_bars(post, post_path)
    post_record = {"resolved_path": str(post_path)}

    with pytest.raises(DataReadinessError, match="membership gaps"):
        module._combine_ticker(
            "AAA",
            pre_records=[],
            post_record=post_record,
            expected_sessions={first.date(), second.date()},
            calendar=calendar,
        )

    retained = module._combine_ticker(
        "AAA",
        pre_records=[],
        post_record=post_record,
        expected_sessions={second.date()},
        abstained_sessions={first.date()},
        calendar=calendar,
    )
    assert set(pd.to_datetime(retained["bar_start_utc"]).dt.date) == {
        second.date()
    }

    with pytest.raises(DataReadinessError, match="session-gap audit is stale"):
        module._combine_ticker(
            "AAA",
            pre_records=[],
            post_record=post_record,
            expected_sessions={first.date()},
            abstained_sessions={second.date()},
            calendar=calendar,
        )

    duplicate = pd.concat([post, post], ignore_index=True)
    duplicate_path = tmp_path / "duplicate.parquet"
    _write_bars(duplicate, duplicate_path, audit=False)
    with pytest.raises(DataReadinessError, match="overlaps"):
        module._combine_ticker(
            "AAA",
            pre_records=[],
            post_record={"resolved_path": str(duplicate_path)},
            expected_sessions={second.date()},
            calendar=calendar,
        )

    invalid = post.assign(high=0.5)
    with pytest.raises(DataReadinessError, match="invalid OHLCV"):
        module._validate_ohlcv(invalid, ticker="AAA")


def test_identity_refuses_same_ticker_for_two_securities_on_one_session() -> None:
    memberships = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "security_id": ["sec:a", "sec:b"],
            "effective_from_utc": [
                pd.Timestamp("2024-01-02T05:00:00Z"),
                pd.Timestamp("2024-01-02T05:00:00Z"),
            ],
            "effective_to_utc": [pd.NaT, pd.NaT],
        }
    )
    with pytest.raises(DataReadinessError, match="multiple securities"):
        module._expected_ticker_sessions(
            "AAA",
            memberships=memberships,
            benchmark_tickers=("SPY",),
            benchmark_start_sessions={"SPY": module.START_DATE},
            all_sessions=(pd.Timestamp("2024-01-02").date(),),
        )


def test_xlc_contiguous_pre_inception_prefix_is_accepted() -> None:
    sessions = tuple(
        pd.Timestamp(value).date()
        for value in xcals.get_calendar("XNYS").sessions_in_range(
            module.START_DATE,
            "2018-07-10",
        )
    )
    record = module._benchmark_coverage_record(
        "XLC",
        observed_sessions=set(sessions[15:]),
        market_sessions=sessions,
    )
    assert record["first_observed_session"] == sessions[15].isoformat()
    assert record["pre_inception_missing_session_count"] == 15
    assert record["missing_session_count"] == 15

    expected = module._expected_ticker_sessions(
        "XLC",
        memberships=pd.DataFrame(),
        benchmark_tickers=("XLC",),
        benchmark_start_sessions={"XLC": sessions[15]},
        all_sessions=sessions,
    )
    assert expected == set(sessions[15:])


def test_sector_benchmark_internal_gap_is_rejected() -> None:
    sessions = tuple(
        pd.Timestamp(value).date()
        for value in xcals.get_calendar("XNYS").sessions_in_range(
            module.START_DATE,
            "2018-07-10",
        )
    )
    observed = set(sessions[15:])
    observed.remove(sessions[20])
    with pytest.raises(DataReadinessError, match="internal or post-inception"):
        module._benchmark_coverage_record(
            "XLC",
            observed_sessions=observed,
            market_sessions=sessions,
        )


def test_preflight_excludes_aet_style_gap_and_rejects_more_than_five_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memberships = _one_session_memberships(20)
    session = pd.Timestamp("2024-01-02")
    session_open = pd.Timestamp(
        xcals.get_calendar("XNYS").session_open(session)
    ).tz_convert("UTC")
    frames = {
        ticker: _canonical_bars(ticker, [session_open])
        for ticker in memberships["ticker"].astype(str)
        if ticker != "AET"
    }
    monkeypatch.setattr(
        module,
        "load_canonical_artifact",
        lambda path, **_kwargs: (frames[Path(path).stem].copy(), {}),
    )
    post_records = {
        ticker: {"resolved_path": str(Path(f"{ticker}.parquet"))}
        for ticker in frames
    }

    coverage = module._preflight_exact_coverage(
        memberships=memberships,
        pre_records=(),
        post_records=post_records,
        benchmark_tickers=(),
        initial_reasons={},
    )
    assert coverage.excluded_security_ids == ("sec:AET",)
    exclusion = coverage.exclusion_records[0]
    assert exclusion["reasons"] == ["membership_session_gap"]
    assert exclusion["missing_session_count"] == 1
    assert exclusion["first_missing_session"] == "2024-01-02"

    too_many_records = dict(post_records)
    too_many_records.pop("T01")
    with pytest.raises(DataReadinessError, match="exclusions exceed 5%"):
        module._preflight_exact_coverage(
            memberships=memberships,
            pre_records=(),
            post_records=too_many_records,
            benchmark_tickers=(),
            initial_reasons={},
        )


def test_preflight_retains_sparse_gaps_as_bound_session_abstentions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range("2024-01-02", "2025-12-31")
    )
    missing = {sessions[200], sessions[201]}
    observed = [
        pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
        for session in sessions
        if session not in missing
    ]
    frame = _canonical_bars("AAA", observed)
    monkeypatch.setattr(
        module,
        "load_canonical_artifact",
        lambda *_args, **_kwargs: (frame.copy(), {}),
    )
    memberships = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security_id": ["sec:AAA"],
            "effective_from_utc": [
                pd.Timestamp(sessions[0]).tz_localize("America/New_York").tz_convert("UTC")
            ],
            "effective_to_utc": [
                (pd.Timestamp(sessions[-1]) + pd.Timedelta(days=1))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
            ],
            "primary_benchmark": ["XLK"],
        }
    )

    coverage = module._preflight_exact_coverage(
        memberships=memberships,
        pre_records=(),
        post_records={"AAA": {"resolved_path": "AAA.parquet"}},
        benchmark_tickers=(),
        initial_reasons={},
    )

    assert coverage.excluded_security_ids == ()
    security = coverage.audit["security_audit"][0]
    assert security["action"] == "retain_with_session_abstentions"
    assert security["missing_session_count"] == 2
    assert security["maximum_contiguous_missing_sessions"] == 2
    gap_audit = coverage.audit["session_gap_audit"]
    assert gap_audit["missing_session_count"] == 2
    assert gap_audit["gaps"][0]["missing_sessions"] == [
        session.isoformat() for session in sorted(missing)
    ]
    abstentions = module._session_abstentions_by_ticker(gap_audit)
    exact_expected = module._expected_ticker_sessions(
        "AAA",
        memberships=memberships,
        benchmark_tickers=(),
        benchmark_start_sessions={},
        all_sessions=sessions,
        session_abstentions=abstentions["AAA"],
    )
    assert exact_expected == set(sessions).difference(missing)


def test_preflight_excludes_long_contiguous_gap_even_when_fraction_is_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range("2020-01-02", "2026-07-08")
    )
    missing = set(sessions[500:506])
    assert len(missing) / len(sessions) <= module.MAXIMUM_SPARSE_MISSING_FRACTION
    frame = _canonical_bars(
        "AAA",
        [
            pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
            for session in sessions
            if session not in missing
        ],
    )
    monkeypatch.setattr(
        module,
        "load_canonical_artifact",
        lambda *_args, **_kwargs: (frame.copy(), {}),
    )
    memberships = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security_id": ["sec:AAA"],
            "effective_from_utc": [
                pd.Timestamp(sessions[0]).tz_localize("America/New_York").tz_convert("UTC")
            ],
            "effective_to_utc": [
                (pd.Timestamp(sessions[-1]) + pd.Timedelta(days=1))
                .tz_localize("America/New_York")
                .tz_convert("UTC")
            ],
            "primary_benchmark": ["XLK"],
        }
    )

    with pytest.raises(DataReadinessError, match="exclusions exceed 5%"):
        module._preflight_exact_coverage(
            memberships=memberships,
            pre_records=(),
            post_records={"AAA": {"resolved_path": "AAA.parquet"}},
            benchmark_tickers=(),
            initial_reasons={},
        )


def test_preflight_refuses_benchmark_session_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memberships = _one_session_memberships(1)
    session_open = pd.Timestamp(
        xcals.get_calendar("XNYS").session_open("2024-01-02")
    ).tz_convert("UTC")
    frames = {
        "AET": _canonical_bars("AET", [session_open]),
        "SPY": _canonical_bars("SPY", [session_open]),
    }
    monkeypatch.setattr(
        module,
        "load_canonical_artifact",
        lambda path, **_kwargs: (frames[Path(path).stem].copy(), {}),
    )
    with pytest.raises(DataReadinessError, match="requires exact full-window"):
        module._preflight_exact_coverage(
            memberships=memberships,
            pre_records=(),
            post_records={
                ticker: {"resolved_path": str(Path(f"{ticker}.parquet"))}
                for ticker in frames
            },
            benchmark_tickers=("SPY",),
            initial_reasons={},
        )


def test_combined_store_is_exact_resumable_and_refuses_corruption(tmp_path: Path) -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = [
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range(module.START_DATE, module.CUTOFF_DATE)
    ]
    pre_sessions = [value for value in sessions if value <= module.PRE_END_DATE]
    post_sessions = [value for value in sessions if value >= module.POST_START_DATE]
    pre_frame = _pre_bars("SPY", pre_sessions)
    pre_path = tmp_path / "source" / "pre.parquet"
    pre_path.parent.mkdir()
    pre_frame.to_parquet(pre_path, index=False)
    post_starts = [
        pd.Timestamp(calendar.session_open(value)).tz_convert("UTC")
        for value in post_sessions
    ]
    post_path = tmp_path / "source" / "post.parquet"
    _write_bars(_canonical_bars("SPY", post_starts), post_path)
    coverage_audit = _coverage_audit(benchmark_tickers=("SPY",))
    verified = module.VerifiedCombinedInputs(
        memberships=pd.DataFrame(
            columns=[
                "ticker",
                "security_id",
                "effective_from_utc",
                "effective_to_utc",
            ]
        ),
        request_payload={
            "schema": module.COMBINED_REQUEST_SCHEMA,
            "pre_collection": {"manifest_sha256": "a", "authority_sha256": "b"},
            "post_collection": {"manifest_sha256": "c"},
            "membership_authority": {"authority_sha256": "d"},
            "excluded_security_ids_sha256": "e",
            "coverage_audit_sha256": module._json_sha256(coverage_audit),
            "session_gap_audit_schema": module.SESSION_GAP_AUDIT_SCHEMA,
            "session_gap_audit_sha256": module._json_sha256(
                coverage_audit["session_gap_audit"]
            ),
            "session_gap_abstention_count": 0,
            "security_exclusions": [],
            "benchmark_coverage": coverage_audit["benchmark_audit"],
        },
        pre_records=(
            {
                "ticker": "SPY",
                "security_id": "benchmark:SPY",
                "role": "benchmark",
                "start_date": module.START_DATE.isoformat(),
                "end_date": module.PRE_END_DATE.isoformat(),
                "rows": len(pre_frame),
                "resolved_path": str(pre_path),
            },
        ),
        post_records={"SPY": {"resolved_path": str(post_path)}},
        excluded_security_ids=(),
        benchmark_tickers=("SPY",),
        coverage_audit=coverage_audit,
    )
    output = tmp_path / "combined"

    first = module.prepare_combined_daily_store(
        verified=verified,
        output_directory=output,
        parent_request_sha256="p" * 64,
        memory_budget_gib=4.0,
        memory_headroom_gib=0.75,
    )
    replay = module.prepare_combined_daily_store(
        verified=verified,
        output_directory=output,
        parent_request_sha256="p" * 64,
        memory_budget_gib=4.0,
        memory_headroom_gib=0.75,
    )
    assert first.manifest["rows"] == len(sessions)
    assert replay.manifest == first.manifest
    assert first.manifest["coverage_audit"]["semantic_sha256"] == module._json_sha256(
        coverage_audit
    )
    assert first.manifest["session_gap_audit"]["semantic_sha256"] == (
        module._json_sha256(coverage_audit["session_gap_audit"])
    )

    changed_audit = {**coverage_audit, "security_count": 1}
    changed_request = {
        **verified.request_payload,
        "coverage_audit_sha256": module._json_sha256(changed_audit),
    }
    with pytest.raises(DataReadinessError, match="resume request differs"):
        module.prepare_combined_daily_store(
            verified=replace(
                verified,
                request_payload=changed_request,
                coverage_audit=changed_audit,
            ),
            output_directory=output,
            parent_request_sha256="p" * 64,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.75,
        )

    session_gap_path = output / "_session_gap_audit.json"
    session_gap_raw = session_gap_path.read_text(encoding="utf-8")
    session_gap_payload = json.loads(session_gap_raw)
    _write_json(session_gap_path, {**session_gap_payload, "gap_count": 1})
    with pytest.raises(DataReadinessError, match="session-gap audit differs"):
        module.prepare_combined_daily_store(
            verified=verified,
            output_directory=output,
            parent_request_sha256="p" * 64,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.75,
        )
    session_gap_path.write_text(session_gap_raw, encoding="utf-8")

    artifact = next(iter(first.artifacts.values()))[0]
    with artifact.open("ab") as handle:
        handle.write(b"poison")
    with pytest.raises(DataReadinessError, match="artifact is invalid"):
        module.prepare_combined_daily_store(
            verified=verified,
            output_directory=output,
            parent_request_sha256="p" * 64,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.75,
        )


def _authority_inputs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unavailable: str = "BBB",
) -> dict[str, Path | None]:
    membership = root / "membership"
    plan = root / "plan"
    pre = root / "pre"
    post = root / "post"
    for directory in (membership, plan, pre, post):
        directory.mkdir()

    memberships = _memberships()
    membership_request = {
        "schema": module.MEMBERSHIP_REQUEST_SCHEMA,
        "start_date": module.START_DATE.isoformat(),
        "cutoff_date": module.CUTOFF_DATE.isoformat(),
        "maximum_security_exclusion_fraction": 0.05,
    }
    _write_json(membership / "_request.json", membership_request)
    for name in ("_authority.json",):
        _write_json(membership / name, {"state": "complete"})
    _write_json(
        membership / "_manifest.json",
        {
            "membership_artifact": {"sha256": "m" * 64},
            "parent_lineage": {"raw": "r" * 64},
            "universe_sha256": "u" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "require_sp500_membership_authority",
        lambda *_args, **_kwargs: memberships.copy(),
    )

    pre_bars = pre / "units" / "aaa" / "bars.parquet"
    pre_bars.parent.mkdir(parents=True)
    pre_bars.write_bytes(b"verified by exact collector")
    pre_manifest: dict[str, Any] = {
        "unit_set_sha256": "v" * 64,
        "universe_sha256": "u" * 64,
        "unavailable_security_count": 1,
        "unavailable_units": [
            {
                "security_id": "sec:ccc",
                "ticker": "CCC",
                "role": "stock",
                "allowed": True,
            }
        ],
        "unit_artifacts": [
            {
                "status": "observed",
                "security_id": "sec:aaa",
                "ticker": "AAA",
                "role": "stock",
                "start_date": module.START_DATE.isoformat(),
                "end_date": module.PRE_END_DATE.isoformat(),
                "bars_path": "units/aaa/bars.parquet",
                "bars_sha256": file_sha256(pre_bars),
                "rows": 1,
            }
        ],
    }
    pre_request = {"request_sha256": "p" * 64}
    _write_json(pre / "_request.json", pre_request)
    _write_json(pre / "_manifest.json", pre_manifest)
    _write_json(pre / "_authority.json", {"unit_set_sha256": "v" * 64})
    monkeypatch.setattr(
        module,
        "load_complete_swing_history_collection",
        lambda *_args, **_kwargs: pre_manifest,
    )
    exclusion_records = (
        {
            "security_id": "sec:bbb",
            "missing_session_count": 1,
            "first_missing_session": module.POST_START_DATE.isoformat(),
            "reasons": ["post_collection_unavailable"],
            "action": "exclude_security",
        },
        {
            "security_id": "sec:ccc",
            "missing_session_count": 1,
            "first_missing_session": module.START_DATE.isoformat(),
            "reasons": ["pre_collection_unavailable"],
            "action": "exclude_security",
        },
    )
    coverage_audit = _coverage_audit(
        excluded=2,
        security_count=40,
    )
    monkeypatch.setattr(
        module,
        "_preflight_exact_coverage",
        lambda **_kwargs: module._CoveragePreflight(
            audit=coverage_audit,
            excluded_security_ids=("sec:bbb", "sec:ccc"),
            exclusion_records=exclusion_records,
        ),
    )

    post_hashes = _post_collection(post, unavailable=unavailable)
    membership_hashes = module._membership_hashes(membership)
    _write_json(
        plan / "_request.json",
        {"membership_authority": membership_hashes, **post_hashes},
    )
    _write_json(plan / "_manifest.json", {"status": "complete"})
    _write_json(plan / "_authority.json", {"state": "complete"})
    placeholders = {
        "raw_archive_directory": root / "raw",
        "event_directory": root / "events",
        "transition_directory": root / "transitions",
        "reviewed_transitions_path": root / "review.csv",
        "anchor_path": root / "anchor.csv",
    }
    return {
        "pre_plan_directory": plan,
        "pre_collection_directory": pre,
        "post_collection_directory": post,
        "membership_directory": membership,
        **placeholders,
        "security_exclusions_path": None,
    }


def _post_collection(directory: Path, *, unavailable: str) -> dict[str, str]:
    bars_path = directory / "bars" / "AAA.parquet"
    session_open = pd.Timestamp(
        xcals.get_calendar("XNYS").session_open(module.POST_START_DATE)
    ).tz_convert("UTC")
    bars = _canonical_bars("AAA", [session_open])
    canonical = _write_bars(bars, bars_path)
    request_payload = {
        "schema": module.POST_REQUEST_SCHEMA,
        "source": "alpaca",
        "timeframe": "1d",
        "price_feed": "sip",
        "adjustment": "all",
        "start_date": module.POST_START_DATE.isoformat(),
        "end_date": module.CUTOFF_DATE.isoformat(),
    }
    identity = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(directory / "_request.json", {**request_payload, "request_sha256": identity})
    ledger = pd.DataFrame(
        {
            "collection_id": ["alpaca-AAA-test", f"alpaca-{unavailable}-test"],
            "ticker": ["AAA", unavailable],
            "source_family": ["alpaca_daily_bars", "alpaca_daily_bars"],
            "requested_start_utc": [
                pd.Timestamp("2019-07-09T00:00:00Z"),
                pd.Timestamp("2019-07-09T00:00:00Z"),
            ],
            "requested_end_utc": [
                pd.Timestamp("2026-07-08T23:59:59Z"),
                pd.Timestamp("2026-07-08T23:59:59Z"),
            ],
            "status": ["observed", "observed_empty"],
            "row_count": [1, 0],
        }
    )
    ledger_path = directory / "_source_collections.parquet"
    ledger.to_parquet(ledger_path, index=False)
    artifact = {
        "ticker": "AAA",
        "path": "bars/AAA.parquet",
        "manifest_path": "bars/AAA.parquet.manifest.json",
        "sha256": str(canonical["artifact_sha256"]),
        "rows": 1,
        "first_bar_start_utc": session_open.isoformat(),
        "last_bar_start_utc": session_open.isoformat(),
        "price_feed": "sip",
        "adjustment": "all",
    }
    terminal = {
        "schema": module.POST_MANIFEST_SCHEMA,
        "status": "complete_with_gaps",
        "request_sha256": identity,
        "requested_symbols": 2,
        "observed_symbols": 1,
        "unavailable_symbols": [unavailable],
        "failed_symbols": {},
        "skipped_symbols": 0,
        "source_collections_sha256": file_sha256(ledger_path),
    }
    _write_json(directory / "_status.json", terminal)
    _write_json(
        directory / "_manifest.json",
        {**terminal, "artifact_count": 1, "total_rows": 1, "artifacts": [artifact]},
    )
    return {
        "daily_request_file_sha256": file_sha256(directory / "_request.json"),
        "daily_status_sha256": file_sha256(directory / "_status.json"),
        "daily_manifest_sha256": file_sha256(directory / "_manifest.json"),
        "daily_request_identity_sha256": identity,
    }


def _memberships() -> pd.DataFrame:
    security_ids = ["sec:aaa", "sec:bbb", "sec:ccc"] + [
        f"sec:{index:02d}" for index in range(37)
    ]
    tickers = ["AAA", "BBB", "CCC"] + [f"T{index:02d}" for index in range(37)]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "security_id": security_ids,
            "effective_from_utc": pd.Timestamp("2018-05-29T04:00:00Z"),
            "effective_to_utc": pd.NaT,
            "available_at_utc": pd.Timestamp("2018-05-29T04:00:00Z"),
            "primary_benchmark": "XLK",
        }
    )


def _one_session_memberships(count: int) -> pd.DataFrame:
    tickers = ["AET", *[f"T{index:02d}" for index in range(1, count)]]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "security_id": [f"sec:{ticker}" for ticker in tickers],
            "effective_from_utc": pd.Timestamp("2024-01-02T05:00:00Z"),
            "effective_to_utc": pd.Timestamp("2024-01-03T05:00:00Z"),
            "available_at_utc": pd.Timestamp("2024-01-02T05:00:00Z"),
            "primary_benchmark": "XLK",
        }
    )


def _canonical_bars(ticker: str, starts: list[pd.Timestamp]) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    closes = [
        pd.Timestamp(calendar.session_close(value.date())).tz_convert("UTC")
        for value in starts
    ]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1d",
            "bar_start_utc": starts,
            "bar_end_utc": closes,
            "available_at_utc": [value + pd.Timedelta(minutes=15) for value in closes],
            "ingested_at_utc": pd.Timestamp("2026-07-09T00:00:00Z"),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
            "availability_policy": "market_interval_close",
            "schema_version": "market_data.v1",
        }
    )


def _pre_bars(
    ticker: str,
    sessions: list[object],
) -> pd.DataFrame:
    starts = [
        pd.Timestamp(value).tz_localize("America/New_York").tz_convert("UTC")
        for value in sessions
    ]
    return pd.DataFrame(
        {
            "security_id": f"benchmark:{ticker}",
            "ticker": ticker,
            "role": "benchmark",
            "bar_start_utc": starts,
            "session_date": [str(value) for value in sessions],
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100,
            "source": "alpaca",
            "timeframe": "1Day",
            "price_feed": "sip",
            "adjustment": "all",
            "ingested_at_utc": pd.Timestamp("2026-07-09T00:00:00Z"),
        }
    )


def _write_bars(
    bars: pd.DataFrame,
    path: Path,
    *,
    audit: bool = True,
) -> dict[str, object]:
    checks = audit_canonical_bars(bars, require_sip=True)
    if not audit:
        checks = tuple(check.model_copy(update={"status": "pass", "failures": 0}) for check in checks)
    return write_canonical_artifact(
        bars,
        path,
        artifact_type="bars",
        audit=CanonicalAuditReport(checks=checks),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _coverage_audit(
    *,
    excluded: int = 0,
    security_count: int = 0,
    benchmark_tickers: tuple[str, ...] = (),
) -> dict[str, object]:
    session_gap_audit = {
        "schema": module.SESSION_GAP_AUDIT_SCHEMA,
        "classification_policy": {
            "maximum_missing_fraction": module.MAXIMUM_SPARSE_MISSING_FRACTION,
            "maximum_contiguous_missing_sessions": (
                module.MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
            ),
            "unavailable_source_action": "exclude_security",
            "substantial_gap_action": "exclude_security",
            "sparse_gap_action": "abstain",
            "imputation": "prohibited",
            "downstream_rule": (
                "abstain from every feature or label row whose required "
                "membership-session window intersects a missing session"
            ),
        },
        "gap_count": 0,
        "security_count": 0,
        "missing_session_count": 0,
        "gaps": [],
    }
    return {
        "schema": module.COVERAGE_AUDIT_SCHEMA,
        "security_count": security_count,
        "excluded_security_count": excluded,
        "security_audit": [],
        "session_gap_audit": session_gap_audit,
        "benchmark_audit": [
            {
                "ticker": ticker,
                "coverage_policy": "exact_full_window",
                "requested_first_session": module.START_DATE.isoformat(),
                "requested_last_session": module.CUTOFF_DATE.isoformat(),
                "first_observed_session": module.START_DATE.isoformat(),
                "expected_session_count": 1,
                "observed_session_count": 1,
                "missing_session_count": 0,
                "pre_inception_missing_session_count": 0,
                "first_missing_session": None,
                "action": "retain",
            }
            for ticker in benchmark_tickers
        ],
    }
