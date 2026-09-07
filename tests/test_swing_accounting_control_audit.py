from __future__ import annotations

import json
import shutil
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import sequence_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.research import swing_accounting_control as audit

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/swing_accounting_audit.toml"


def _policy() -> audit._Policy:
    return audit._Policy.model_validate(tomllib.loads(POLICY.read_text(encoding="utf-8")))


def _features() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "decision_id": f"{day}:{i}", "decision_group_id": day, "security_id": str(i),
            "sector": f"sector{i}", "primary_benchmark": "XLB", "session_date_et": day,
            "decision_time_utc": f"{day}T20:15:00Z", "feature_available_at_utc": f"{day}T20:00:00Z",
            "feature_eligible": True, "cross_section_eligible": True, "return_20d_xs_rank": float(i) / 5,
        }
        for day in ("2019-07-09", "2019-07-10") for i in range(5)
    ])


def _outcomes(features: pd.DataFrame, valuation: tuple[str, ...]) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    rows = []
    for row in features.to_dict(orient="records"):
        entry = calendar.session_offset(row["session_date_et"], 1)
        exit_session = calendar.session_offset(row["session_date_et"], 10)
        record = dict.fromkeys(audit._OUTCOMES, 0.01)
        record.update({
            "decision_id": row["decision_id"], "entry_time_utc": calendar.session_open(entry),
            "exit_time_utc": calendar.session_close(exit_session), "horizon_sessions": 10,
            "entry_session_date_et": str(entry.date()), "exit_session_date_et": str(exit_session.date()),
            "label_available_at_utc": calendar.session_close(exit_session) + pd.Timedelta(minutes=15),
            "barrier_label_available_at_utc": calendar.session_close(entry),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def test_selection_cannot_consume_future_label_poison() -> None:
    strategy = load_strategy_contract(ROOT / _policy().strategy_contract)
    frame = _features()
    expected = audit._select_features(frame, strategy)
    frame["label_eligible"] = False
    frame["managed_path_eligible"] = False
    frame["rank_label"] = -1
    frame["barrier_net_return"] = float("nan")
    actual = audit._select_features(frame, strategy)
    pd.testing.assert_frame_equal(actual, expected)
    assert set(actual.columns).isdisjoint({"rank_label", "label_eligible", "managed_path_eligible", "barrier_net_return"})


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "late", "wrong_open", "wrong_close"])
def test_selected_outcome_failure_never_becomes_a_smaller_cohort(problem: str) -> None:
    selected = _features()
    valuation = audit.swing_valuation_sessions(("2019-07-09", "2019-07-10"), 10)
    outcomes = _outcomes(selected, valuation)
    if problem == "missing":
        outcomes = outcomes.iloc[1:]
    elif problem == "duplicate":
        outcomes = pd.concat([outcomes, outcomes.iloc[:1]], ignore_index=True)
    elif problem == "nonfinite":
        outcomes.loc[0, "future_net_return_10d"] = float("nan")
    elif problem == "late":
        outcomes["label_available_at_utc"] = pd.Timestamp("2025-07-01", tz="UTC")
    else:
        field = "entry_time_utc" if problem == "wrong_open" else "exit_time_utc"
        outcomes[field] = pd.to_datetime(outcomes[field], utc=True) + pd.Timedelta(minutes=1)
    with pytest.raises(DataReadinessError):
        audit._join_outcomes(selected, outcomes, valuation)
    assert len(selected) == 10


def test_nonfinite_error_counts_fields_rows_and_at_most_five_ids() -> None:
    selected = _features()
    valuation = audit.swing_valuation_sessions(("2019-07-09", "2019-07-10"), 10)
    outcomes = _outcomes(selected, valuation)
    outcomes.loc[:5, "future_net_return_10d"] = float("nan")
    outcomes.loc[:1, "future_excess_return_10d_vs_spy"] = float("inf")
    with pytest.raises(audit._IncompleteFixedOutcomes) as caught:
        audit._join_outcomes(selected, outcomes, valuation)
    details = caught.value.details
    assert details == {
        "code": "selected_fixed_horizon_nonfinite", "selected_row_count": 10,
        "affected_row_count": 6, "nonfinite_value_count": 8,
        "field_counts": {"future_net_return_10d": 6, "future_excess_return_10d_vs_spy": 2},
        "first_decision_ids": selected["decision_id"].head(5).tolist(),
    }
    assert json.loads(str(caught.value).split(": ", 1)[1]) == details


@pytest.mark.parametrize("name", ["decision_id", "decision_group_id", "security_id", "sector", "primary_benchmark", "session_date_et"])
def test_null_candidate_identity_fails_before_selection(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _features()
    frame.loc[0, name] = None
    monkeypatch.setattr(audit, "select_constrained_swing_portfolio", lambda *a, **kw: pytest.fail("selector reached"))
    with pytest.raises(DataReadinessError, match="identity"):
        audit._select_features(frame, load_strategy_contract(ROOT / _policy().strategy_contract))


def test_fixed_calendar_covers_idle_days_and_tail() -> None:
    decisions, valuation = audit._calendars(ROOT, _policy())
    assert len(decisions) == 1221
    assert len(valuation) == 1230
    assert decisions[0] == "2019-07-09"
    assert valuation[0] == "2019-07-10"
    assert valuation[-1] == "2024-05-28"
    assert valuation == audit.swing_valuation_sessions(decisions, 10)


def test_tampered_file_rejected_before_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "part.parquet"
    path.write_bytes(b"tampered, not parquet")
    monkeypatch.setattr(audit, "_read_projection", lambda *a, **kw: pytest.fail("parquet must not open"))
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        checked = audit._checked_file(tmp_path, path.name, "0" * 64, {})
        audit._read_projection(checked, audit._FEATURES, ("2019-07-09",))


def test_source_mutation_after_projection_blocks_receipt_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("original", encoding="utf-8")
    bound = {str(source): audit.file_sha256(source)}
    source.write_text("changed after projection", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="input changed"):
        audit._bound_report(tmp_path, _policy(), bound, _features(), ("2019-07-09",), ("2019-07-10",))


@pytest.fixture
def control_root(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    for path in ("temporal_policy", "training_policy", "strategy_contract", "research_contract"):
        relative = getattr(_policy(), path)
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    shutil.copyfile(POLICY, tmp_path / "configs/swing_accounting_audit.toml")
    return tmp_path


def test_tail_guard_runs_before_any_parquet_load(control_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "swing_valuation_sessions", lambda *a: ("2025-07-01",))
    monkeypatch.setattr(audit, "_sources", lambda *a: pytest.fail("source loading preceded tail guard"))
    monkeypatch.setattr(audit, "_read_projection", lambda *a, **kw: pytest.fail("parquet must not open"))
    with pytest.raises(DataReadinessError, match="maturation tail"):
        audit._run(control_root, control_root / "configs/swing_accounting_audit.toml", control_root / "report")


def test_lease_precedes_even_policy_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def refused(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["runtime_dir"] == tmp_path / "data/runtime"
        raise RuntimeError("lease busy")
        yield

    monkeypatch.setattr(audit, "heavy_job_runtime_dir", lambda: Path("data/runtime"))
    monkeypatch.setattr(audit, "heavy_job_lease", refused)
    with pytest.raises(RuntimeError, match="lease busy"):
        audit.run_swing_accounting_control_audit(tmp_path, tmp_path / "absent.toml", tmp_path / "report")


@pytest.mark.parametrize("blocked", [False, True])
def test_control_two_phase_projection_and_immutable_report(
    control_root: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool,
) -> None:
    decisions = ("2019-07-09", "2019-07-10")
    valuation = audit.swing_valuation_sessions(decisions, 10)
    features = _features()
    outcomes = _outcomes(features, valuation)
    if blocked:
        outcomes.loc[0, "future_net_return_10d"] = float("nan")
    phases: list[str] = []
    panel = {"files_by_profile": {"technical_market": [{
        "partition_month": "2019-07", "path": "part.parquet", "sha256": "0" * 64,
    }]}}
    combined = {"artifacts": [
        {"ticker": ticker, "path": ticker + ".parquet", "sha256": "0" * 64, "canonical_manifest_sha256": "1" * 64}
        for ticker in ("SPY", "QQQ", "XLB")
    ]}
    monkeypatch.setattr(audit, "_calendars", lambda *a: (decisions, valuation))
    monkeypatch.setattr(audit, "_sources", lambda *a: (panel, combined))
    monkeypatch.setattr(audit, "_checked_file", lambda root, relative, *a: root / relative)
    monkeypatch.setattr(audit, "_metadata", lambda *a: {"artifact_sha256": "0" * 64})
    monkeypatch.setattr(audit, "_guard", lambda *a, **kw: None)

    def projection(path: Path, columns: list[str], sessions: tuple[str, ...], ids: list[str] | None = None) -> pd.DataFrame:
        if columns == audit._FEATURES:
            phases.append("features")
            assert ids is None and sessions == decisions
            return features.copy()
        if columns == audit._OUTCOMES:
            assert phases == ["features"]
            assert set(ids or []) == set(features["decision_id"])
            phases.append("selected_outcomes")
            return outcomes.copy()
        assert not blocked, "blocked control must not read benchmark payloads"
        assert sessions == valuation
        return pd.DataFrame({
            "ticker": path.stem, "bar_start_utc": pd.to_datetime(list(valuation), utc=True) + pd.Timedelta(hours=14),
            "open": 100.0, "close": 100.0, "price_feed": "sip", "adjustment": "all",
        })

    def accounting(selected: pd.DataFrame, bars: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        assert not blocked, "blocked control must not run accounting"
        assert len(selected) == 10
        assert kwargs["session_calendar"] == decisions
        assert set(bars["session_date_et"]) == set(valuation)
        return {"eligible": False, "price_basis_status": "price_basis_pending", "base_ledger": {"final_cash": 1.0}}

    monkeypatch.setattr(audit, "_read_projection", projection)
    monkeypatch.setattr(audit, "evaluate_funded_swing_accounting", accounting)
    report = audit._run(control_root, control_root / "configs/swing_accounting_audit.toml", control_root / "report")
    assert report["eligible"] is False
    assert report["validation_or_test_outcomes_read"] is False
    assert report["selected_decision_ids_sha256"] == sequence_sha256(report["selected_decision_ids"])
    assert report["selected_decision_ids"] == features["decision_id"].tolist()
    assert report["decision_sessions"] == list(decisions)
    assert report["valuation_sessions"] == list(valuation)
    assert report["bound_files"]
    if blocked:
        assert report["status"] == "blocked"
        assert report["accounting_status"] == "not_run"
        assert report["benchmark_payloads_read"] is False
        assert "accounting" not in report
        assert report["errors"][0]["affected_row_count"] == 1
        assert report["errors"][0]["field_counts"] == {"future_net_return_10d": 1}
        assert report["errors"][0]["first_decision_ids"] == [features["decision_id"].iloc[0]]
    else:
        assert report["accounting"]["base_ledger"]["final_cash"] == 1.0
    assert json.loads((control_root / "report/_manifest.json").read_text()) == report
    with pytest.raises(DataReadinessError, match="immutable"):
        audit._publish(control_root / "report", report)


def test_policy_cannot_change_dates_hashes_or_admit_price_basis() -> None:
    for changes in ({"initial_fit_end": "2026-06-30"}, {"panel_manifest_sha256": "0" * 64}, {"passed": True}):
        with pytest.raises(ValueError):
            audit._Policy.model_validate({**_policy().model_dump(), **changes})


def test_arrow_projection_filters_before_outcome_admission(tmp_path: Path) -> None:
    frame = _features()
    held = frame.iloc[:1].copy()
    held["session_date_et"] = "2025-07-01"
    held["decision_id"] = "held"
    frame = pd.concat([frame, held], ignore_index=True)
    frame["future_net_return_10d"] = 999.0
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path, index=False)
    features = audit._read_projection(path, audit._FEATURES, ("2019-07-09", "2019-07-10"))
    assert "future_net_return_10d" not in features
    assert "held" not in set(features["decision_id"])
    outcome = audit._read_projection(
        path, ["decision_id", "future_net_return_10d"], ("2019-07-09", "2019-07-10"), ["2019-07-09:0"],
    )
    assert outcome["decision_id"].tolist() == ["2019-07-09:0"]


def test_benchmark_projection_excludes_unrequested_calendar(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "bar_start_utc": pd.to_datetime(["2019-07-08T13:30:00Z", "2019-07-10T13:30:00Z", "2025-07-01T13:30:00Z"]),
        "ticker": "SPY", "open": [99.0, 100.0, 999.0], "close": [99.0, 100.0, 999.0],
    })
    path = tmp_path / "spy.parquet"
    frame.to_parquet(path, index=False)
    result = audit._read_projection(path, frame.columns.tolist(), ("2019-07-10",))
    assert result["close"].tolist() == [100.0]
