from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.contracts.holding_accounting import HoldingSpecification
from market_predictor.swing.datasets.holding_materialization import materialize_fixed_holdings
from market_predictor.swing.evaluation.holding_accounting import replay_holding

CALENDAR = xcals.get_calendar("XNYS")
SESSIONS = CALENDAR.sessions_in_range("2024-01-03", "2024-01-17")


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_sha256(path)


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "raw").mkdir()
    policy_sha = _write(tmp_path / "policy.json", {"scope": "test_fixture_interpretation_only"})
    ownership_sha = _write(tmp_path / "ownership.json", {"class": "common:A", "scope": "test_fixture_only"})
    source = Path(__file__).resolve().parents[1] / "configs/swing_research.toml"
    (tmp_path / "research.toml").write_bytes(source.read_bytes())
    frame = pd.DataFrame({"security_id": ["common:A"] * 10, "ticker": ["A"] * 10,
        "session_date": [str(day.date()) for day in SESSIONS],
        "bar_start_utc": [pd.Timestamp(day.date(), tz="America/New_York").tz_convert("UTC") for day in SESSIONS],
        "open": [100.0] * 10, "high": [120.0] * 10, "low": [90.0] * 10,
        "close": [101.0 + i for i in range(10)], "volume": [1000] * 10,
        "source": ["alpaca"] * 10, "timeframe": ["1Day"] * 10, "price_feed": ["sip"] * 10,
        "adjustment": ["raw"] * 10, "ingested_at_utc": pd.to_datetime(["2026-09-09T12:00:00Z"] * 10)})
    frame.to_parquet(tmp_path / "raw/bars.parquet", index=False)
    digest = file_sha256(tmp_path / "raw/bars.parquet")
    artifact = {"unit_id": "unit-A", "security_id": "common:A", "ticker": "A", "provider_symbol": "A",
        "bars_path": "bars.parquet", "bars_sha256": digest}
    selection = {"schema": "market_predictor.swing_symbol_corrected_sources", "segments": [{
        "archive": str(tmp_path / "raw"), "artifact": artifact, "first_session": "2024-01-03", "last_session": "2024-01-17"}]}
    evidence = {"reference": "ownership.json", "artifact_sha256": ownership_sha, "record_locator": "class",
        "interpretation_policy_sha256": policy_sha, "retrieved_at": "2026-09-09T12:00:00Z", "available_at": None}
    batch = {"schema": "market_predictor.fixed_holding_materialization",
        "source_selection": {"path": "selection.json", "sha256": _write(tmp_path / "selection.json", selection)},
        "interpretation_policy": {"path": "policy.json", "sha256": policy_sha},
        "research_contract": {"path": "research.toml", "sha256": file_sha256(tmp_path / "research.toml")},
        "requests": [{"decision_id": "d1", "decision_session": "2024-01-02", "security_id": "common:A",
            "sector": "technology", "initial_position_id": "shares", "bindings": [{
                "position_id": "shares", "security_id": "common:A", "unit_id": "unit-A", "provider_symbol": "A",
                "bars_sha256": digest, "first_session": "2024-01-03", "last_session": "2024-01-17",
                "ownership_evidence": [evidence]}]}]}
    return {"root": tmp_path, "selection": selection, "batch": batch, "evidence": evidence, "frame": frame}


def _run(inputs: dict[str, Any], *, expected_output: str | None = None) -> dict[str, Any]:
    root = inputs["root"]
    pin = _write(root / "request.json", inputs["batch"])

    @contextmanager
    def selected() -> Any:
        with heavy_job_lease("fixture-holding", runtime_dir=root / "runtime"):
            yield inputs["selection"]

    return materialize_fixed_holdings(root=root, request_path=root / "request.json", request_sha256=pin,
        output=root / "compiled.json", selection_context=selected, expected_output_sha256=expected_output)


def _change_bars(inputs: dict[str, Any], frame: pd.DataFrame) -> None:
    root = inputs["root"]
    frame.to_parquet(root / "raw/bars.parquet", index=False)
    digest = file_sha256(root / "raw/bars.parquet")
    inputs["selection"]["segments"][0]["artifact"]["bars_sha256"] = digest
    inputs["batch"]["requests"][0]["bindings"][0]["bars_sha256"] = digest
    inputs["batch"]["source_selection"]["sha256"] = _write(root / "selection.json", inputs["selection"])


def test_raw_specification_roundtrip_replays_kernel_and_never_fabricates_cash(inputs: dict[str, Any]) -> None:
    row = _run(inputs)["rows"][0]
    spec = HoldingSpecification.model_validate_json(json.dumps(row["specification"]))
    assert replay_holding(spec).model_dump(mode="json") == row["diagnostic_replay"]
    assert spec.initial_entry_timestamp == CALENDAR.session_open(SESSIONS[0]).to_pydatetime()
    assert len(spec.session_end_timestamps) == 10
    assert spec.session_end_timestamps[-1] == CALENDAR.session_close(SESSIONS[-1]).to_pydatetime()
    final = row["diagnostic_replay"]["snapshots"][-1]
    assert final["net_return"] == pytest.approx(0.098)
    assert final["available_cash"] == 0 and not final["fully_settled"]
    assert final["label_available_at"] is None
    assert spec.events == ()
    assert {gap["code"] for gap in row["source_gaps"]} == {
        "independent_source_admission_required", "action_coverage_unproven"}
    assert row["reportable_net_return"] is None and not row["label_eligible"]


@pytest.mark.parametrize("change", ["missing", "invalid", "ownership"])
def test_entry_cannot_be_fabricated(inputs: dict[str, Any], change: str) -> None:
    if change == "missing":
        _change_bars(inputs, inputs["frame"].iloc[1:])
    elif change == "invalid":
        inputs["frame"].loc[0, "volume"] = 0
        _change_bars(inputs, inputs["frame"])
    else:
        inputs["batch"]["requests"][0]["bindings"] = []
    row = _run(inputs)["rows"][0]
    assert row["specification"] is None and row["diagnostic_replay"] is None
    assert row["materialization_gaps"]


def test_missing_later_session_never_shifts_horizon(inputs: dict[str, Any]) -> None:
    _change_bars(inputs, inputs["frame"].iloc[:-1])
    row = _run(inputs)["rows"][0]
    assert row["specification"]["marks"][-1]["reason"] == "observation_missing"
    assert row["diagnostic_replay"]["snapshots"][-1]["net_return"] is None


@pytest.mark.parametrize("change", ["artifact", "selection", "issuer", "overlap", "unit", "feed", "adjustment", "clock"])
def test_poisoned_source_fails(inputs: dict[str, Any], change: str) -> None:
    request = inputs["batch"]["requests"][0]
    if change == "artifact":
        (inputs["root"] / "raw/bars.parquet").write_bytes(b"tampered")
    elif change == "selection":
        (inputs["root"] / "selection.json").write_text("{}")
    elif change == "issuer":
        request["bindings"][0]["security_id"] = "common:WRONG"
    elif change == "overlap":
        request["bindings"].append(copy.deepcopy(request["bindings"][0]))
    elif change == "unit":
        request["bindings"][0]["unit_id"] = "wrong-parent-unit"
    else:
        column, value = {"feed": ("price_feed", "iex"), "adjustment": ("adjustment", "all"),
            "clock": ("bar_start_utc", pd.Timestamp("2024-01-03T12:00:00Z"))}[change]
        inputs["frame"].loc[0, column] = value
        _change_bars(inputs, inputs["frame"])
    with pytest.raises(DataReadinessError):
        _run(inputs)


def test_future_scope_rejected_before_raw_decode(inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs["batch"]["requests"][0]["decision_session"] = "2024-05-14"
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("must not decode future numeric data"))
    with pytest.raises(DataReadinessError, match="initial-fit"):
        _run(inputs)


def test_publication_is_immutable_and_requires_independent_pin(inputs: dict[str, Any]) -> None:
    result = _run(inputs)
    pin = file_sha256(inputs["root"] / "compiled.json")
    assert _run(inputs, expected_output=pin) == result
    with pytest.raises(DataReadinessError):
        _run(inputs)


def test_lease_conflict_prevents_loading(inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("competing process decoded raw data"))
    with heavy_job_lease("occupied", runtime_dir=inputs["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(inputs)


def test_hash_valid_unrelated_coverage_never_admits_returns(inputs: dict[str, Any]) -> None:
    inputs["batch"]["requests"][0]["action_coverage"] = [{"security_id": "common:A",
        "first_session": "2024-01-03", "last_session": "2024-01-17", "evidence": [inputs["evidence"]]}]
    row = _run(inputs)["rows"][0]
    assert row["diagnostic_replay"]["snapshots"][-1]["net_return"] is not None
    assert row["reportable_net_return"] is None
    assert row["source_gaps"] == [{"code": "independent_source_admission_required", "reference_id": "d1"}]


@pytest.mark.parametrize("gap", ["ownership", "missing", "invalid"])
def test_mid_holding_gap_survives_later_price_recovery(inputs: dict[str, Any], gap: str) -> None:
    if gap == "ownership":
        bindings = inputs["batch"]["requests"][0]["bindings"]
        bindings.append(copy.deepcopy(bindings[0]))
        bindings[0]["last_session"] = str(SESSIONS[3].date())
        bindings[1]["first_session"] = str(SESSIONS[5].date())
    elif gap == "missing":
        _change_bars(inputs, inputs["frame"].drop(4))
    else:
        inputs["frame"].loc[4, "volume"] = 0
        _change_bars(inputs, inputs["frame"])
    row = _run(inputs)["rows"][0]
    expected = {"ownership": "ownership_unproven", "missing": "observation_missing", "invalid": "observation_invalid"}[gap]
    assert any(item["code"] == expected for item in row["materialization_gaps"])
    assert row["diagnostic_replay"]["snapshots"][-1]["net_return"] is not None
    assert row["reportable_net_return"] is None


def _action(inputs: dict[str, Any], *, shares: bool = False) -> dict[str, Any]:
    return {"kind": "corporate_action", "event_id": "corporate-change",
        "effective_at": CALENDAR.session_close(SESSIONS[4]).isoformat(), "order": 0,
        "evidence": [inputs["evidence"]], "owned_position_id": "shares", "owned_security_id": "common:A",
        "treatment": "replace" if shares else "distribution", "fractional_treatment": "proportional",
        "legs": [{"position_id": "new-position", "kind": "tradable_shares" if shares else "unpaid_proceeds",
            "security_id": "common:B" if shares else "common:A", "units_per_owned_unit": 1.0,
            "currency": "USD", "face_value_per_unit": 5.0}]}


def test_cash_claim_face_is_not_a_mark_or_spendable_cash(inputs: dict[str, Any]) -> None:
    inputs["batch"]["requests"][0]["events"] = [_action(inputs)]
    row = _run(inputs)["rows"][0]
    final = row["diagnostic_replay"]["snapshots"][-1]
    assert final["available_cash"] == 0
    assert final["unpaid_proceeds_value"] is None and final["net_return"] is None
    assert len(row["specification"]["events"]) == 1


def test_successor_missing_binding_only_matters_after_creation(inputs: dict[str, Any]) -> None:
    inputs["batch"]["requests"][0]["events"] = [_action(inputs, shares=True)]
    row = _run(inputs)["rows"][0]
    gaps = row["materialization_gaps"]
    assert len(gaps) == 6
    assert all(item["reference_id"].startswith("new-position:") for item in gaps)
    assert all(snapshot["net_return"] is not None for snapshot in row["diagnostic_replay"]["snapshots"][:4])
    assert row["diagnostic_replay"]["snapshots"][-1]["net_return"] is None


def test_successor_uses_own_dated_source_and_shares(inputs: dict[str, Any]) -> None:
    request = inputs["batch"]["requests"][0]
    request["events"] = [_action(inputs, shares=True)]
    frame = inputs["frame"].copy().assign(security_id="common:B", ticker="B", close=105.0)
    root = inputs["root"]
    frame.to_parquet(root / "raw/successor.parquet", index=False)
    segment = copy.deepcopy(inputs["selection"]["segments"][0])
    segment["artifact"].update(unit_id="unit-B", security_id="common:B", ticker="B", provider_symbol="B",
        bars_path="successor.parquet", bars_sha256=file_sha256(root / "raw/successor.parquet"))
    inputs["selection"]["segments"].append(segment)
    inputs["batch"]["source_selection"]["sha256"] = _write(root / "selection.json", inputs["selection"])
    binding = copy.deepcopy(request["bindings"][0])
    binding.update(position_id="new-position", security_id="common:B", unit_id="unit-B", provider_symbol="B",
        bars_sha256=segment["artifact"]["bars_sha256"], first_session=str(SESSIONS[4].date()))
    request["bindings"].append(binding)
    row = _run(inputs)["rows"][0]
    assert row["materialization_gaps"] == []
    final = row["diagnostic_replay"]["snapshots"][-1]
    assert final["net_return"] == pytest.approx(0.048)
    assert [item["security_id"] for item in final["residual_positions"]] == ["common:B"]


def test_mutation_after_replay_prevents_publication(inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import market_predictor.swing.datasets.holding_materialization as module
    original = module.replay_holding

    def changed(spec: HoldingSpecification) -> Any:
        result = original(spec)
        (inputs["root"] / "ownership.json").write_text("tampered")
        return result

    monkeypatch.setattr(module, "replay_holding", changed)
    with pytest.raises(DataReadinessError, match="changed before publication"):
        _run(inputs)
    assert not (inputs["root"] / "compiled.json").exists()


@pytest.mark.parametrize("mutation", [None, "request", "selection", "busy"])
def test_cli_adapter_owns_lease_through_publication_and_rechecks_bootstrap(
    inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mutation: str | None,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import typer
    from typer.testing import CliRunner

    import market_predictor.commands.swing_holding_materialization as command
    import market_predictor.swing.datasets.holding_materialization as compiler
    from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE

    root = inputs["root"]
    plan = root / "plan"
    plan.mkdir()
    request_pin = _write(plan / "_request.json", {"policy": {"parent_plan": "parent", "parent_plan_sha256": "a" * 64}})
    authority_pin = _write(plan / "_authority.json", {"request_sha256": request_pin})
    inputs["selection"].update(correction_plan=str(plan), correction_plan_sha256=authority_pin,
        correction_archive="raw", correction_archive_sha256="b" * 64)
    inputs["batch"]["source_selection"]["sha256"] = _write(root / "selection.json", inputs["selection"])
    batch_pin = _write(root / "request.json", inputs["batch"])
    active = False
    stages = []

    @contextmanager
    def verifier(*args: Any, **kwargs: Any) -> Any:
        nonlocal active
        with heavy_job_lease("fixture-parent-verifier", runtime_dir=root / "runtime"):
            active = True
            stages.append("parent_verified")
            if mutation in {"request", "selection"}:
                (root / f"{mutation}.json").write_text("{}")
            try:
                yield {}
            finally:
                active = False

    def reconstruct(**kwargs: Any) -> Any:
        assert active
        stages.append("reconstructed")
        return inputs["selection"]

    original = compiler.publish_or_verify_symbol_corrected_sources

    def publish(*args: Any, **kwargs: Any) -> None:
        assert active
        stages.append("published")
        original(*args, **kwargs)

    monkeypatch.setattr(command, "verified_initial_fit_raw_share_plan", verifier)
    monkeypatch.setattr(command, "reconstruct_symbol_corrected_sources", reconstruct)
    monkeypatch.setattr(compiler, "publish_or_verify_symbol_corrected_sources", publish)
    app = typer.Typer()
    command.register_holding_materialization_commands(app, SimpleNamespace(print=Mock()))
    args = ["--root", str(root), "--request-file", "request.json", "--expected-request-sha256", batch_pin,
        "--output", "compiled.json"]
    if mutation == "busy":
        with heavy_job_lease("other", runtime_dir=root / "runtime"):
            result = CliRunner().invoke(app, args)
        assert result.exit_code == HEAVY_JOB_BUSY_EXIT_CODE and stages == []
    else:
        result = CliRunner().invoke(app, args)
        if mutation is None:
            assert result.exit_code == 0, result.exception
            assert stages == ["parent_verified", "reconstructed", "published"]
        else:
            assert result.exit_code != 0 and stages == ["parent_verified"]
    assert not active
    assert (root / "compiled.json").exists() == (mutation is None)


def test_multiclass_output_is_identical_across_python_hash_seeds(inputs: dict[str, Any]) -> None:
    inputs["batch"]["requests"][0]["events"] = [_action(inputs, shares=True)]
    root = inputs["root"]
    bundle = {key: value for key, value in inputs.items() if key in {"batch", "selection"}}
    _write(root / "seed-input.json", bundle)
    code = (
        "import json,runpy,sys; from pathlib import Path; "
        "ns=runpy.run_path(sys.argv[1]); root=Path(sys.argv[2]); "
        "data=json.loads((root/'seed-input.json').read_text()); data['root']=root; "
        "output=root/'compiled.json'; pin=ns['file_sha256'](output) if output.exists() else None; "
        "print(ns['_run'](data,expected_output=pin)['audit_sha256'])"
    )
    results = [subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(root)],
        env={**os.environ, "PYTHONHASHSEED": seed}, check=True, capture_output=True, text=True).stdout.strip()
        for seed in ("1", "37")]
    assert results[0] == results[1]
