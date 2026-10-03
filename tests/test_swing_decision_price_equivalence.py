"""Two-policy proof tests use synthetic metadata, never source or target payloads."""
from __future__ import annotations

import json
import tomllib
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_decisions
from market_predictor.swing.datasets.decision_price_equivalence import verify_decision_price_equivalence

REPO = Path(__file__).resolve().parents[1]
ACTION_FIELDS = {"action_config", "action_archive", "action_audit_sha256"}


def _policy() -> dict[str, Any]:
    pin = {"path": "unread/source.json", "sha256": "a" * 64}
    return dict(schema_version="market_predictor.corrected_outcomes", scope="partial_source_bounded_initial_fit_research",
        source_selection=dict(pin), parent_config=dict(pin), action_config=dict(pin), action_archive="unread/actions",
        action_audit_sha256="a" * 64, symbol_corrections=dict(pin), research_contract=dict(pin), simulation_policy=dict(pin),
        decision_corrections=[dict(security_id="synthetic-a", parent_ticker="OLD", ticker="AAA", first_session="2019-07-09",
            last_session="2024-05-28", document_ids=["synthetic-symbol-change"])],
        official_windows=[dict(security_id="synthetic-a", first_session="2023-01-01", last_session="2024-05-28",
            document=dict(pin), record_locator="synthetic paragraph", reason="unproven payment")],
        reviewed_cash_distribution_scopes=[dict(action_id="synthetic-action", symbol="AAA", family="cash_dividends",
            entitlement_basis="reviewed_cash_ex_distribution_cutoff", record_sha256="a" * 64, ex_date="2023-01-03",
            record_date="2023-01-04", official_payable_date="2023-01-05", inventory=dict(pin), archive="unread/distributions",
            report_sha256="a" * 64, document_id="synthetic-distribution", document=dict(pin), record_locator="paragraph 1")],
        decision_start="2019-07-09", numerical_end="2024-05-28", cost_prepaid_fraction=0.002,
        dividend_policy="evidenced_marks_only_no_inferred_receivables", managed_policy="unavailable", batch_rows=64,
        promotion_eligible=False)


def _value(value: Any) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{name} = {_value(item)}" for name, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return json.dumps(value)


def _write_policy(root: Path, name: str, policy: dict[str, Any]) -> SourcePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key} = {_value(value)}" for key, value in policy.items()) + "\n", encoding="utf-8")
    return SourcePin(path=name, sha256=file_sha256(path))


@pytest.fixture
def policies(tmp_path: Path) -> tuple[Path, SourcePin, SourcePin]:
    decision = _write_policy(tmp_path, "decision.toml", _policy())
    target = _policy()
    target.update(action_config={"path": "unread/current-actions.toml", "sha256": "b" * 64},
        action_archive="unread/current-actions", action_audit_sha256="c" * 64)
    return tmp_path, decision, _write_policy(tmp_path, "target.toml", target)


def test_action_rebinding_has_serializable_frozen_shared_semantics(policies: tuple[Path, SourcePin, SourcePin]) -> None:
    root, decision, target = policies
    proof = verify_decision_price_equivalence(root, decision, target)
    shared = {name: value for name, value in proof.decision_policy.model_dump(mode="json").items() if name not in ACTION_FIELDS}
    assert proof.shared_semantics_sha256 == json_sha256(shared)
    assert proof.decision_policy.decision_corrections == proof.target_policy.decision_corrections
    assert proof.decision_policy.action_config != proof.target_policy.action_config
    assert json.loads(json.dumps(proof.as_record())) == {"schema": "market_predictor.decision_price_equivalence",
        "decision_config": decision.model_dump(mode="json"), "target_config": target.model_dump(mode="json"),
        "shared_semantics_sha256": json_sha256(shared)}
    record = proof.as_record()
    record["decision_config"]["path"] = "tampered"
    assert proof.as_record()["decision_config"]["path"] == "decision.toml"
    with pytest.raises(FrozenInstanceError):
        proof.shared_semantics_sha256 = "a" * 64  # type: ignore[misc]


def test_only_configuration_bytes_are_opened(policies: tuple[Path, SourcePin, SourcePin], monkeypatch: pytest.MonkeyPatch) -> None:
    root, decision, target = policies
    expected = {root / decision.path, root / target.path}
    seen: list[Path] = []
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        assert path in expected, "equivalence must not load unadmitted sources, actions or targets"
        seen.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    verify_decision_price_equivalence(root, decision, target)
    assert set(seen) == expected


def test_action_rebinding_preserves_corrected_and_parent_decision_ids(policies: tuple[Path, SourcePin, SourcePin]) -> None:
    proof = verify_decision_price_equivalence(*policies)
    metadata = pd.DataFrame([dict(security_id=identity, ticker=ticker, sector="technology", primary_benchmark="XLK",
        session_date_et=date(2024, 1, 2), timeframe="1Day", bar_start_utc=pd.Timestamp("2024-01-02T05:00:00Z"),
        decision_time_utc=pd.Timestamp("2024-01-02T22:00:00Z"), prediction_cutoff_policy_id="synthetic-close-cutoff")
        for identity, ticker in (("synthetic-a", "OLD"), ("synthetic-b", "KEEP"))])
    parent = stamp_canonical_decision_ids(metadata)
    decision = corrected_decisions(parent, proof.decision_policy.decision_corrections)
    target = corrected_decisions(parent, proof.target_policy.decision_corrections)
    pd.testing.assert_frame_equal(decision, target)
    assert decision.ticker.tolist() == ["AAA", "KEEP"]
    assert decision.parent_ticker.tolist() == ["OLD", "KEEP"]
    assert decision.parent_decision_id.tolist() == target.parent_decision_id.tolist() == parent.decision_id.tolist()
    canonical = metadata.copy()
    canonical.loc[0, "ticker"] = "AAA"
    assert decision.decision_id.tolist() == target.decision_id.tolist() == stamp_canonical_decision_ids(canonical).decision_id.tolist()
    assert decision.decision_id.iloc[0] != parent.decision_id.iloc[0]
    assert decision.decision_id.iloc[1] == parent.decision_id.iloc[1]


@pytest.mark.parametrize("side", ["decision", "target"])
@pytest.mark.parametrize("poison", ["missing", "wrong_pin", "changed_bytes"])
def test_both_configuration_pins_are_independent(policies: tuple[Path, SourcePin, SourcePin], side: str, poison: str) -> None:
    root, decision, target = policies
    pin = decision if side == "decision" else target
    if poison == "missing":
        changed = SourcePin(path="missing.toml", sha256=pin.sha256)
    elif poison == "wrong_pin":
        changed = SourcePin(path=pin.path, sha256="f" * 64)
    else:
        with (root / pin.path).open("ab") as stream:
            stream.write(b"# changed bytes\n")
        changed = pin
    with pytest.raises((DataReadinessError, FileNotFoundError)):
        verify_decision_price_equivalence(root, changed if side == "decision" else decision, changed if side == "target" else target)


_CHANGES: dict[str, Any] = {
    "schema_version": "other", "scope": "other", "source_selection": {"path": "different.json", "sha256": "b" * 64},
    "parent_config": {"path": "different.json", "sha256": "b" * 64},
    "symbol_corrections": {"path": "different.json", "sha256": "b" * 64},
    "research_contract": {"path": "different.json", "sha256": "b" * 64},
    "simulation_policy": {"path": "different.json", "sha256": "b" * 64},
    "decision_corrections": [{**_policy()["decision_corrections"][0], "ticker": "BBB"}],
    "official_windows": [{**_policy()["official_windows"][0], "reason": "different interpretation"}],
    "reviewed_cash_distribution_scopes": [{**_policy()["reviewed_cash_distribution_scopes"][0], "record_sha256": "b" * 64}],
    "decision_start": "2019-07-10", "numerical_end": "2024-05-29", "cost_prepaid_fraction": 0.001,
    "dividend_policy": "other", "managed_policy": "other", "batch_rows": 32, "promotion_eligible": True,
}


@pytest.mark.parametrize("field", sorted(_CHANGES))
def test_every_other_typed_field_is_rejected(policies: tuple[Path, SourcePin, SourcePin], field: str) -> None:
    root, decision, _ = policies
    assert set(_CHANGES) | ACTION_FIELDS == set(CorrectedOutcomePolicy.model_fields)
    target = _write_policy(root, "target.toml", {**_policy(), field: _CHANGES[field]})
    with pytest.raises((DataReadinessError, ValidationError)):
        verify_decision_price_equivalence(root, decision, target)


def test_policy_formatting_does_not_change_shared_semantics(policies: tuple[Path, SourcePin, SourcePin]) -> None:
    root, decision, target = policies
    before = verify_decision_price_equivalence(root, decision, target)
    path = root / target.path
    path.write_text("# equivalent typed policy, different bytes\n" + path.read_text(), encoding="utf-8")
    updated = SourcePin(path=target.path, sha256=file_sha256(path))
    after = verify_decision_price_equivalence(root, decision, updated)
    assert before.shared_semantics_sha256 == after.shared_semantics_sha256
    assert before.as_record() != after.as_record()


def test_separate_target_configuration_changes_exactly_three_bindings() -> None:
    original = REPO / "configs/swing_corrected_outcomes.toml"
    target = REPO / "configs/swing_canonical_targets.toml"
    assert file_sha256(original) == "2be2077ae44a1cddfb02feeac139a23246f481d9159455955a956fd94ecbecdf"
    left_bytes, right_bytes = original.read_bytes(), target.read_bytes()
    left, right = tomllib.loads(left_bytes.decode()), tomllib.loads(right_bytes.decode())
    assert {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)} == ACTION_FIELDS
    assert right["action_archive"] == "data/raw/swing_corporate_action_sources_canonical"
    assert right["action_audit_sha256"] == "a713f53f2169be60ddd5cbf81786be8f772127496d65a93862eb72992f643add"
    assert right["action_config"] == {"path": "configs/swing_corporate_action_sources.toml",
        "sha256": "c883953e51df99990523be95504e34c6ae7d5ff48ba20e3951d2d7e8d0262554"}
    assert file_sha256(REPO / right["action_config"]["path"]) == right["action_config"]["sha256"]
    original_other_lines = [line for line in left_bytes.splitlines(keepends=True) if not line.startswith(
        (b"action_config =", b"action_archive =", b"action_audit_sha256 ="))]
    target_other_lines = [line for line in right_bytes.splitlines(keepends=True) if not line.startswith(
        (b"action_config =", b"action_archive =", b"action_audit_sha256 ="))]
    assert target_other_lines == original_other_lines
