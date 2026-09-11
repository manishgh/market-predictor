from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence import readiness_authority as authority_module
from market_predictor.evidence.readiness_authority import (
    CURRENT_RUN_SCHEMA,
    CURRENT_SUMMARY_SCHEMA,
    current_exchange_calendar_identity,
    load_readiness_authority,
    publish_readiness_authority,
)

ROOT = Path(__file__).parents[1]
HISTORICAL = ROOT / "tests" / "fixtures" / "historical_readiness_authority"


def test_retained_historical_v1_replays_strictly() -> None:
    verified = load_readiness_authority(HISTORICAL)

    assert verified.format_version == "historical_v1"
    assert verified.summary["status"] == "blocked_pending_targeted_acquisition"
    assert len(verified.session_calendar) == 1_730
    assert verified.request_sha256 == verified.manifest["request_sha256"]


def test_historical_v1_does_not_depend_on_current_calendar_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authority_module.xcals,
        "get_calendar",
        lambda _name: (_ for _ in ()).throw(AssertionError("calendar consulted")),
    )

    verified = load_readiness_authority(HISTORICAL)

    assert verified.format_version == "historical_v1"


def test_historical_v1_cannot_authorize_current_planning() -> None:
    with pytest.raises(DataReadinessError, match="cannot authorize current planning"):
        load_readiness_authority(HISTORICAL, require_current=True)


def test_current_publication_replays_existing_output(tmp_path: Path) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    output = tmp_path / "readiness"

    first = publish_readiness_authority(
        output_directory=output,
        request=request,
        request_sha256=request_hash,
        summary=summary,
        evidence=evidence,
    )
    second = publish_readiness_authority(
        output_directory=output,
        request=request,
        request_sha256=request_hash,
        summary=summary,
        evidence=evidence,
    )

    assert first.format_version == "current_v3"
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.session_calendar.equals(first.session_calendar)
    assert not output.with_name(f".{output.name}.staging").exists()


def test_empty_evidence_cannot_publish_ready_authority(tmp_path: Path) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    empty = {
        name: pd.DataFrame(columns=columns)
        for name, columns in authority_module._CSV_COLUMNS.items()
    }
    summary.update(
        {
            "status": "ready_for_ER2",
            "er2_authorized": True,
            "blocking_findings": 0,
            "nonblocking_required_work": 0,
        }
    )
    summary["acquisition_plan"]["authorized_by_audit"] = False

    with pytest.raises(DataReadinessError, match="source inventory"):
        publish_readiness_authority(
            output_directory=tmp_path / "empty",
            request=request,
            request_sha256=request_hash,
            summary=summary,
            evidence=empty,
        )


def test_boolean_integer_evidence_is_rejected(tmp_path: Path) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    evidence["catalyst_readiness.csv"]["observed_count"] = evidence[
        "catalyst_readiness.csv"
    ]["observed_count"].astype(object)
    evidence["catalyst_readiness.csv"].loc[0, "observed_count"] = True

    with pytest.raises(DataReadinessError, match="must not contain Booleans"):
        publish_readiness_authority(
            output_directory=tmp_path / "boolean-count",
            request=request,
            request_sha256=request_hash,
            summary=summary,
            evidence=evidence,
        )


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    with (output / "session_calendar.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(DataReadinessError, match="does not verify"):
        load_readiness_authority(output, require_current=True)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    raw = json.dumps(summary)[:-1] + ',"status":"ready_for_ER2"}'
    (output / "summary.json").write_text(raw, encoding="utf-8")
    _rebind(output, "summary.json")

    with pytest.raises(DataReadinessError, match="summary is invalid"):
        load_readiness_authority(output, require_current=True)


def test_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["memory"]["peak_working_set_gib"] = float("nan")
    (output / "summary.json").write_text(
        json.dumps(summary, allow_nan=True),
        encoding="utf-8",
    )
    _rebind(output, "summary.json")

    with pytest.raises(DataReadinessError, match="summary is invalid"):
        load_readiness_authority(output, require_current=True)


def test_artifact_record_requires_exact_schema(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    manifest = _json(output / "_manifest.json")
    manifest["artifacts"][0]["rows"] = 1
    _write_manifest_and_rebind_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="artifact record 0 schema differs"):
        load_readiness_authority(output, require_current=True)


def test_duplicate_artifact_paths_are_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    manifest = _json(output / "_manifest.json")
    manifest["artifacts"][1]["path"] = manifest["artifacts"][0]["path"]
    _write_manifest_and_rebind_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="duplicate artifact paths"):
        load_readiness_authority(output, require_current=True)


def test_missing_artifact_inventory_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    manifest = _json(output / "_manifest.json")
    manifest["artifacts"].pop()
    _write_manifest_and_rebind_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="artifact inventory differs"):
        load_readiness_authority(output, require_current=True)


def test_unexpected_directory_inventory_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    (output / "unexpected").mkdir()

    with pytest.raises(DataReadinessError, match="file inventory differs"):
        load_readiness_authority(output, require_current=True)


def test_boolean_artifact_size_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    manifest = _json(output / "_manifest.json")
    manifest["artifacts"][0]["bytes"] = True
    _write_manifest_and_rebind_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="nonnegative integer"):
        load_readiness_authority(output, require_current=True)


def test_manifest_summary_status_mismatch_is_rejected(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    manifest = _json(output / "_manifest.json")
    manifest["status"] = "ready_for_ER2"
    _write_manifest_and_rebind_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="manifest and summary status differ"):
        load_readiness_authority(output, require_current=True)


def test_csv_schema_mismatch_is_rejected_even_when_hash_bound(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    calendar = output / "session_calendar.csv"
    calendar.write_text("wrong_column\nvalue\n", encoding="utf-8")
    _rebind(output, "session_calendar.csv")

    with pytest.raises(DataReadinessError, match="artifact schema differs"):
        load_readiness_authority(output, require_current=True)


def test_negative_calendar_count_is_rejected_even_when_hash_bound(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    calendar = pd.read_csv(output / "session_calendar.csv")
    calendar.loc[0, "source_rows"] = -1
    calendar.to_csv(output / "session_calendar.csv", index=False)
    _rebind(output, "session_calendar.csv")

    with pytest.raises(DataReadinessError, match="nonnegative integers"):
        load_readiness_authority(output, require_current=True)


def test_invalid_calendar_date_is_rejected_even_when_hash_bound(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    calendar = pd.read_csv(output / "session_calendar.csv")
    calendar.loc[0, "session_date_et"] = "2026-02-31"
    calendar.to_csv(output / "session_calendar.csv", index=False)
    _rebind(output, "session_calendar.csv")

    with pytest.raises(DataReadinessError, match="invalid date"):
        load_readiness_authority(output, require_current=True)


def test_non_trading_calendar_date_is_rejected_even_when_hash_bound(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    calendar = pd.read_csv(output / "session_calendar.csv")
    calendar.loc[0, "session_date_et"] = "2021-07-10"
    calendar.to_csv(output / "session_calendar.csv", index=False)
    _rebind(output, "session_calendar.csv")

    with pytest.raises(DataReadinessError, match="non-trading session"):
        load_readiness_authority(output, require_current=True)


def test_duplicate_strategy_session_is_rejected_even_when_hash_bound(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    calendar = pd.read_csv(output / "session_calendar.csv")
    calendar = pd.concat([calendar, calendar.iloc[[0]]], ignore_index=True)
    calendar.to_csv(output / "session_calendar.csv", index=False)
    _rebind(output, "session_calendar.csv")

    with pytest.raises(DataReadinessError, match="duplicate rows|duplicate strategy sessions"):
        load_readiness_authority(output, require_current=True)


def test_unrelated_blocker_cannot_authorize_intraday_acquisition(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    blockers = pd.read_csv(output / "blockers.csv")
    blockers.loc[len(blockers)] = {
        "blocker_code": "swing_session_history_below_gate",
        "scope": "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1",
        "blocks_er2": True,
        "required_action": "repair swing history",
        "detail": "test",
    }
    blockers.to_csv(output / "blockers.csv", index=False)
    _rebind(output, "blockers.csv")
    summary = _json(output / "summary.json")
    summary["blocking_findings"] = 2
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _rebind(output, "summary.json")

    with pytest.raises(DataReadinessError, match="eligible blockers"):
        load_readiness_authority(output, require_current=True)


def test_unknown_intraday_scope_cannot_authorize_acquisition(tmp_path: Path) -> None:
    output = _publish(tmp_path)
    blockers = pd.read_csv(output / "blockers.csv")
    blocker = blockers["blocker_code"].eq("intraday_session_history_below_gate")
    blockers.loc[blocker, "scope"] = "INTRADAY.UNKNOWN"
    blockers.to_csv(output / "blockers.csv", index=False)
    _rebind(output, "blockers.csv")

    with pytest.raises(DataReadinessError, match="eligible blockers"):
        load_readiness_authority(output, require_current=True)


def test_intraday_evidence_identity_must_match_request(tmp_path: Path) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    original = "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1"
    replacement = "INTRADAY.DISCONNECTED.30M.V1"
    for frame in evidence.values():
        for column in ("strategy_id", "scope"):
            if column in frame:
                frame.loc[frame[column].eq(original), column] = replacement

    with pytest.raises(DataReadinessError, match="intraday evidence identity"):
        publish_readiness_authority(
            output_directory=tmp_path / "disconnected",
            request=request,
            request_sha256=request_hash,
            summary=summary,
            evidence=evidence,
        )


def test_publication_failure_cleans_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    output = tmp_path / "failed"

    def fail_csv(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)
    with pytest.raises(OSError, match="injected write failure"):
        publish_readiness_authority(
            output_directory=output,
            request=request,
            request_sha256=request_hash,
            summary=summary,
            evidence=evidence,
        )

    assert not output.exists()
    assert not output.with_name(f".{output.name}.staging").exists()
    assert not output.with_name(f".{output.name}.staging.owner.json").exists()


def test_unowned_staging_is_preserved(tmp_path: Path) -> None:
    request, request_hash, summary, evidence = _current_inputs()
    output = tmp_path / "blocked"
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir()
    marker = staging / "keep.txt"
    marker.write_text("external", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="not owned"):
        publish_readiness_authority(
            output_directory=output,
            request=request,
            request_sha256=request_hash,
            summary=summary,
            evidence=evidence,
        )

    assert marker.read_text(encoding="utf-8") == "external"


def test_reparse_ancestry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _publish(tmp_path)
    original = path_integrity.is_reparse_point
    monkeypatch.setattr(
        path_integrity,
        "is_reparse_point",
        lambda path: path == output or original(path),
    )

    with pytest.raises(DataReadinessError, match="reparse point"):
        load_readiness_authority(output)


def _publish(tmp_path: Path) -> Path:
    request, request_hash, summary, evidence = _current_inputs()
    output = tmp_path / "readiness"
    publish_readiness_authority(
        output_directory=output,
        request=request,
        request_sha256=request_hash,
        summary=summary,
        evidence=evidence,
    )
    return output


def _current_inputs() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, pd.DataFrame]]:
    historical = _json(HISTORICAL / "_request.json")
    request: dict[str, Any] = {
        "schema": CURRENT_RUN_SCHEMA,
        "policy_sha256": historical["policy_sha256"],
        "policy_file_sha256": historical["policy_file_sha256"],
        "swing_training_policy_file_sha256": historical["swing_policy_file_sha256"],
        "strategy_contract_file_sha256": "a" * 64,
        "intraday_policy_file_sha256": historical["intraday_policy_file_sha256"],
        "sources": {
            **historical["sources"],
            "swing": {
                "type": "verified_edge_rebuild_swing_sources",
                "strategy_id": "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1",
                "horizon_sessions": 10,
                "panel_manifest_sha256": "1" * 64,
                "panel_authority_sha256": "2" * 64,
                "panel_request_sha256": "3" * 64,
                "candidate_status": "no_candidate",
                "candidate_id": None,
                "candidate_authority_sha256": "4" * 64,
                "promoted_bundle_status": "unavailable",
                "promoted_bundle_sha256": None,
                "technical_rows": 630_978,
                "proxy_rows": 103_837,
            },
            "intraday": {
                **historical["sources"]["intraday"],
                "strategy_id": "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1",
                "proxy_strategy_id": "INTRADAY.VWAP_REVERSION.30M.V1",
            },
        },
        "implementation": {
            "authority": {"path": "readiness_authority.py", "sha256": "1" * 64},
            "catalyst_source_verification": {
                "path": "catalyst_lineage.py",
                "sha256": "7" * 64,
            },
            "contracts": {"path": "contracts.py", "sha256": "2" * 64},
            "intraday_source_verification": {
                "path": "specialist_experiments.py",
                "sha256": "3" * 64,
            },
            "promotion_verification": {
                "path": "bundle_verification.py",
                "sha256": "4" * 64,
            },
            "readiness": {"path": "audit.py", "sha256": "5" * 64},
            "swing_source_verification": {
                "path": "swing_training.py",
                "sha256": "6" * 64,
            },
        },
        "exchange_calendar": current_exchange_calendar_identity(),
        "training_performed": False,
        "download_performed": False,
    }
    request_hash = authority_module._json_sha256(request)
    historical_summary = _json(HISTORICAL / "summary.json")
    summary: dict[str, Any] = {
        **historical_summary,
        "schema": CURRENT_SUMMARY_SCHEMA,
        "request_sha256": request_hash,
    }
    evidence = {
        name: pd.read_csv(HISTORICAL / name)
        for name in authority_module._CSV_COLUMNS
    }
    return request, request_hash, summary, evidence


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _rebind(root: Path, name: str) -> None:
    manifest = _json(root / "_manifest.json")
    artifact = root / name
    for record in manifest["artifacts"]:
        if record["path"] == name:
            record["bytes"] = artifact.stat().st_size
            record["sha256"] = authority_module._file_sha256(artifact)
            break
    _write_manifest_and_rebind_authority(root, manifest)


def _write_manifest_and_rebind_authority(root: Path, manifest: dict[str, Any]) -> None:
    (root / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authority = _json(root / "_authority.json")
    authority["artifact_sha256"] = authority_module._file_sha256(root / "_manifest.json")
    (root / "_authority.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
