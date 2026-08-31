from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import market_predictor.intraday.datasets.bar_execution_evidence as module
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.bar_dataset import _transformation_identity


def test_resumed_publication_produces_complete_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    evidence = tmp_path / "execution"
    manifest = _dataset_manifest()
    calls = 0

    def publish(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            work = tmp_path / ".dataset.abc.work"
            _write_request(work, manifest)
            _write_unit(work, "2024-01-02")
            return {
                "state": "work_incomplete",
                "request_sha256": manifest["request_sha256"],
                "work_directory": str(work),
                "summary": {"memory": _memory(0.5)},
            }
        work = tmp_path / ".dataset.abc.work"
        _write_unit(work, "2024-01-03")
        work.replace(dataset)
        for name in ("_manifest.json", "_authority.json"):
            (dataset / name).write_text("{}", encoding="utf-8")
        return manifest

    monkeypatch.setattr(module, "publish_intraday_bar_dataset", publish)
    monkeypatch.setattr(
        module,
        "load_complete_intraday_bar_dataset",
        lambda _path: manifest,
    )

    first = module.publish_intraday_bar_dataset_with_execution_evidence(**_publication_arguments(tmp_path, dataset, evidence))
    second = module.publish_intraday_bar_dataset_with_execution_evidence(**_publication_arguments(tmp_path, dataset, evidence))

    assert first["state"] == "work_incomplete"
    assert second["state"] == "complete"
    authority = module.load_complete_intraday_bar_dataset_execution_evidence(
        evidence,
        dataset_directory=dataset,
    )
    assert authority["summary"]["invocations"] == 2
    assert authority["summary"]["accounted_sessions"] == 2
    assert authority["summary"]["complete_run_memory_proven"] is True


def test_failed_invocation_remains_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    evidence = tmp_path / "execution"
    manifest = _dataset_manifest()

    def fail(**_kwargs: Any) -> dict[str, Any]:
        work = tmp_path / ".dataset.abc.work"
        _write_request(work, manifest)
        _write_unit(work, "2024-01-02")
        raise RuntimeError("stopped")

    monkeypatch.setattr(module, "publish_intraday_bar_dataset", fail)

    with pytest.raises(RuntimeError, match="stopped"):
        module.publish_intraday_bar_dataset_with_execution_evidence(**_publication_arguments(tmp_path, dataset, evidence))

    receipt_path = next((tmp_path / ".execution.work" / "invocations").glob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "failed"
    assert receipt["processed_sessions"] == ["2024-01-02"]
    assert receipt["exception"]["type"] == "RuntimeError"


def test_started_invocation_prevents_complete_evidence(tmp_path: Path) -> None:
    receipt = _receipt(
        state="started",
        processed=[],
        completed_at=None,
        dataset_directory=tmp_path / "dataset",
    )

    with pytest.raises(DataReadinessError, match="receipt is incomplete"):
        module._validate_execution_receipts(
            [receipt],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_memory_breach_prevents_complete_evidence(tmp_path: Path) -> None:
    memory = _memory(0.5)
    memory["aggregate_peak_upper_bound_gib"] = 3.5
    receipt = _receipt(
        state="complete",
        processed=["2024-01-02", "2024-01-03"],
        memory=memory,
        dataset_directory=tmp_path / "dataset",
    )

    with pytest.raises(DataReadinessError, match="breached memory budget"):
        module._validate_execution_receipts(
            [receipt],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_failed_multi_worker_receipt_requires_aggregate_memory(tmp_path: Path) -> None:
    memory = _memory(0.5)
    memory.pop("aggregate_peak_upper_bound_gib")
    receipt = _receipt(
        state="failed",
        processed=["2024-01-02", "2024-01-03"],
        memory=memory,
        dataset_directory=tmp_path / "dataset",
    )
    receipt["session_workers"] = 2

    with pytest.raises(DataReadinessError, match="omits aggregate memory"):
        module._validate_execution_receipts(
            [receipt],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_aggregate_memory_cannot_be_below_parent_peak(tmp_path: Path) -> None:
    memory = _memory(3.5)
    memory["aggregate_peak_upper_bound_gib"] = 0.5
    receipt = _receipt(
        state="complete",
        processed=["2024-01-02", "2024-01-03"],
        memory=memory,
        dataset_directory=tmp_path / "dataset",
    )

    with pytest.raises(DataReadinessError, match="breached memory budget"):
        module._validate_execution_receipts(
            [receipt],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_request_or_transformation_drift_prevents_complete_evidence(
    tmp_path: Path,
) -> None:
    receipt = _receipt(
        state="complete",
        processed=["2024-01-02", "2024-01-03"],
        dataset_directory=tmp_path / "dataset",
    )
    receipt["transformation_sha256"] = "f" * 64

    with pytest.raises(DataReadinessError, match="receipt identity differs"):
        module._validate_execution_receipts(
            [receipt],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_duplicate_session_receipts_are_rejected(tmp_path: Path) -> None:
    first = _receipt(
        state="work_incomplete",
        processed=["2024-01-02"],
        dataset_directory=tmp_path / "dataset",
    )
    second = _receipt(
        state="complete",
        processed=["2024-01-02", "2024-01-03"],
        dataset_directory=tmp_path / "dataset",
    )

    with pytest.raises(DataReadinessError, match="duplicate sessions"):
        module._validate_execution_receipts(
            [first, second],
            dataset=_dataset_manifest(),
            dataset_directory=tmp_path / "dataset",
        )


def test_tampered_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    evidence = tmp_path / "execution"
    manifest = _dataset_manifest()
    _write_dataset_metadata(dataset)
    receipt_dir = evidence / "invocations"
    receipt_dir.mkdir(parents=True)
    receipt_path = receipt_dir / "one.json"
    receipt_path.write_text(
        json.dumps(
            _receipt(
                state="complete",
                processed=["2024-01-02", "2024-01-03"],
                dataset_directory=dataset,
            )
        ),
        encoding="utf-8",
    )
    inventory = [{"path": "invocations/one.json", "sha256": "0" * 64}]
    (evidence / "_manifest.json").write_text(
        json.dumps(
            {
                "schema": module.INTRADAY_BAR_EXECUTION_MANIFEST_SCHEMA,
                "state": "complete",
                "invocations": inventory,
            }
        ),
        encoding="utf-8",
    )
    (evidence / "_authority.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_complete_intraday_bar_dataset",
        lambda _path: manifest,
    )

    with pytest.raises(DataReadinessError, match="receipt hash differs"):
        module.load_complete_intraday_bar_dataset_execution_evidence(
            evidence,
            dataset_directory=dataset,
        )


def test_post_hoc_assessment_cannot_claim_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    report = tmp_path / "assessment.json"
    _write_dataset_metadata(dataset)
    manifest = _dataset_manifest()
    monkeypatch.setattr(
        module,
        "load_complete_intraday_bar_dataset",
        lambda _path: manifest,
    )

    assessment = module.publish_incomplete_intraday_bar_execution_assessment(
        dataset_directory=dataset,
        output_path=report,
    )

    assert assessment["status"] == "incomplete"
    assert assessment["recorded_scope"] == "final_invocation_only"
    assert assessment["complete_run_memory_proven"] is False


@pytest.mark.parametrize("remove_authority", [False, True])
def test_execution_authority_finalization_is_crash_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_authority: bool,
) -> None:
    dataset = tmp_path / "dataset"
    evidence = tmp_path / "execution"
    work = tmp_path / ".execution.work"
    _write_dataset_metadata(dataset)
    receipt_dir = work / "invocations"
    receipt_dir.mkdir(parents=True)
    receipt = _receipt(
        state="complete",
        processed=["2024-01-02", "2024-01-03"],
        dataset_directory=dataset,
    )
    (receipt_dir / "one.json").write_text(json.dumps(receipt), encoding="utf-8")
    manifest = _dataset_manifest()
    monkeypatch.setattr(
        module,
        "load_complete_intraday_bar_dataset",
        lambda _path: manifest,
    )
    original_replace = Path.replace

    def interrupt_final_rename(self: Path, target: Path) -> Path:
        if self == work:
            raise RuntimeError("interrupted before final rename")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_final_rename)
    with pytest.raises(RuntimeError, match="interrupted before final rename"):
        module._publish_execution_authority(
            work,
            output_directory=dataset,
            execution_evidence_directory=evidence,
        )
    monkeypatch.setattr(Path, "replace", original_replace)
    if remove_authority:
        (work / "_authority.json").unlink()

    assert module._recover_execution_work(
        work,
        output_directory=dataset,
        execution_evidence_directory=evidence,
    )
    recovered = module.load_complete_intraday_bar_dataset_execution_evidence(
        evidence,
        dataset_directory=dataset,
    )
    assert recovered["summary"]["complete_run_memory_proven"] is True


def test_execution_evidence_is_not_part_of_feature_transformation() -> None:
    identity = _transformation_identity()

    assert identity["sha256"] == ("6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51")
    assert "bar_execution_evidence.py" not in {str(item["path"]) for item in identity["files"]}


def _publication_arguments(
    root: Path,
    dataset: Path,
    evidence: Path,
) -> dict[str, Any]:
    return {
        "selection_directory": root / "selection",
        "stock_collection_directory": root / "stock",
        "stock_coverage_directory": root / "coverage",
        "benchmark_collection_directory": root / "benchmark",
        "membership_authority_directory": root / "membership",
        "five_minute_projection_directory": root / "projection",
        "strategy_contract": cast(Any, object()),
        "strategy_contract_path": root / "contract.toml",
        "output_directory": dataset,
        "execution_evidence_directory": evidence,
        "intraday_contract_lineage_path": root / "lineage.toml",
        "max_sessions_per_invocation": 1,
        "session_workers": 1,
    }


def _dataset_manifest() -> dict[str, Any]:
    return {
        "state": "complete",
        "request_sha256": "1" * 64,
        "transformation_sha256": "2" * 64,
        "planned_sessions": ["2024-01-02", "2024-01-03"],
        "summary": {"memory": _memory(0.6)},
    }


def _receipt(
    *,
    state: str,
    processed: list[str],
    completed_at: str | None = "2026-08-31T00:01:00+00:00",
    memory: dict[str, float] | None = None,
    dataset_directory: Path = Path("dataset"),
) -> dict[str, Any]:
    return {
        "schema": module.INTRADAY_BAR_EXECUTION_RECEIPT_SCHEMA,
        "invocation_id": "a" * 32,
        "state": state,
        "started_at_utc": "2026-08-31T00:00:00+00:00",
        "completed_at_utc": completed_at,
        "output_directory": str(dataset_directory.resolve()),
        "request_sha256": "1" * 64,
        "transformation_sha256": "2" * 64,
        "processed_sessions": processed,
        "completed_sessions_after_invocation": len(processed),
        "session_workers": 1,
        "memory": memory or _memory(0.5),
        "exception": None,
    }


def _memory(peak: float) -> dict[str, float]:
    return {
        "hard_budget_gib": 4.0,
        "safety_threshold_gib": 3.25,
        "current_working_set_gib": 0.25,
        "peak_working_set_gib": peak,
        "aggregate_peak_upper_bound_gib": peak,
    }


def _write_request(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "_request.json").write_text(
        json.dumps(
            {
                "request_sha256": manifest["request_sha256"],
                "transformation_sha256": manifest["transformation_sha256"],
            }
        ),
        encoding="utf-8",
    )


def _write_unit(root: Path, session: str) -> None:
    unit = root / "sessions" / f"session_date_et={session}"
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "_unit.json").write_text("{}", encoding="utf-8")


def _write_dataset_metadata(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("_request.json", "_manifest.json", "_authority.json"):
        (root / name).write_text("{}", encoding="utf-8")
