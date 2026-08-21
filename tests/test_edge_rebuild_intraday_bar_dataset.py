from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import intraday_bar_dataset as module
from market_predictor.edge_rebuild.intraday_bar_dataset import (
    INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
    load_complete_intraday_bar_dataset,
    publish_intraday_bar_dataset,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.core.errors import DataReadinessError

CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")


def test_publication_resumes_completed_sessions_and_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    calls: list[str] = []

    def build_session(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        session = str(kwargs["session_date"])
        calls.append(session)
        decision = pd.Timestamp(f"{session}T15:01:00Z")
        rows = pd.DataFrame(
            {
                "session_date_et": [session],
                "ticker": ["AAA"],
                "decision_time_utc": [decision],
                "dataset_eligible": [True],
            }
        )
        return rows, {"session_date_et": session, "stock_sessions": []}

    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_build_session", build_session)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    output = tmp_path / "output" / "dataset"
    progress = _publish(inputs, output, max_sessions=1)

    work = next(output.parent.glob(f".{output.name}.*.work"))
    first_rows = work / "sessions" / "session_date_et=2024-01-03" / "rows.parquet"
    first_sha256 = file_sha256(first_rows)
    manifest = _publish(inputs, output)

    assert progress["state"] == "work_incomplete"
    assert progress["summary"]["completed_sessions"] == 1
    assert calls == ["2024-01-03", "2024-01-04"]
    assert manifest["summary"]["completed_sessions"] == 2
    assert manifest["summary"]["rows"] == 2
    assert file_sha256(
        output / "sessions" / "session_date_et=2024-01-03" / "rows.parquet"
    ) == first_sha256
    assert load_complete_intraday_bar_dataset(output) == manifest


def test_resume_rejects_tampered_completed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)

    def build_session(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        session = str(kwargs["session_date"])
        return (
            pd.DataFrame(
                {
                    "session_date_et": [session],
                    "ticker": ["AAA"],
                    "decision_time_utc": [pd.Timestamp(f"{session}T15:01:00Z")],
                    "dataset_eligible": [False],
                }
            ),
            {"session_date_et": session, "stock_sessions": []},
        )

    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_build_session", build_session)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    output = tmp_path / "output" / "dataset"
    _publish(inputs, output, max_sessions=1)
    work = next(output.parent.glob(f".{output.name}.*.work"))
    audit = work / "sessions" / "session_date_et=2024-01-03" / "audit.json"
    audit.write_text("{}", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="session unit differs"):
        _publish(inputs, output)


def test_complete_authority_rejects_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_build_session",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "session_date_et": [str(kwargs["session_date"])],
                    "ticker": ["AAA"],
                    "decision_time_utc": [
                        pd.Timestamp(f"{kwargs['session_date']}T15:01:00Z")
                    ],
                    "dataset_eligible": [True],
                }
            ),
            {"session_date_et": str(kwargs["session_date"]), "stock_sessions": []},
        ),
    )
    output = tmp_path / "output" / "dataset"
    _publish(inputs, output)
    (output / "unexpected.txt").write_text("tamper", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="immutable file set differs"):
        load_complete_intraday_bar_dataset(output)


def test_output_cannot_overlap_an_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(DataReadinessError, match="output overlaps an input"):
        module._require_path_isolation(source / "nested", (source,))


def test_worker_watchdog_exits_when_parent_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter((True, False))

    class Owner:
        @staticmethod
        def is_alive() -> bool:
            return next(states)

    exit_codes: list[int] = []

    def exit_process(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.os, "_exit", exit_process)

    with pytest.raises(SystemExit, match="70"):
        module._exit_when_parent_stops(Owner())

    assert exit_codes == [70]


def test_small_publication_does_not_force_full_gc_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_build_session",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "session_date_et": [str(kwargs["session_date"])],
                    "ticker": ["AAA"],
                    "decision_time_utc": [
                        pd.Timestamp(f"{kwargs['session_date']}T15:01:00Z")
                    ],
                    "dataset_eligible": [True],
                }
            ),
            {"session_date_et": str(kwargs["session_date"]), "stock_sessions": []},
        ),
    )
    monkeypatch.setattr(
        module,
        "release_process_memory",
        lambda: pytest.fail("full GC must not run after every session"),
    )

    manifest = _publish(inputs, tmp_path / "output" / "dataset")

    assert manifest["summary"]["completed_sessions"] == 2


def test_resume_removes_stale_session_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_build_session",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "session_date_et": [str(kwargs["session_date"])],
                    "ticker": ["AAA"],
                    "decision_time_utc": [
                        pd.Timestamp(f"{kwargs['session_date']}T15:01:00Z")
                    ],
                    "dataset_eligible": [True],
                }
            ),
            {"session_date_et": str(kwargs["session_date"]), "stock_sessions": []},
        ),
    )
    output = tmp_path / "output" / "dataset"
    _publish(inputs, output, max_sessions=1)
    work = next(output.parent.glob(f".{output.name}.*.work"))
    stale = work / "sessions" / ".session_date_et=2024-01-04.dead.staging"
    stale.mkdir()
    (stale / "partial.tmp").write_text("partial", encoding="utf-8")

    manifest = _publish(inputs, output)

    assert manifest["summary"]["completed_sessions"] == 2
    assert not stale.exists()


def test_complete_work_is_recovered_after_interrupted_final_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_build_session",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "session_date_et": [str(kwargs["session_date"])],
                    "ticker": ["AAA"],
                    "decision_time_utc": [
                        pd.Timestamp(f"{kwargs['session_date']}T15:01:00Z")
                    ],
                    "dataset_eligible": [True],
                }
            ),
            {"session_date_et": str(kwargs["session_date"]), "stock_sessions": []},
        ),
    )
    output = tmp_path / "output" / "dataset"
    manifest = _publish(inputs, output)
    work = output.with_name(
        f".{output.name}.{str(manifest['request_sha256'])[:16]}.work"
    )
    output.replace(work)

    recovered = _publish(inputs, output)

    assert recovered == manifest
    assert output.is_dir()
    assert not work.exists()


def test_transformation_change_cannot_reuse_completed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    calls: list[str] = []
    _patch_verified_inputs(monkeypatch, inputs)
    monkeypatch.setattr(module, "_projection_bar_files", lambda *_args: {})

    def build_session(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        session = str(kwargs["session_date"])
        calls.append(session)
        return (
            pd.DataFrame(
                {
                    "session_date_et": [session],
                    "ticker": ["AAA"],
                    "decision_time_utc": [pd.Timestamp(f"{session}T15:01:00Z")],
                    "dataset_eligible": [True],
                }
            ),
            {"session_date_et": session, "stock_sessions": []},
        )

    monkeypatch.setattr(module, "_build_session", build_session)
    output = tmp_path / "output" / "dataset"
    _publish(inputs, output, max_sessions=1)
    original = module._transformation_identity()
    changed = {**original, "sha256": "f" * 64}
    monkeypatch.setattr(module, "_transformation_identity", lambda: changed)

    manifest = _publish(inputs, output)

    assert manifest["transformation_sha256"] == "f" * 64
    assert calls == ["2024-01-03", "2024-01-03", "2024-01-04"]
    assert len(list(output.parent.glob(f".{output.name}.*.work"))) == 1


def _publish(
    inputs: dict[str, Any],
    output: Path,
    *,
    max_sessions: int | None = None,
    session_workers: int = 1,
) -> dict[str, Any]:
    contract = load_strategy_contract(CONTRACT_PATH)
    return publish_intraday_bar_dataset(
        selection_directory=inputs["selection"],
        stock_collection_directory=inputs["stock"],
        stock_coverage_directory=inputs["coverage"],
        benchmark_collection_directory=inputs["benchmarks"],
        membership_authority_directory=inputs["memberships"],
        five_minute_projection_directory=inputs["projection"],
        strategy_contract=contract,
        strategy_contract_path=CONTRACT_PATH,
        output_directory=output,
        max_sessions_per_invocation=max_sessions,
        session_workers=session_workers,
    )


def _inputs(tmp_path: Path) -> dict[str, Any]:
    paths = {
        name: tmp_path / "inputs" / name
        for name in (
            "selection",
            "stock",
            "coverage",
            "benchmarks",
            "memberships",
            "projection",
        )
    }
    for path in paths.values():
        path.mkdir(parents=True)
    _write_json(paths["projection"] / "_manifest.json", {"state": "complete"})
    _write_json(
        paths["projection"] / "_authority.json",
        {"schema": INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA},
    )
    return paths


def _patch_verified_inputs(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, Any],
) -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    selection = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "session_date_et": ["2024-01-03", "2024-01-04"],
            "activation_time_utc": pd.to_datetime(
                ["2024-01-03T15:01:00Z", "2024-01-04T15:01:00Z"]
            ),
        }
    )
    parent = {
        "selection_authority_sha256": "selection-authority",
        "selection_manifest_sha256": "selection-manifest",
        "selection_table_sha256": "selection-table",
        "five_minute_canonical_authority_sha256": "canonical-authority",
        "five_minute_canonical_manifest_sha256": "canonical-manifest",
        "five_minute_canonical_file_inventory_sha256": "canonical-inventory",
        "strategy_contract_sha256": contract.sha256(),
        "intraday_data_contract_sha256": (
            "c88f1a2c1eb3cc3065f5a7bc38d97662da546260c760b45be321aa1718a50b39"
        ),
        "intraday_parent_contract_sha256": contract.sha256(),
        "intraday_contract_lineage_file_sha256": "",
    }
    verified = SimpleNamespace(
        selection=selection,
        excluded_tickers=frozenset(),
        parent_lineage=parent,
        contract_sha256=contract.sha256(),
        stock_artifacts=(),
        benchmark_artifacts=(),
        incomplete_pairs=frozenset(),
    )
    projection = {
        "selection_directory": str(inputs["selection"].resolve()),
        "selection_authority_sha256": parent["selection_authority_sha256"],
        "selection_manifest_sha256": parent["selection_manifest_sha256"],
        "selection_table_sha256": parent["selection_table_sha256"],
        "five_minute_canonical_authority_sha256": parent[
            "five_minute_canonical_authority_sha256"
        ],
        "five_minute_canonical_manifest_sha256": parent[
            "five_minute_canonical_manifest_sha256"
        ],
        "five_minute_canonical_file_inventory_sha256": parent[
            "five_minute_canonical_file_inventory_sha256"
        ],
        "strategy_contract_file_sha256": file_sha256(CONTRACT_PATH),
        "strategy_contract_sha256": contract.sha256(),
        "intraday_data_contract_sha256": parent[
            "intraday_data_contract_sha256"
        ],
        "intraday_parent_contract_sha256": parent[
            "intraday_parent_contract_sha256"
        ],
        "intraday_contract_lineage_file_sha256": parent[
            "intraday_contract_lineage_file_sha256"
        ],
        "file_inventory_sha256": "projection-inventory",
        "files": [],
    }
    monkeypatch.setattr(module, "_verify_inputs", lambda **_kwargs: verified)
    monkeypatch.setattr(
        module,
        "load_complete_selected_session_five_minute_projection",
        lambda _path: projection,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
