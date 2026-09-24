"""Test-only source stubs cover orchestration/CLI, not source correctness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.commands import issuer_content_inventory as commands
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import issuer_content_inventory as inventory
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.issuer_news_preparation import FIRST

START = pd.Timestamp("2019-07-09T00:00:00Z")
CUTOFF = pd.Timestamp("2024-05-28T22:00:00Z")
HEALTHY_MEMORY = SystemMemory(total_bytes=20 * 1024**3, available_bytes=8 * 1024**3)


@pytest.fixture
def arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "data-root"
    root.mkdir()
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(inventory, "assert_memory_budget", Mock())
    monkeypatch.setattr(inventory, "system_memory_snapshot", Mock(return_value=HEALTHY_MEMORY))
    return dict(root=root, event_artifact=SourcePin(path="data/raw/events.parquet", sha256="a" * 64),
        event_manifest=SourcePin(path="data/raw/events.manifest.json", sha256="b" * 64),
        security_id="test-security", ticker="AAA", start_utc=START, cutoff_utc=CUTOFF,
        output=Path("data/research/content-inventory"))


@pytest.fixture
def inspection(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # These arbitrary rows intentionally make no claim about provider validation.
    rows = pd.DataFrame({"event_id": ["fixture-1", "fixture-2"], "test_only_value": [1, 2]})
    helper = Mock(return_value=(rows, {"test_only_orchestration": True, "records": 2}))
    monkeypatch.setattr(inventory, "inspect_saved_alpaca_content", helper)
    return helper


def _output(arguments: dict[str, Any]) -> Path:
    return Path(arguments["root"]) / Path(arguments["output"])


def _assert_unpublished(arguments: dict[str, Any]) -> None:
    output = _output(arguments)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.pending"))
    assert not (arguments["root"] / "data/runtime/heavy-job.owner.json").exists()


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None:
        pass

    commands.register_issuer_content_commands(app)
    return app


def _cli_arguments() -> list[str]:
    return ["inspect-saved-issuer-content", "--event-artifact", "data/raw/events.parquet",
        "--event-sha256", "a" * 64, "--event-manifest", "data/raw/events.manifest.json",
        "--manifest-sha256", "b" * 64, "--security-id", "test-security", "--ticker", "AAA",
        "--output", "data/research/content-inventory"]


@pytest.mark.parametrize("absolute_output", [False, True])
def test_publication_binds_rows_report_source_and_installed_code(
    arguments: dict[str, Any], inspection: Mock, absolute_output: bool,
) -> None:
    if absolute_output:
        arguments["output"] = _output(arguments)
    rows, summary = inspection.return_value
    original_rows = rows.copy(deep=True)
    original_summary = dict(summary)
    report = inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_called_once_with(**{key: value for key, value in arguments.items() if key != "output"},
                                       memory_check=inventory._guard)
    output = _output(arguments)
    manifest = output / "_manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert report == {**saved, "manifest_sha256": file_sha256(manifest)}
    assert report["status"] == "complete_inventory_only"
    assert report["rows"] == len(rows) == 2
    assert report["inspection"] == original_summary
    assert report["event_artifact"] == arguments["event_artifact"].model_dump()
    assert report["event_manifest"] == arguments["event_manifest"].model_dump()
    assert report["query_security_id"] == arguments["security_id"]
    assert report["query_ticker"] == arguments["ticker"]
    assert pd.Timestamp(report["start_utc"]) == START
    assert pd.Timestamp(report["cutoff_utc"]) == CUTOFF
    assert report["content_qualification"] == "not_established"
    assert report["issuer_attribution"] == "not_established_by_inventory"
    for flag in ("training_eligible", "promotion_eligible", "serving_eligible"):
        assert report[flag] is False
    records = output / report["records_path"]
    assert records.name == "records.parquet"
    assert report["records_sha256"] == file_sha256(records)
    pd.testing.assert_frame_equal(pd.read_parquet(records), original_rows)
    pd.testing.assert_frame_equal(rows, original_rows)
    assert summary == original_summary
    package = Path(inventory.__file__).resolve().parents[1]
    pins = report["implementation_files"]
    assert "market_predictor/research/issuer_content_inventory.py" in pins
    assert "market_predictor/catalysts/issuer_events/content_inventory.py" in pins
    assert {f"market_predictor/{name}" for name in inventory.IMPLEMENTATION_PATHS} == set(pins)
    assert "market_predictor/catalysts/issuer_events/news_history_contracts.py" in pins
    for name, digest in pins.items():
        assert name.startswith("market_predictor/")
        assert digest == file_sha256(package.parent / name)
    assert {path.name for path in output.iterdir()} == {"records.parquet", "_manifest.json"}
    assert not list(output.parent.glob(f".{output.name}.*.pending"))
    assert not (arguments["root"] / "data/runtime/heavy-job.owner.json").exists()
    inspection.reset_mock()
    before = {path.name: file_sha256(path) for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    assert before == {path.name: file_sha256(path) for path in output.iterdir()}


@pytest.mark.parametrize("runtime_kind", ["default", "relative", "absolute"])
def test_busy_shared_lease_prevents_guards_and_source_reads(
    arguments: dict[str, Any], inspection: Mock, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, runtime_kind: str,
) -> None:
    runtime = arguments["root"] / "data/runtime"
    if runtime_kind == "relative":
        monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", "custom/runtime")
        runtime = arguments["root"] / "custom/runtime"
    elif runtime_kind == "absolute":
        runtime = tmp_path / "external-runtime"
        monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    guard = Mock()
    monkeypatch.setattr(inventory, "_guard", guard)
    with heavy_job_lease("test-owner", runtime_dir=runtime):
        owner_before = (runtime / "heavy-job.owner.json").read_bytes()
        with pytest.raises(HeavyJobBusyError):
            inventory.publish_saved_content_inventory(**arguments)
        assert (runtime / "heavy-job.owner.json").read_bytes() == owner_before
    guard.assert_not_called()
    inspection.assert_not_called()
    _assert_unpublished(arguments)
    assert not list(elsewhere.iterdir())


def test_lease_and_memory_guards_span_read_write_and_atomic_publish(
    arguments: dict[str, Any], inspection: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _output(arguments)
    owner = arguments["root"] / "data/runtime/heavy-job.owner.json"
    events: list[str] = []

    def memory_budget(**kwargs: Any) -> None:
        assert kwargs["hard_budget_gib"] == 5.0
        assert kwargs["headroom_gib"] == 0.75
        assert json.loads(owner.read_text(encoding="utf-8"))["command"] == "inspect-saved-issuer-content"
        events.append("guard")

    def inspect(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        assert events == ["guard"]
        events.append("inspect")
        rows, summary = inspection.return_value
        return rows, summary

    original_write = pd.DataFrame.to_parquet
    original_rename = Path.rename

    def write(frame: pd.DataFrame, path: Path, **kwargs: Any) -> Any:
        assert events == ["guard", "inspect", "guard"]
        assert owner.exists() and not output.exists()
        assert path.parent.parent == output.parent and path.parent != output
        events.append("write")
        return original_write(frame, path, **kwargs)

    def rename(path: Path, target: Path) -> Path:
        assert events == ["guard", "inspect", "guard", "write", "guard"]
        assert owner.exists() and not output.exists()
        assert target == output
        assert (path / "_manifest.json").is_file()
        assert (path / "records.parquet").is_file()
        events.append("publish")
        return original_rename(path, target)

    monkeypatch.setattr(inventory, "assert_memory_budget", memory_budget)
    inspection.side_effect = inspect
    monkeypatch.setattr(pd.DataFrame, "to_parquet", write)
    monkeypatch.setattr(Path, "rename", rename)
    inventory.publish_saved_content_inventory(**arguments)
    assert events == ["guard", "inspect", "guard", "write", "guard", "publish"]
    assert not owner.exists()


@pytest.mark.parametrize("output", ["../escaped", "data/reports/inventory", "data/research",
    "data/research/existing-authority/nested", "data/research/../raw/inventory"])
def test_output_outside_new_direct_research_child_is_rejected(
    arguments: dict[str, Any], inspection: Mock, output: str,
) -> None:
    arguments["output"] = Path(output)
    with pytest.raises(DataReadinessError):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    assert not _output(arguments).exists()


@pytest.mark.parametrize("directory", [False, True])
def test_existing_output_is_preserved_without_source_read(
    arguments: dict[str, Any], inspection: Mock, directory: bool,
) -> None:
    output = _output(arguments)
    output.parent.mkdir(parents=True)
    if directory:
        output.mkdir()
    marker = output / "unrelated.bin" if directory else output
    marker.write_bytes(b"existing immutable evidence")
    with pytest.raises(FileExistsError):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    assert marker.read_bytes() == b"existing immutable evidence"


@pytest.mark.parametrize("pin_name", ["event_artifact", "event_manifest"])
def test_source_inside_output_is_rejected_before_inspection(
    arguments: dict[str, Any], inspection: Mock, pin_name: str,
) -> None:
    arguments[pin_name] = SourcePin(path="data/research/content-inventory/source.json", sha256="a" * 64)
    with pytest.raises(DataReadinessError, match="overlaps source"):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    _assert_unpublished(arguments)


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_memory_pressure_never_publishes_partial_artifact(
    arguments: dict[str, Any], inspection: Mock, monkeypatch: pytest.MonkeyPatch, phase: int,
) -> None:
    high_memory = SystemMemory(total_bytes=20 * 1024**3, available_bytes=2 * 1024**3)
    snapshots = Mock(side_effect=[HEALTHY_MEMORY] * phase + [high_memory])
    monkeypatch.setattr(inventory, "system_memory_snapshot", snapshots)
    with pytest.raises(MemoryBudgetError, match="below 90 percent"):
        inventory.publish_saved_content_inventory(**arguments)
    assert inspection.call_count == (0 if phase == 0 else 1)
    assert snapshots.call_count == phase + 1
    _assert_unpublished(arguments)


@pytest.mark.parametrize("kind", ["unmeasured_system", "process_budget"])
def test_memory_guard_failure_prevents_source_read(
    arguments: dict[str, Any], inspection: Mock, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    if kind == "unmeasured_system":
        monkeypatch.setattr(inventory, "system_memory_snapshot", Mock(return_value=None))
    else:
        monkeypatch.setattr(inventory, "assert_memory_budget", Mock(side_effect=MemoryBudgetError("process limit")))
    with pytest.raises(MemoryBudgetError):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    _assert_unpublished(arguments)


@pytest.mark.parametrize("phase", ["source", "parquet", "manifest", "rename"])
def test_partial_failure_cleans_only_own_staging(
    arguments: dict[str, Any], inspection: Mock, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    output = _output(arguments)
    sibling = output.with_name("unrelated-authority")
    sibling.mkdir(parents=True)
    sibling_file = sibling / "records.parquet"
    sibling_file.write_bytes(b"unrelated saved evidence")
    other_staging = output.with_name(f".{output.name}.other-owner.pending")
    other_staging.mkdir()
    other_file = other_staging / "keep.bin"
    other_file.write_bytes(b"another owner's work")
    error = OSError(f"test-only {phase} failure")
    if phase == "source":
        inspection.side_effect = error
    elif phase == "parquet":
        def partial_write(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
            path.write_bytes(b"partial parquet")
            raise error
        monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_write)
    elif phase == "manifest":
        def partial_manifest(path: Path, value: Any) -> None:
            path.write_bytes(b'{"partial":')
            raise error
        monkeypatch.setattr(inventory, "write_json_object", partial_manifest)
    else:
        monkeypatch.setattr(Path, "rename", Mock(side_effect=error))
    with pytest.raises(OSError, match=f"test-only {phase} failure"):
        inventory.publish_saved_content_inventory(**arguments)
    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.*.pending")) == [other_staging]
    assert sibling_file.read_bytes() == b"unrelated saved evidence"
    assert other_file.read_bytes() == b"another owner's work"
    assert not (arguments["root"] / "data/runtime/heavy-job.owner.json").exists()


@pytest.mark.parametrize(("field", "value"), [("start_utc", pd.Timestamp("2019-07-09")),
    ("cutoff_utc", pd.Timestamp("2024-05-28")), ("start_utc", pd.NaT), ("cutoff_utc", pd.NaT),
    ("start_utc", "2019-07-09T00:00:00Z"), ("start_utc", CUTOFF + pd.Timedelta(seconds=1)),
    ("start_utc", START - pd.Timedelta(1, "ns")), ("cutoff_utc", CUTOFF + pd.Timedelta(1, "ns"))])
def test_api_rejects_invalid_dates_without_reading(
    arguments: dict[str, Any], inspection: Mock, field: str, value: Any,
) -> None:
    arguments[field] = value
    with pytest.raises(DataReadinessError):
        inventory.publish_saved_content_inventory(**arguments)
    inspection.assert_not_called()
    _assert_unpublished(arguments)


@pytest.mark.parametrize("custom_dates_and_root", [False, True])
def test_cli_wires_pins_dates_and_compact_report(
    monkeypatch: pytest.MonkeyPatch, custom_dates_and_root: bool,
) -> None:
    report = dict(status="complete_inventory_only", rows=2, manifest_sha256="f" * 64,
        training_eligible=False, promotion_eligible=False, serving_eligible=False,
        inspection={"test_only_orchestration": True}, implementation_files={"test-only": "e" * 64})
    publisher = Mock(return_value=report)
    monkeypatch.setattr(commands, "publish_saved_content_inventory", publisher)
    args = _cli_arguments()
    start, cutoff, root = START, CUTOFF, Path(".")
    if custom_dates_and_root:
        args += ["--root", "custom-root", "--start-utc", "2020-01-01T01:00:00+01:00",
            "--cutoff-utc", "2023-12-31T22:00:00Z"]
        start, cutoff, root = pd.Timestamp("2020-01-01T00:00:00Z"), pd.Timestamp("2023-12-31T22:00:00Z"), Path("custom-root")
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.exception
    publisher.assert_called_once_with(root=root,
        event_artifact=SourcePin(path="data/raw/events.parquet", sha256="a" * 64),
        event_manifest=SourcePin(path="data/raw/events.manifest.json", sha256="b" * 64),
        security_id="test-security", ticker="AAA", output=Path("data/research/content-inventory"),
        start_utc=start, cutoff_utc=cutoff)
    assert json.loads(result.stdout) == {key: report[key] for key in (
        "status", "rows", "manifest_sha256", "training_eligible", "serving_eligible")}
    assert len(result.stdout.strip().splitlines()) == 1


@pytest.mark.parametrize(("error", "exit_code"), [(HeavyJobBusyError("lease busy"), 75),
    (DataReadinessError("source not ready"), 2), (OSError("read failure"), 2),
    (ValueError("invalid value"), 2), (MemoryBudgetError("memory limit"), 2),
    (FileExistsError("immutable output exists"), 2)])
def test_cli_expected_failure_codes(monkeypatch: pytest.MonkeyPatch, error: Exception, exit_code: int) -> None:
    publisher = Mock(side_effect=error)
    monkeypatch.setattr(commands, "publish_saved_content_inventory", publisher)
    result = CliRunner().invoke(_app(), _cli_arguments())
    assert result.exit_code == exit_code, result.exception
    publisher.assert_called_once()
    assert str(error) in result.output
    assert "complete_inventory_only" not in result.output


@pytest.mark.parametrize("flag", ["--event-sha256", "--manifest-sha256", "--start-utc", "--cutoff-utc"])
def test_cli_malformed_pin_or_date_never_reaches_publisher(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    publisher = Mock()
    monkeypatch.setattr(commands, "publish_saved_content_inventory", publisher)
    args = _cli_arguments()
    if flag in args:
        args[args.index(flag) + 1] = "not-a-sha256"
    else:
        args += [flag, "not-a-timestamp"]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 2, result.exception
    publisher.assert_not_called()


@pytest.mark.parametrize(("start", "cutoff"), [("2019-07-09", "2024-05-28T22:00:00Z"),
    ("2019-07-09T00:00:00Z", "2024-05-28"), ("NaT", "2024-05-28T22:00:00Z"),
    ("2024-05-29T00:00:00Z", "2024-05-28T22:00:00Z")])
def test_cli_invalid_date_interval_fails_before_source_read(
    arguments: dict[str, Any], inspection: Mock, start: str, cutoff: str,
) -> None:
    result = CliRunner().invoke(_app(), [*_cli_arguments(), "--root", str(arguments["root"]),
        "--start-utc", start, "--cutoff-utc", cutoff])
    assert result.exit_code == 2, result.exception
    inspection.assert_not_called()
    _assert_unpublished(arguments)


def test_research_app_exposes_registered_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research_cli import app

    publisher = Mock(return_value=dict(status="complete_inventory_only", rows=0,
        manifest_sha256="f" * 64, training_eligible=False, serving_eligible=False))
    monkeypatch.setattr(commands, "publish_saved_content_inventory", publisher)
    result = CliRunner().invoke(app, _cli_arguments())
    assert result.exit_code == 0, result.exception
    publisher.assert_called_once()
    assert json.loads(result.stdout)["rows"] == 0


def test_default_window_is_the_initial_fit_authority_window() -> None:
    assert (START, CUTOFF) == (FIRST, LAST_INITIAL_FIT_CUTOFF)


def test_absolute_source_paths_are_recorded_relative_to_root(arguments: dict[str, Any], inspection: Mock) -> None:
    for name in ("event_artifact", "event_manifest"):
        pin = arguments[name]
        arguments[name] = SourcePin(path=str(arguments["root"] / pin.path), sha256=pin.sha256)
    report = inventory.publish_saved_content_inventory(**arguments)
    assert report["event_artifact"] == {"path": "data/raw/events.parquet", "sha256": "a" * 64}
    assert report["event_manifest"] == {"path": "data/raw/events.manifest.json", "sha256": "b" * 64}


def test_manifest_records_canonical_query_ticker(arguments: dict[str, Any], inspection: Mock) -> None:
    arguments["ticker"] = "brk.b"
    assert inventory.publish_saved_content_inventory(**arguments)["query_ticker"] == "BRK-B"


@pytest.mark.parametrize("resume", [None, "c" * 64])
def test_cohort_cli_wires_config_resume_and_compact_report(monkeypatch: pytest.MonkeyPatch, resume: str | None) -> None:
    report = dict(status="complete_inventory_only", records_rows=3, manifest_sha256="f" * 64, training_eligible=False,
                  serving_eligible=False, totals={"test_only": 1})
    publisher = Mock(return_value=report)
    monkeypatch.setattr(commands, "publish_cohort_content_inventory", publisher)
    args = ["inspect-issuer-content-cohort", "--config", "configs/cohort.json", "--config-sha256", "d" * 64,
            "--output", "data/research/cohort"] + (["--resume-checkpoint-sha256", resume] if resume else [])
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.exception
    publisher.assert_called_once_with(root=Path("."), config=Path("configs/cohort.json"), config_sha256="d" * 64,
                                      output=Path("data/research/cohort"), resume_checkpoint_sha256=resume)
    assert json.loads(result.stdout) == {key: report[key] for key in (
        "status", "records_rows", "manifest_sha256", "training_eligible", "serving_eligible")}


@pytest.mark.parametrize(("error", "exit_code"), [(HeavyJobBusyError("lease busy"), 75),
    (DataReadinessError("parity differs"), 2), (OSError("read failure"), 2), (MemoryBudgetError("memory limit"), 2)])
def test_cohort_cli_expected_failure_codes(monkeypatch: pytest.MonkeyPatch, error: Exception, exit_code: int) -> None:
    monkeypatch.setattr(commands, "publish_cohort_content_inventory", Mock(side_effect=error))
    result = CliRunner().invoke(_app(), ["inspect-issuer-content-cohort", "--config", "c.json", "--config-sha256",
                                         "d" * 64, "--output", "data/research/cohort"])
    assert result.exit_code == exit_code and str(error) in result.output


def test_proof_cli_wires_config_and_compact_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = dict(status="complete", totals={"proof_rows": 2}, manifest_sha256="f" * 64, training_eligible=False,
                  serving_eligible=False, source_files={"test_only": "a" * 64})
    publisher = Mock(return_value=report)
    monkeypatch.setattr(commands, "publish_legacy_query_identity_proofs", publisher)
    result = CliRunner().invoke(_app(), ["prove-legacy-query-identities", "--config", "configs/proofs.json",
                                         "--config-sha256", "d" * 64, "--output", "data/research/proofs"])
    assert result.exit_code == 0, result.exception
    publisher.assert_called_once_with(root=Path("."), config=Path("configs/proofs.json"), config_sha256="d" * 64,
                                      output=Path("data/research/proofs"))
    assert json.loads(result.stdout) == {key: report[key] for key in (
        "status", "totals", "manifest_sha256", "training_eligible", "serving_eligible")}


@pytest.mark.parametrize(("error", "exit_code"), [(HeavyJobBusyError("lease busy"), 75),
    (DataReadinessError("lineage differs"), 2), (FileExistsError("immutable proofs"), 2)])
def test_proof_cli_expected_failure_codes(monkeypatch: pytest.MonkeyPatch, error: Exception, exit_code: int) -> None:
    monkeypatch.setattr(commands, "publish_legacy_query_identity_proofs", Mock(side_effect=error))
    result = CliRunner().invoke(_app(), ["prove-legacy-query-identities", "--config", "c.json", "--config-sha256",
                                         "d" * 64, "--output", "data/research/proofs"])
    assert result.exit_code == exit_code and str(error) in result.output
